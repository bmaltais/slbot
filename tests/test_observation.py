"""Observation coverage: what the model can see about food, walls and snakes."""
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sector_layout as L
from food_sense import FOOD_SECTOR_MASS_CAP, squash_mass
from slither_env import BOOST_SPEED_THRESHOLD, SlitherEnv
from ws_engine import Food, GameState, Snake, SnakePoint, SlitherWSClient, snake_scale_from_parts
import worker_session
import config


def _make_env(size=160):
    with patch('slither_env._create_browser', return_value=MagicMock()):
        env = SlitherEnv(headless=True, nickname="T", matrix_size=size, view_plus=False)
    env.browser = MagicMock()
    return env


def _scene(mx=21600.0, my=21600.0, ang=0.0, foods=(), enemies=(), sp=5.7, view_radius=400):
    return {
        'dead': False, 'valid': True,
        'self': {'x': mx, 'y': my, 'ang': ang, 'len': 30, 'sc': 1.0, 'sp': sp, 'pts': []},
        'foods': list(foods), 'enemies': list(enemies),
        'map_radius': 21600, 'map_center_x': 21600, 'map_center_y': 21600,
        'view_radius': view_radius, 'gsc': 1.0,
    }


class TestLayoutConsistency(unittest.TestCase):
    def test_every_consumer_agrees_on_sector_dim(self):
        self.assertEqual(worker_session.SECTOR_DIM, L.SECTOR_DIM)
        self.assertEqual(config.ModelConfig().sector_dim, L.SECTOR_DIM)
        env = _make_env(16)
        self.assertEqual(env._matrix_zeros()['sectors'].shape, (L.SECTOR_DIM,))
        self.assertEqual(env._compute_sectors({}).shape, (L.SECTOR_DIM,))
        self.assertEqual(env._compute_sectors(_scene()).shape, (L.SECTOR_DIM,))

    def test_legacy_block_indices_are_frozen(self):
        # agent._check_reflexes reads these positions directly
        self.assertEqual(L.OBSTACLE_SCORE, 24)
        self.assertEqual(L.OBSTACLE_TYPE, 48)
        self.assertEqual(L.ENEMY_APPROACH, 72)
        self.assertEqual(L.G_WALL_DIST, 96)
        self.assertEqual(L.LEGACY_DIM, 99)
        bands = [L.FOOD_SCORE, L.OBSTACLE_SCORE, L.OBSTACLE_TYPE, L.ENEMY_APPROACH,
                 L.FOOD_MASS, L.FOOD_DIST, L.WALL_DIST, L.ENEMY_SIZE]
        for b in bands:
            self.assertLessEqual(b + L.NUM_SECTORS, L.SECTOR_DIM)
        self.assertEqual(L.G_TARGET_COS + 1, L.SECTOR_DIM)

    def test_empty_scene_has_only_type_sentinels(self):
        s = _make_env(16)._compute_sectors(_scene())
        self.assertTrue(np.all(s[L.sector_band(L.OBSTACLE_TYPE)] == -1.0))
        rest = np.delete(s, np.r_[L.sector_band(L.OBSTACLE_TYPE), L.G_WALL_DIST, L.G_LENGTH, L.G_SPEED])
        self.assertTrue(np.all(rest == 0.0))


class TestFoodBands(unittest.TestCase):
    """A dead snake's remains far away must look different from a crumb nearby."""

    def setUp(self):
        self.env = _make_env(16)

    def test_far_pile_and_near_crumb_are_separable(self):
        crumb = self.env._compute_sectors(_scene(foods=[[21700.0, 21600.0, 1.0]]))
        pile = self.env._compute_sectors(_scene(
            foods=[[23400.0 + (i % 10) * 8, 21600.0 + (i // 10) * 8, 6.0] for i in range(60)]
        ))
        # Heading east (ang=0) -> both sit in sector 0 (ahead)
        self.assertGreater(crumb[L.FOOD_SCORE], 0.0)
        self.assertGreater(pile[L.FOOD_SCORE], 0.0)
        # mass band: pile >> crumb, independent of distance
        self.assertGreater(pile[L.FOOD_MASS], crumb[L.FOOD_MASS] * 3)
        self.assertAlmostEqual(float(crumb[L.FOOD_MASS]), squash_mass(1.0, FOOD_SECTOR_MASS_CAP), places=5)
        self.assertAlmostEqual(float(pile[L.FOOD_MASS]), squash_mass(360.0, FOOD_SECTOR_MASS_CAP), places=5)
        # distance band: crumb is close, pile is far
        self.assertAlmostEqual(float(crumb[L.FOOD_DIST]), 1.0 - 100.0 / 2000.0, places=4)
        self.assertLess(pile[L.FOOD_DIST], 0.15)
        self.assertGreater(crumb[L.FOOD_DIST], 0.9)
        # total food global follows mass
        self.assertGreater(pile[L.G_FOOD_TOTAL], crumb[L.G_FOOD_TOTAL])
        # nothing leaks into other sectors
        for band in (L.FOOD_SCORE, L.FOOD_MASS, L.FOOD_DIST):
            self.assertTrue(np.all(pile[band + 1:band + L.NUM_SECTORS] == 0.0))

    def test_food_dist_is_mass_weighted(self):
        s = self.env._compute_sectors(_scene(foods=[[21700.0, 21600.0, 1.0], [23500.0, 21600.0, 9.0]]))
        mean_d = (100.0 * 1.0 + 1900.0 * 9.0) / 10.0
        self.assertAlmostEqual(float(s[L.FOOD_DIST]), 1.0 - mean_d / 2000.0, places=4)

    def test_food_beyond_scope_ignored(self):
        s = self.env._compute_sectors(_scene(foods=[[23700.0, 21600.0, 50.0]]))
        self.assertEqual(float(s[L.FOOD_MASS]), 0.0)
        self.assertEqual(float(s[L.FOOD_DIST]), 0.0)
        self.assertEqual(float(s[L.G_FOOD_TOTAL]), 0.0)

    def test_sectors_follow_heading(self):
        # Food due north of the snake; snake heading north (ang=-pi/2, y-down) -> ahead.
        north = _scene(ang=-math.pi / 2, foods=[[21600.0, 21100.0, 2.0]])
        s = self.env._compute_sectors(north)
        self.assertGreater(s[L.FOOD_MASS + 0], 0.0)
        # Same food, heading east -> food is 90° left -> sector 18
        east = _scene(ang=0.0, foods=[[21600.0, 21100.0, 2.0]])
        s = self.env._compute_sectors(east)
        self.assertGreater(s[L.FOOD_MASS + 18], 0.0)
        self.assertEqual(float(s[L.FOOD_MASS + 0]), 0.0)


class TestTargetGlobals(unittest.TestCase):
    def test_matrix_pass_publishes_locked_target_to_sectors(self):
        env = _make_env(32)
        # Rich pile 1500 ahead (off the 1000-unit crop) plus a crumb behind.
        foods = [[23100.0 + (i % 5) * 10, 21600.0 + (i // 5) * 10, 8.0] for i in range(25)]
        foods.append([21500.0, 21600.0, 1.0])
        data = _scene(foods=foods)
        env._process_data_to_matrix(data)
        s = env._compute_sectors(data)
        self.assertGreater(s[L.G_TARGET_MASS], squash_mass(100.0, FOOD_SECTOR_MASS_CAP))
        self.assertAlmostEqual(float(s[L.G_TARGET_DIST]), 1.0 - 1500.0 / 2000.0, places=2)
        self.assertAlmostEqual(float(s[L.G_TARGET_SIN]), 0.0, places=3)
        self.assertAlmostEqual(float(s[L.G_TARGET_COS]), 1.0, places=3)

    def test_target_cache_cleared_on_dead_frame(self):
        env = _make_env(16)
        env._process_data_to_matrix(_scene(foods=[[21800.0, 21600.0, 5.0]]))
        self.assertIsNotNone(env._matrix_food_target)
        env._process_data_to_matrix({'dead': True})
        self.assertIsNone(env._matrix_food_target)
        s = env._compute_sectors(_scene())
        self.assertEqual(float(s[L.G_TARGET_DIST]), 0.0)


class TestWallBand(unittest.TestCase):
    def setUp(self):
        self.env = _make_env(16)

    def _ref_wall_scores(self, mx, my, ang):
        """Scalar per-sector ray/circle reference (the pre-vectorized loop)."""
        ns = L.NUM_SECTORS
        sa = 2 * math.pi / ns
        out = np.zeros(ns)
        for si in range(ns):
            ea = si * sa + sa / 2
            rx, ry = math.sin(ea), -math.cos(ea)
            dx = -math.sin(ang) * rx - math.cos(ang) * ry
            dy = math.cos(ang) * rx - math.sin(ang) * ry
            ocx, ocy = mx - 21600.0, my - 21600.0
            b = 2.0 * (ocx * dx + ocy * dy)
            c = ocx * ocx + ocy * ocy - 21600.0 ** 2
            disc = b * b - 4.0 * c
            if disc >= 0:
                sq = math.sqrt(disc)
                t1, t2 = (-b - sq) / 2.0, (-b + sq) / 2.0
                t = t2 if t2 > 0 else t1
                if 0 < t < 2000.0:
                    out[si] = 1.0 - t / 2000.0
        return out

    def test_vectorized_rays_match_scalar_reference(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            r = rng.uniform(19000.0, 21590.0)
            th = rng.uniform(-math.pi, math.pi)
            mx, my = 21600.0 + r * math.cos(th), 21600.0 + r * math.sin(th)
            ang = rng.uniform(-math.pi, math.pi)
            data = _scene(mx=mx, my=my, ang=ang)
            self.env._update_from_game_data(data)
            s = self.env._compute_sectors(data)
            ref = self._ref_wall_scores(mx, my, ang)
            np.testing.assert_allclose(s[L.sector_band(L.WALL_DIST)], ref, atol=1e-5)
            np.testing.assert_allclose(s[L.sector_band(L.OBSTACLE_SCORE)], ref, atol=1e-5)
            self.assertTrue(np.all(s[L.sector_band(L.OBSTACLE_TYPE)][ref > 0] == 0.0))

    def test_wall_band_survives_a_closer_enemy(self):
        # Wall 300 ahead, enemy body 100 ahead: obstacle band shows the enemy,
        # wall band still says where the wall is.
        data = _scene(mx=21600.0 + 21300.0, ang=0.0,
                      enemies=[{'x': 30000.0, 'y': 30000.0, 'sc': 1.0, 'ang': 0.0, 'sp': 5.7,
                                'pts': [[21600.0 + 21400.0, 21600.0]]}])
        self.env._update_from_game_data(data)
        s = self.env._compute_sectors(data)
        # sector 0's centre ray is 7.5° off the heading, so slightly > 300
        self.assertAlmostEqual(float(s[L.WALL_DIST]), 1.0 - 300.0 / 2000.0, places=2)
        self.assertAlmostEqual(float(s[L.OBSTACLE_SCORE]), 1.0 - (100.0 - 14.5) / 2000.0, places=4)
        self.assertEqual(float(s[L.WALL_DIST + 12]), 0.0)  # wall behind is ~41000 away

    def test_wall_rasterized_out_to_grid_edge_regardless_of_camera(self):
        # Wall 900 units ahead, camera radius 400: old code (view_size*1.5=600)
        # skipped the wall; the 1000-unit grid must show it.
        env = _make_env(160)
        data = _scene(mx=21600.0 + 20700.0, ang=0.0, view_radius=400)
        m = env._process_data_to_matrix(data)
        # Egocentric: ahead = up. 12.5 units/px, wall margin 200 -> band starts ~row 24.
        self.assertEqual(int(m[1, 5, 80]), 255)
        self.assertEqual(int(m[1, 60, 80]), 0)
        self.assertEqual(int(m[1, 80, 80]), 0)      # head cell clear


class TestEnemyBands(unittest.TestCase):
    def setUp(self):
        self.env = _make_env(16)

    def test_enemy_size_band_reports_scale_of_nearest(self):
        big = {'x': 22200.0, 'y': 21600.0, 'sc': 3.0, 'ang': math.pi, 'sp': 5.7, 'pts': []}
        small_closer = {'x': 21900.0, 'y': 21600.0, 'sc': 1.0, 'ang': math.pi, 'sp': 5.7, 'pts': []}
        s = self.env._compute_sectors(_scene(enemies=[big]))
        self.assertAlmostEqual(float(s[L.ENEMY_SIZE]), 0.5, places=5)
        self.assertEqual(float(s[L.OBSTACLE_TYPE]), 1.0)
        s = self.env._compute_sectors(_scene(enemies=[big, small_closer]))
        self.assertAlmostEqual(float(s[L.ENEMY_SIZE]), 1.0 / 6.0, places=5)
        self.assertTrue(np.all(s[L.ENEMY_SIZE + 1:L.ENEMY_SIZE + L.NUM_SECTORS] == 0.0))

    def test_body_only_enemy_still_has_size(self):
        e = {'x': 40000.0, 'y': 40000.0, 'sc': 4.5, 'ang': 0.0, 'sp': 5.7,
             'pts': [[22000.0, 21600.0]]}
        s = self.env._compute_sectors(_scene(enemies=[e]))
        self.assertAlmostEqual(float(s[L.ENEMY_SIZE]), 0.75, places=5)
        self.assertEqual(float(s[L.OBSTACLE_TYPE]), 0.0)

    def test_wall_does_not_get_an_enemy_size(self):
        data = _scene(mx=21600.0 + 21300.0, ang=0.0)
        self.env._update_from_game_data(data)
        s = self.env._compute_sectors(data)
        self.assertGreater(float(s[L.OBSTACLE_SCORE]), 0.0)
        self.assertEqual(float(s[L.ENEMY_SIZE]), 0.0)


class TestSelfGlobals(unittest.TestCase):
    def test_boost_flag(self):
        env = _make_env(16)
        self.assertEqual(float(env._compute_sectors(_scene(sp=5.7))[L.G_BOOST]), 0.0)
        self.assertEqual(float(env._compute_sectors(_scene(sp=14.0))[L.G_BOOST]), 1.0)
        self.assertEqual(float(env._compute_sectors(_scene(sp=BOOST_SPEED_THRESHOLD))[L.G_BOOST]), 1.0)


class TestWsSnakeScale(unittest.TestCase):
    def test_scale_formula(self):
        self.assertEqual(snake_scale_from_parts(0), 1.0)
        self.assertEqual(snake_scale_from_parts(2), 1.0)
        self.assertAlmostEqual(snake_scale_from_parts(108), 2.0)
        self.assertEqual(snake_scale_from_parts(10000), 6.0)

    def test_game_data_scale_comes_from_body_parts_not_fullness(self):
        client = SlitherWSClient.__new__(SlitherWSClient)
        client.state = GameState()
        client.state.playing = True
        client.state.dead = False
        client.state.my_id = 1
        client.state.gsc = 0.9
        me = Snake(id=1, x=21600.0, y=21600.0, fam=0.95)
        me.pts = [SnakePoint(x=21600.0 - i * 10.0, y=21600.0) for i in range(320)]
        enemy = Snake(id=2, x=21800.0, y=21600.0, fam=0.0)
        enemy.pts = [SnakePoint(x=21800.0 + i * 10.0, y=21600.0) for i in range(20)]
        client.state.snakes[1] = me
        client.state.snakes[2] = enemy
        client.state.foods[1] = Food(id=1, x=21700.0, y=21600.0, size=1.0)
        data = client._build_game_data()
        self.assertAlmostEqual(data['self']['sc'], 4.0, places=5)     # 1 + 318/106
        self.assertAlmostEqual(data['enemies'][0]['sc'], 1.0 + 18.0 / 106.0, places=5)


if __name__ == "__main__":
    unittest.main()
