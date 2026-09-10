import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from food_sense import (
    FOOD_CHANNEL_LOG_CAP,
    FOOD_KEEP_DIST_BIAS,
    FOOD_SENSE_RANGE,
    MAX_FOODS,
    PREY_DEFAULT_SIZE,
    best_food_target,
    cluster_eat_bonus,
    cluster_foods,
    eaten_food_mass,
    food_draw_radius_px,
    js_collect_foods,
    select_visible_foods,
    squash_mass,
)
from slither_env import SlitherEnv
from ws_engine import Food, GameState, Prey, SlitherWSClient, Snake


class TestFoodSenseHelpers(unittest.TestCase):
    def test_squash_ranks_piles_above_crumbs(self):
        crumb = squash_mass(1.0, FOOD_CHANNEL_LOG_CAP)
        pile = squash_mass(20.0, FOOD_CHANNEL_LOG_CAP)
        self.assertGreater(pile, crumb * 2)
        self.assertLessEqual(pile, 1.0)
        self.assertEqual(squash_mass(0.0, FOOD_CHANNEL_LOG_CAP), 0.0)

    def test_select_keeps_food_beyond_camera_inside_sense_range(self):
        mx, my = 21600.0, 21600.0
        far = (mx + 800.0, my, 1.0)
        too_far = (mx + 2500.0, my, 1.0)
        kept = select_visible_foods([far, too_far], mx, my)
        coords = {(round(x), round(y)) for x, y, _sz in kept}
        self.assertIn((round(far[0]), round(far[1])), coords)
        self.assertNotIn((round(too_far[0]), round(too_far[1])), coords)

    def test_select_prefers_mass_when_truncating(self):
        mx, my = 0.0, 0.0
        crumbs = [(10.0, float(i), 1.0) for i in range(50)]
        remains = [(400.0, 0.0, 12.0)]
        kept = select_visible_foods(crumbs + remains, mx, my, max_foods=20)
        sizes = [sz for _x, _y, sz in kept]
        self.assertIn(12.0, sizes)
        self.assertEqual(len(kept), 20)

    def test_best_target_is_cluster_not_nearest_crumb(self):
        mx, my = 0.0, 0.0
        crumb = [[30.0, 0.0, 1.0]]
        pile = [[400.0 + i, 0.0, 1.0] for i in range(25)]
        target = best_food_target(crumb + pile, mx, my)
        self.assertIsNotNone(target)
        cx, cy, dist, mass = target
        self.assertGreater(mass, 10.0)
        self.assertGreater(cx, 300.0)
        self.assertGreater(dist, 100.0)

    def test_cluster_bins_sum_mass(self):
        mx, my = 0.0, 0.0
        foods = [[120.0, 0.0, 2.0], [125.0, 4.0, 3.0]]
        clusters = cluster_foods(foods, mx, my, cell=120.0)
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0][3], 5.0)

    def test_js_collect_scans_all_foods_to_sense_range(self):
        js = js_collect_foods()
        self.assertIn(str(int(FOOD_SENSE_RANGE)), js)
        self.assertIn(str(FOOD_KEEP_DIST_BIAS), js)
        self.assertNotIn("MAX_FOODS * 2", js)
        self.assertIn("for (var i = 0; i < window.foods.length; i++)", js)
        self.assertIn("window.preys", js)
        self.assertIn("foodList = []", js)

    def test_draw_radius_grows_with_size(self):
        scale = 160 / 2000.0
        crumb = food_draw_radius_px(1.0, scale)
        pile = food_draw_radius_px(12.0, scale)
        self.assertGreater(pile, crumb * 2)
        self.assertGreaterEqual(crumb, 0.85)

    def test_cluster_eat_bonus_ignores_crumb_and_pays_piles(self):
        self.assertEqual(cluster_eat_bonus(1.0, 3.0), 0.0)
        self.assertEqual(cluster_eat_bonus(0.0, 3.0), 0.0)
        pile = cluster_eat_bonus(20.0, 3.0)
        small = cluster_eat_bonus(4.0, 3.0)
        self.assertGreater(pile, small)
        self.assertGreater(small, 0.0)
        capped = cluster_eat_bonus(400.0, 3.0)
        self.assertAlmostEqual(capped, cluster_eat_bonus(41.0, 3.0))

    def test_eaten_mass_counts_vanished_food_on_path(self):
        pre = [[10.0, 0.0, 1.0], [12.0, 1.0, 3.0], [400.0, 0.0, 8.0]]
        post = [[400.0, 0.0, 8.0]]
        mass, n = eaten_food_mass(pre, post, 0.0, 0.0, 20.0, 0.0, eat_radius=50.0)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(mass, 4.0)

    def test_eaten_mass_ignores_food_eaten_by_someone_else(self):
        pre = [[10.0, 0.0, 1.0], [400.0, 0.0, 12.0]]
        post = [[10.0, 0.0, 1.0]]
        mass, n = eaten_food_mass(pre, post, 0.0, 0.0, 20.0, 0.0, eat_radius=50.0)
        self.assertEqual(n, 0)
        self.assertEqual(mass, 0.0)

    def test_eaten_mass_still_matches_prey_that_slid(self):
        pre = [[100.0, 0.0, 10.0]]
        post = [[130.0, 0.0, 10.0]]  # moved 30 units, still present
        mass, n = eaten_food_mass(pre, post, 0.0, 0.0, 20.0, 0.0, eat_radius=50.0)
        self.assertEqual(n, 0)
        self.assertEqual(mass, 0.0)


class TestFoodObservation(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('slither_env._create_browser', return_value=MagicMock())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = SlitherEnv(headless=True, nickname="TestBot", matrix_size=84, view_plus=False)

    def _data(self, foods):
        return {
            'dead': False,
            'self': {'x': 21600.0, 'y': 21600.0, 'len': 10, 'ang': 0.0, 'sp': 5.7},
            'foods': foods,
            'enemies': [],
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'view_radius': 400,
            'gsc': 1.0,
        }

    def test_food_channel_pile_brighter_than_crumb(self):
        crumb_data = self._data([[21700.0, 21600.0, 1.0]])
        pile_data = self._data([[21700.0 + (i % 3), 21600.0 + (i // 3), 1.0] for i in range(18)])
        crumb = self.env._process_data_to_matrix(crumb_data)[0]
        pile = self.env._process_data_to_matrix(pile_data)[0]
        self.assertGreater(float(pile.max()), float(crumb.max()))
        self.assertLessEqual(float(pile.max()), 1.0)

    def test_sector_score_sums_cluster_mass(self):
        crumb = self.env._compute_sectors(self._data([[21800.0, 21600.0, 1.0]]))
        pile = self.env._compute_sectors(self._data(
            [[21800.0, 21600.0 + i, 1.0] for i in range(16)]
        ))
        self.assertGreater(float(pile[:24].max()), float(crumb[:24].max()))
        self.assertLessEqual(float(pile[:24].max()), 1.0)

    def test_food_beyond_camera_still_paints(self):
        # 800 units east — outside ~400 camera, inside 1000 matrix / 2000 sense
        data = self._data([[22400.0, 21600.0, 4.0]])
        matrix = self.env._process_data_to_matrix(data)
        self.assertGreater(float(matrix[0].max()), 0.0)

    def test_large_pellet_brighter_and_wider_than_crumb(self):
        crumb = self.env._process_data_to_matrix(self._data([[21700.0, 21600.0, 1.0]]))[0]
        big = self.env._process_data_to_matrix(self._data([[21700.0, 21600.0, 12.0]]))[0]
        self.assertGreater(float(big.max()), float(crumb.max()))
        self.assertGreater(int((big > 0.05).sum()), int((crumb > 0.05).sum()))
        self.assertLessEqual(float(big.max()), 1.0)


class TestWsFoodRange(unittest.TestCase):
    def test_build_game_data_includes_food_past_camera(self):
        client = SlitherWSClient.__new__(SlitherWSClient)
        client.state = GameState()
        client.state.playing = True
        client.state.dead = False
        client.state.my_id = 1
        client.state.gsc = 0.9  # camera radius ~444
        client.state.snakes[1] = Snake(id=1, x=21600.0, y=21600.0)
        client.state.foods[1] = Food(id=1, x=22400.0, y=21600.0, size=2.0)
        client.state.foods[2] = Food(id=2, x=25000.0, y=21600.0, size=2.0)  # 3400 away
        data = client._build_game_data()
        foods = data['foods']
        xs = [f[0] for f in foods]
        self.assertIn(22400.0, xs)
        self.assertNotIn(25000.0, xs)
        self.assertEqual(client.MAX_FOODS, MAX_FOODS)

    def test_build_game_data_includes_prey_as_large_food(self):
        client = SlitherWSClient.__new__(SlitherWSClient)
        client.state = GameState()
        client.state.playing = True
        client.state.dead = False
        client.state.my_id = 1
        client.state.gsc = 0.9
        client.state.snakes[1] = Snake(id=1, x=21600.0, y=21600.0)
        client.state.foods[1] = Food(id=1, x=21700.0, y=21600.0, size=1.0)
        client.state.preys[9] = Prey(id=9, x=22000.0, y=21600.0, size=PREY_DEFAULT_SIZE)
        data = client._build_game_data()
        foods = data['foods']
        self.assertTrue(any(abs(f[0] - 22000.0) < 1e-6 and f[2] >= 8.0 for f in foods))


class TestBrowserFoodJs(unittest.TestCase):
    def test_injected_getstate_formats_and_uses_cluster_js(self):
        from browser_engine import SlitherBrowser
        browser = object.__new__(SlitherBrowser)
        browser.MAX_FOODS = MAX_FOODS
        browser.MAX_ENEMIES = 50
        browser.MAX_BODY_PTS = 150
        browser.driver = MagicMock()
        browser.inject_fast_getstate()
        js = browser.driver.execute_script.call_args[0][0]
        self.assertIn("var MAX_FOODS = %d" % MAX_FOODS, js)
        self.assertIn(str(int(FOOD_SENSE_RANGE)), js)
        self.assertNotIn("MAX_FOODS * 2", js)
        self.assertIn("window._botGetState", js)
        self.assertIn("window.preys", js)


if __name__ == '__main__':
    unittest.main()
