import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from food_sense import (
    CLUSTER_EAT_MASS_CAP,
    FOOD_CHANNEL_LOG_CAP,
    FOOD_KEEP_DIST_BIAS,
    FOOD_LIST_PRUNE_MULT,
    FOOD_LOCK_RADIUS,
    FOOD_SENSE_RANGE,
    MAX_FOODS,
    MAX_FOODS_DEFAULT,
    PREY_DEFAULT_SIZE,
    configured_max_foods,
    best_food_target,
    boost_cluster_bonus,
    cluster_eat_bonus,
    cluster_foods,
    eaten_food_mass,
    food_draw_radius_px,
    idle_food_cost,
    js_collect_foods,
    locked_food_target,
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
        self.assertIn("for (var i = 0; i < window.foods.length; i++)", js)
        self.assertIn("window.preys", js)
        self.assertIn("foodList = []", js)
        self.assertIn("considerFood", js)
        self.assertNotIn("viewRadius * 1.2", js)
        self.assertIn("var senseRange = %.1f" % FOOD_SENSE_RANGE, js)
        self.assertIn("MAX_FOODS * %d" % FOOD_LIST_PRUNE_MULT, js)
        self.assertIn("foodList.length = MAX_FOODS", js)

    def test_configured_max_foods_reads_env(self):
        with patch.dict(os.environ, {"SLBOT_MAX_FOODS": "120"}, clear=False):
            self.assertEqual(configured_max_foods(), 120)
        with patch.dict(os.environ, {"SLBOT_MAX_FOODS": "99999"}, clear=False):
            self.assertEqual(configured_max_foods(), 4000)
        with patch.dict(os.environ, {"SLBOT_MAX_FOODS": "nope"}, clear=False):
            self.assertEqual(configured_max_foods(), MAX_FOODS_DEFAULT)

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

    def test_lock_holds_when_a_nearby_pile_scores_higher(self):
        mx, my = 0.0, 0.0
        pile_a = [[200.0 + i, 0.0, 1.0] for i in range(8)]
        pile_b = [[0.0, 280.0 + i, 1.0] for i in range(20)]
        foods = pile_a + pile_b
        first = best_food_target(pile_a, mx, my)
        self.assertIsNotNone(first)
        locked = locked_food_target(foods, mx, my, first)
        self.assertIsNotNone(locked)
        # Unlocked scoring prefers the denser pile_b; the lock must stay on A.
        unlocked = best_food_target(foods, mx, my)
        self.assertGreater(unlocked[1], 200.0)
        self.assertLess(abs(locked[0] - first[0]), 40.0)
        self.assertLess(abs(locked[1] - first[1]), 40.0)

    def test_lock_releases_when_the_cluster_is_gone(self):
        pile_a = [[200.0 + i, 0.0, 1.0] for i in range(8)]
        pile_b = [[0.0, 280.0 + i, 1.0] for i in range(20)]
        locked_a = best_food_target(pile_a, 0.0, 0.0)
        retarget = locked_food_target(pile_b, 0.0, 0.0, locked_a)
        self.assertIsNotNone(retarget)
        self.assertGreater(retarget[1], 200.0)

    def test_lock_radius_zero_always_picks_best(self):
        pile_a = [[200.0 + i, 0.0, 1.0] for i in range(8)]
        pile_b = [[0.0, 280.0 + i, 1.0] for i in range(20)]
        locked_a = best_food_target(pile_a, 0.0, 0.0)
        best = locked_food_target(pile_a + pile_b, 0.0, 0.0, locked_a, lock_radius=0)
        unlocked = best_food_target(pile_a + pile_b, 0.0, 0.0)
        self.assertAlmostEqual(best[0], unlocked[0])
        self.assertAlmostEqual(best[1], unlocked[1])

    def test_best_target_aims_at_near_end_of_trail(self):
        trail = [[80.0 + i * 80.0, 0.0, 1.0] for i in range(10)]
        target = best_food_target(trail, 0.0, 0.0)
        self.assertIsNotNone(target)
        cx, cy, dist, mass = target
        self.assertAlmostEqual(mass, 10.0)
        self.assertLess(cx, 160.0)
        self.assertGreater(cx, 40.0)
        self.assertAlmostEqual(dist, cx)

    def test_abutting_crumb_joins_the_trail(self):
        crumb = [[30.0, 0.0, 1.0]]
        trail = [[80.0 + i * 80.0, 0.0, 1.0] for i in range(10)]
        target = best_food_target(crumb + trail, 0.0, 0.0)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target[3], 11.0)
        self.assertAlmostEqual(target[0], 30.0)

    def test_trail_beats_disconnected_crumb(self):
        crumb = [[0.0, 250.0, 1.0]]
        trail = [[80.0 + i * 80.0, 0.0, 1.0] for i in range(10)]
        target = best_food_target(crumb + trail, 0.0, 0.0)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target[3], 10.0)
        self.assertGreater(target[0], 50.0)
        self.assertLess(abs(target[1]), 1.0)

    def test_disconnected_piles_are_separate_strings(self):
        pile_a = [[200.0 + i, 0.0, 1.0] for i in range(8)]
        pile_b = [[0.0, 280.0 + i, 1.0] for i in range(20)]
        target = best_food_target(pile_a + pile_b, 0.0, 0.0)
        self.assertIsNotNone(target)
        self.assertGreater(target[1], 200.0)
        self.assertAlmostEqual(target[3], 20.0)

    def test_lock_slides_along_trail(self):
        trail = [[80.0 + i * 80.0, 0.0, 1.0] for i in range(10)]
        first = best_food_target(trail, 0.0, 0.0)
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first[0], 80.0)
        rest = trail[1:]
        locked = locked_food_target(rest, 80.0, 0.0, first)
        self.assertIsNotNone(locked)
        self.assertAlmostEqual(locked[0], 160.0)
        self.assertAlmostEqual(locked[3], 9.0)

    def test_boost_cluster_bonus_requires_close_fat_closing_pile(self):
        self.assertEqual(boost_cluster_bonus(180.0, 16.0, False, 0.4), 0.0)
        self.assertEqual(boost_cluster_bonus(180.0, 2.0, True, 0.4), 0.0)
        self.assertEqual(boost_cluster_bonus(500.0, 16.0, True, 0.4), 0.0)
        self.assertEqual(boost_cluster_bonus(None, 16.0, True, 0.4), 0.0)
        paid = boost_cluster_bonus(180.0, 16.0, True, 0.4)
        self.assertGreater(paid, 0.0)
        expected = 0.4 * squash_mass(16.0, CLUSTER_EAT_MASS_CAP) * (1.0 - 180.0 / 400.0)
        self.assertAlmostEqual(paid, expected)

    def test_idle_food_cost_only_when_in_range_and_hungry(self):
        self.assertEqual(idle_food_cost(200.0, ate=False, penalty=0.03, commit_range=500.0), 0.03)
        self.assertEqual(idle_food_cost(200.0, ate=True, penalty=0.03, commit_range=500.0), 0.0)
        self.assertEqual(idle_food_cost(800.0, ate=False, penalty=0.03, commit_range=500.0), 0.0)
        self.assertEqual(idle_food_cost(None, ate=False, penalty=0.03, commit_range=500.0), 0.0)
        self.assertEqual(idle_food_cost(200.0, ate=False, penalty=0.0, commit_range=500.0), 0.0)
        self.assertEqual(FOOD_LOCK_RADIUS, 240.0)


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
        self.assertNotIn("viewRadius * 1.2", js)
        self.assertIn("window._botGetState", js)
        self.assertIn("window.preys", js)
        self.assertIn("considerFood", js)


if __name__ == '__main__':
    unittest.main()
