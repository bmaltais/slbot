import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from food_sense import CLUSTER_EAT_MASS_CAP, squash_mass
from slither_env import ACTION_BOOST, SlitherEnv
from coord_transform import world_to_grid


class TestSlitherEnv(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('slither_env._create_browser', return_value=MagicMock())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = SlitherEnv(headless=True, nickname="TestBot", matrix_size=84, view_plus=False)
        self.env.browser = MagicMock()
        self.env.browser.get_game_data = MagicMock(return_value={})

    def test_radial_rendering(self):
        # Create a snake near the map boundary (21600)
        # Radius 21600. Center (21600, 21600).
        # Safe position: (21600, 21600) -> dist 0.
        # Near wall: x = 21600 + 21500 = 43100. y = 21600. Dist = 21500. Wall dist = 100.

        data = {
            'boundary_type': 'circle', # Irrelevant, forced anyway
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'dist_to_wall': 100, # Fake JS value, should be ignored/recalc
            'view_radius': 500,
            'self': {'x': 43100, 'y': 21600, 'len': 10},
            'gsc': 1.0
        }

        matrix = self.env._process_data_to_matrix(data)

        # Center (42, 42) is snake head.
        # At (42, 42), distance is 21500.
        # Wall threshold = 21600 - 500 = 21100.
        # 21500 > 21100, so it IS Wall (Warning Zone).
        self.assertEqual(matrix[1, 42, 42], 1.0)

        # Check Wall rendering
        # View radius 500. Scale = 84 / 1000 = 0.084.
        # Wall is at x=43200 (dist 21600).
        # Snake at x=43100.
        # dx_wall = 100.
        # gx = 42 + 100 * 0.084 = 50.4 -> 50.
        # So at gx=52, we should be strictly outside.
        self.assertEqual(matrix[1, 42, 55], 1.0)

    def test_update_from_game_data_ignores_nan(self):
        self.env.last_dist_to_wall = 1234
        self.env.map_radius = 21600
        self.env.map_center_x = 21600
        self.env.map_center_y = 21600

        data = {
            'dist_to_wall': float('nan'),
            'map_radius': float('nan'),
            'map_center_x': float('nan'),
            'map_center_y': float('nan')
        }

        self.env._update_from_game_data(data)

        self.assertEqual(self.env.last_dist_to_wall, 1234)
        self.assertEqual(self.env.map_radius, 21600)
        self.assertEqual(self.env.map_center_x, 21600)
        self.assertEqual(self.env.map_center_y, 21600)

    def test_has_valid_coordinates_rejects_cdp_wrong_snake_garbage(self):
        self.assertFalse(self.env._has_valid_coordinates({
            'self': {'x': 665762, 'y': 296448},
        }))
        self.assertFalse(self.env._has_valid_coordinates({
            'self': {'x': -548101, 'y': 716830},
        }))
        self.assertTrue(self.env._has_valid_coordinates({
            'self': {'x': 21600, 'y': 21600},
        }))

    def test_invalid_frame_returns_last_matrix(self):
        self.env.browser.send_action = MagicMock()
        self.env.last_matrix = np.ones((3, self.env.matrix_size, self.env.matrix_size), dtype=np.float32)
        self.env.browser.get_game_data = MagicMock(return_value={
            'dead': False,
            'valid': False,
            'self': {'x': 0, 'y': 0, 'len': 0}
        })

        state, reward, done, info = self.env.step(0)

        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info.get('cause'), 'InvalidFrame')
        self.assertTrue(np.array_equal(state, self.env.last_matrix))

    def _alive_frame(self):
        return {
            'dead': False,
            'valid': True,
            'self': {'x': 21600, 'y': 21600, 'len': 10, 'ang': 0.0},
            'enemies': [],
            'foods': [],
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'view_radius': 500,
            'gsc': 1.0,
        }

    def test_step_sends_action_without_combined_read(self):
        """Issue #4: do not pay a discarded send_action_get_data() round-trip."""
        alive = self._alive_frame()
        self.env.frame_skip = 0
        self.env._cached_data = alive
        self.env.browser.send_action = MagicMock()
        self.env.browser.send_action_get_data = MagicMock(return_value=alive)
        self.env.browser.get_game_data = MagicMock(return_value=alive)

        _, _, done, _ = self.env.step(0)

        self.assertFalse(done)
        self.env.browser.send_action.assert_called_once()
        args, _kwargs = self.env.browser.send_action.call_args
        self.assertEqual(len(args), 2)
        self.env.browser.send_action_get_data.assert_not_called()
        self.env.browser.get_game_data.assert_called_once()

    def test_step_falls_back_to_combined_send_without_send_action(self):
        """Backends that only expose send_action_get_data() still get a send."""
        alive = self._alive_frame()

        class CombinedOnly:
            def __init__(self):
                self.send_action_get_data = MagicMock(return_value=alive)
                self.get_game_data = MagicMock(return_value=alive)

        self.env.frame_skip = 0
        self.env._cached_data = alive
        self.env.browser = CombinedOnly()

        _, _, done, _ = self.env.step(0)

        self.assertFalse(done)
        self.env.browser.send_action_get_data.assert_called_once()
        self.env.browser.get_game_data.assert_called_once()

    def _quiet_rewards(self):
        self.env.survival_reward = 0.0
        self.env.survival_escalation = 0.0
        self.env.food_reward = 0.0
        self.env.food_shaping = 0.0
        self.env.cluster_eat_reward = 0.0
        self.env.length_bonus = 0.0
        self.env.straight_penalty = 0.0
        self.env.wall_proximity_penalty = 0.0
        self.env.enemy_proximity_penalty = 0.0
        self.env.enemy_approach_penalty = 0.0
        self.env.boost_penalty = 0.0
        self.env.mass_loss_penalty = 0.0
        self.env.starvation_penalty = 0.0
        self.env.contest_food_reward = 0.0
        self.env.kill_opportunity_reward = 0.0
        self.env.enemy_zone_control_reward = 0.0
        self.env.idle_food_penalty = 0.0
        self.env.idle_food_range = 500.0
        self.env.food_lock_radius = 240.0
        self.env.boost_cluster_reward = 0.0
        self.env.boost_cluster_range = 400.0
        self.env.boost_cluster_min_mass = 6.0

    def _step_frames(self, pre, post, action=0):
        self.env.frame_skip = 0
        self.env._cached_data = pre
        self.env.prev_length = pre['self']['len']
        self.env.browser.send_action = MagicMock()
        self.env.browser.get_game_data = MagicMock(return_value=post)
        return self.env.step(action)

    def test_idle_food_penalty_when_nearby_cluster_not_eaten(self):
        self._quiet_rewards()
        self.env.idle_food_penalty = 0.03
        pre = self._alive_frame()
        post = self._alive_frame()
        food = [21600.0 + 200.0, 21600.0, 4.0]
        pre['foods'] = [food]
        post['foods'] = [food]
        _state, reward, done, _info = self._step_frames(pre, post)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.03)

    def test_idle_food_penalty_skips_far_food(self):
        self._quiet_rewards()
        self.env.idle_food_penalty = 0.03
        pre = self._alive_frame()
        post = self._alive_frame()
        food = [21600.0 + 1500.0, 21600.0, 4.0]
        pre['foods'] = [food]
        post['foods'] = [food]
        _state, reward, done, _info = self._step_frames(pre, post)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.0)

    def test_idle_food_penalty_skips_when_eating(self):
        self._quiet_rewards()
        self.env.idle_food_penalty = 0.03
        pre = self._alive_frame()
        post = self._alive_frame()
        eaten = [21610.0, 21600.0, 2.0]
        leftover = [21600.0 + 180.0, 21600.0, 6.0]
        pre['foods'] = [eaten, leftover]
        post['foods'] = [leftover]
        post['self'] = dict(pre['self'])
        post['self']['x'] = 21620.0
        post['self']['len'] = 11
        _state, reward, done, _info = self._step_frames(pre, post)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.0)

    def test_set_curriculum_stage_loads_idle_food_knobs(self):
        self.env.set_curriculum_stage({
            'idle_food_penalty': 0.03,
            'idle_food_range': 500,
            'food_lock_radius': 240,
        })
        self.assertAlmostEqual(self.env.idle_food_penalty, 0.03)
        self.assertEqual(self.env.idle_food_range, 500)
        self.assertEqual(self.env.food_lock_radius, 240)

    def test_set_curriculum_stage_loads_boost_cluster_knobs(self):
        self.env.set_curriculum_stage({
            'boost_cluster_reward': 0.4,
            'boost_cluster_range': 400,
            'boost_cluster_min_mass': 6,
            'boost_penalty': 0.04,
        })
        self.assertAlmostEqual(self.env.boost_cluster_reward, 0.4)
        self.assertEqual(self.env.boost_cluster_range, 400)
        self.assertEqual(self.env.boost_cluster_min_mass, 6)
        self.assertAlmostEqual(self.env.boost_penalty, 0.04)

    def test_boost_toward_nearby_pile_pays(self):
        self._quiet_rewards()
        self.env.boost_cluster_reward = 0.4
        self.env.boost_cluster_range = 400.0
        self.env.boost_cluster_min_mass = 6.0
        self.env.boost_penalty = 0.04
        pre = self._alive_frame()
        post = self._alive_frame()
        pile = [[21600.0 + 180.0 + i, 21600.0, 2.0] for i in range(8)]
        pre['foods'] = pile
        post['foods'] = pile
        post['self'] = dict(pre['self'])
        post['self']['x'] = 21630.0
        _state, reward, done, _info = self._step_frames(pre, post, action=ACTION_BOOST)
        self.assertFalse(done)
        expected = 0.4 * squash_mass(16.0, CLUSTER_EAT_MASS_CAP) * (1.0 - 180.0 / 400.0)
        self.assertAlmostEqual(reward, expected)

    def test_boost_into_empty_pays_penalty(self):
        self._quiet_rewards()
        self.env.boost_penalty = 0.04
        self.env.boost_cluster_reward = 0.4
        pre = self._alive_frame()
        post = self._alive_frame()
        post['self'] = dict(pre['self'])
        post['self']['x'] = 21630.0
        _state, reward, done, _info = self._step_frames(pre, post, action=ACTION_BOOST)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.04)

    def test_boost_at_crumb_pays_penalty(self):
        self._quiet_rewards()
        self.env.boost_penalty = 0.04
        self.env.boost_cluster_reward = 0.4
        self.env.boost_cluster_min_mass = 6.0
        pre = self._alive_frame()
        post = self._alive_frame()
        food = [[21600.0 + 150.0, 21600.0, 1.0]]
        pre['foods'] = food
        post['foods'] = food
        post['self'] = dict(pre['self'])
        post['self']['x'] = 21630.0
        _state, reward, done, _info = self._step_frames(pre, post, action=ACTION_BOOST)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.04)

    def test_negative_boost_penalty_rewards_empty_boost(self):
        self._quiet_rewards()
        self.env.boost_penalty = -0.2
        self.env.boost_cluster_reward = 0.0
        pre = self._alive_frame()
        post = self._alive_frame()
        post['self'] = dict(pre['self'])
        post['self']['x'] = 21630.0
        _state, reward, done, _info = self._step_frames(pre, post, action=ACTION_BOOST)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.2)

if __name__ == '__main__':
    unittest.main()
