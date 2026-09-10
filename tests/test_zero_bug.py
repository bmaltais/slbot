import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slither_env import SlitherEnv


class TestZeroBug(unittest.TestCase):
    @patch('slither_env._create_browser', return_value=MagicMock())
    def test_zero_coordinate_death_handling(self, _mock_create):
        print("\nTesting Zero-Coordinate Death Handling...")

        env = SlitherEnv(headless=True, nickname="TestBot", view_plus=False)
        env.browser = MagicMock()

        # 2. Mock valid previous state (Center of Map)
        # Center 21600, 21600. Radius 21600.
        valid_pos = {'x': 21600, 'y': 21600}

        env.last_valid_data = {
            'self': {'x': 21600, 'y': 21600, 'len': 10, 'ang': 0},
            'enemies': [],
            'foods': [],
            'dead': False,
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'view_radius': 500
        }
        env.map_radius = 21600
        env.map_center_x = 21600
        env.map_center_y = 21600

        # 3. Simulate Step with DEAD frame and (0,0) coords (The Bug)
        bad_frame = {
            'self': {'x': 0, 'y': 0, 'len': 10}, # Corrupted coords
            'enemies': [],
            'foods': [],
            'dead': True, # DEAD
            'map_radius': 21600
        }

        # Step calls get_game_data() multiple times (once at start, once after action)
        # We ensure it returns the bad frame
        env.browser.get_game_data.return_value = bad_frame

        # 4. Execute Step
        # The logic inside step should:
        # - Detect dead=True
        # - Check coords (0,0) -> Invalid
        # - Fallback to env.last_valid_data (Center)
        # - Calculate death cause based on Center position

        next_state, reward, done, info = env.step(0)

        print(f"Info: {info}")

        # 5. Assertions
        self.assertTrue(done, "Episode should be done")

        # Critical Check: Cause should NOT be Wall
        self.assertNotEqual(info['cause'], "Wall",
                           f"Death cause should not be Wall when corrupted frame (0,0) is received. Got: {info['cause']}")

        # Check coordinates used for classification
        pos = info['pos']
        self.assertNotEqual(pos, (0,0), "Should not use (0,0) as death position")
        self.assertEqual(pos, (21600, 21600), "Should fallback to last valid position")

        # Check calculated wall distance (should be ~Radius)
        # Dist to wall at center = Radius - 0 = 21600
        self.assertAlmostEqual(info['wall_dist'], 21600, delta=100,
                              msg=f"Wall distance should be ~21600 at center, got {info['wall_dist']}")

        print("Test Passed: Zero-coordinate bug successfully mitigated.")
        env.close()

if __name__ == '__main__':
    unittest.main()
