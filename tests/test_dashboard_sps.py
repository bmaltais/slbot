import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import SPS_WINDOW_S, rolling_steps_per_sec


class TestRollingStepsPerSec(unittest.TestCase):
    def test_empty_is_zero(self):
        times = deque()
        self.assertEqual(rolling_steps_per_sec(times, now=100.0), 0.0)

    def test_full_window_divides_by_ten(self):
        # 70 steps evenly across the last 10s → 7.0 steps/s
        now = 100.0
        times = deque(now - 10.0 + (i + 1) * (10.0 / 70) for i in range(70))
        self.assertAlmostEqual(rolling_steps_per_sec(times, now), 7.0, places=5)
        self.assertEqual(len(times), 70)

    def test_prunes_samples_older_than_window(self):
        now = 50.0
        times = deque([39.9, 40.0, 45.0, 49.0])
        sps = rolling_steps_per_sec(times, now, window=10.0)
        self.assertEqual(list(times), [40.0, 45.0, 49.0])
        self.assertAlmostEqual(sps, 0.3, places=5)  # 3 / 10

    def test_spawn_idle_lowers_the_average(self):
        # 20 steps in the first 2s, then 8s idle → 2.0 steps/s over 10s
        now = 20.0
        times = deque(10.0 + i * 0.1 for i in range(20))
        self.assertAlmostEqual(rolling_steps_per_sec(times, now), 2.0, places=5)

    def test_warmup_divides_by_elapsed_not_full_window(self):
        started = 10.0
        now = 12.0  # only 2s of tracking
        times = deque([10.5, 11.0, 11.5, 12.0])
        self.assertAlmostEqual(
            rolling_steps_per_sec(times, now, started_at=started),
            2.0,  # 4 / 2s, not 4 / 10
            places=5,
        )

    def test_started_at_now_is_zero(self):
        now = 5.0
        times = deque()
        self.assertEqual(
            rolling_steps_per_sec(times, now, started_at=now),
            0.0,
        )

    def test_window_constant(self):
        self.assertEqual(SPS_WINDOW_S, 10.0)


if __name__ == "__main__":
    unittest.main()
