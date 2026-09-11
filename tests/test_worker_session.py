import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sector_layout import SECTOR_DIM

from worker_session import WorkerSession


def _obs(fill):
    return {
        'matrix': np.full((3, 8, 8), fill, dtype=np.float32),
        'sectors': np.full(SECTOR_DIM, fill, dtype=np.float32),
    }


class FakeEnv:
    def __init__(self):
        self.step_count = 0
        self.reset_count = 0
        self.reset_started = threading.Event()
        self.reset_release = threading.Event()
        self.alive_obs = _obs(1.0)
        self.dead_obs = _obs(0.0)
        self.closed = False
        self.fail_reset = False

    def step(self, action):
        self.step_count += 1
        if action == 99:
            return self.dead_obs, -15.0, True, {'cause': 'Wall'}
        return self.alive_obs, 0.1, False, {'food_eaten': 0, 'length': 10}

    def reset(self):
        self.reset_started.set()
        if not self.reset_release.wait(timeout=2):
            raise TimeoutError("reset was not released")
        if self.fail_reset:
            raise RuntimeError("reset exploded")
        self.reset_count += 1
        return self.alive_obs

    def set_curriculum_stage(self, cfg):
        self.stage = cfg

    def close(self):
        self.reset_release.set()
        self.closed = True


class TestWorkerSession(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.session = WorkerSession(self.env, matrix_size=8)

    def tearDown(self):
        self.env.reset_release.set()
        self.session.close()

    def test_death_returns_before_reset_finishes(self):
        t0 = time.perf_counter()
        obs, reward, done, info = self.session.step(99)
        elapsed = time.perf_counter() - t0

        self.assertTrue(done)
        self.assertEqual(info['cause'], 'Wall')
        self.assertIn('terminal_observation', info)
        self.assertLess(elapsed, 0.05, "death must not wait for env.reset()")
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        self.assertEqual(self.env.reset_count, 0)

    def test_steps_during_reset_are_spawning_then_spawned(self):
        self.session.step(99)
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        steps_at_death = self.env.step_count

        obs, reward, done, info = self.session.step(0)
        self.assertFalse(done)
        self.assertTrue(info.get('spawning'))
        self.assertFalse(info.get('spawned', False))
        self.assertEqual(self.env.step_count, steps_at_death)
        self.assertEqual(self.env.reset_count, 0)

        self.env.reset_release.set()
        spawned = False
        for _ in range(50):
            obs, reward, done, info = self.session.step(0)
            if info.get('spawned'):
                spawned = True
                break
            self.assertTrue(info.get('spawning'), info)
            time.sleep(0.01)

        self.assertTrue(spawned)
        self.assertEqual(self.env.reset_count, 1)
        np.testing.assert_array_equal(obs['matrix'], self.env.alive_obs['matrix'])

        # Next step is live play again
        obs, reward, done, info = self.session.step(1)
        self.assertFalse(done)
        self.assertFalse(info.get('spawning', False))
        self.assertFalse(info.get('spawned', False))
        self.assertEqual(reward, 0.1)

    def test_reset_async_does_not_block(self):
        t0 = time.perf_counter()
        obs = self.session.reset_async()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.05)
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        self.assertEqual(self.env.reset_count, 0)
        self.assertTrue(obs.get('spawning'))
        np.testing.assert_array_equal(obs['matrix'], np.zeros((3, 8, 8), dtype=np.float32))

        self.env.reset_release.set()
        for _ in range(50):
            _, _, _, info = self.session.step(0)
            if info.get('spawned'):
                break
            time.sleep(0.01)
        else:
            self.fail("async reset never completed")
        self.assertEqual(self.env.reset_count, 1)

    def test_reset_async_while_busy_returns_last_obs_not_zeros(self):
        self.session.step(1)  # live frame of ones
        obs = self.session.reset_async()
        self.assertTrue(obs.get('spawning'))
        np.testing.assert_array_equal(obs['matrix'], self.env.alive_obs['matrix'])
        self.assertTrue(self.env.reset_started.wait(timeout=1))

        again = self.session.reset_async()
        self.assertTrue(again.get('spawning'))
        np.testing.assert_array_equal(again['matrix'], self.env.alive_obs['matrix'])

    def test_join_reset_keeps_thread_on_timeout(self):
        self.session.reset_async()
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        finished = self.session._join_reset(timeout=0.05)
        self.assertFalse(finished)
        self.assertTrue(self.session._reset_thread.is_alive())

        steps_before = self.env.step_count
        _, _, _, info = self.session.step(0)
        self.assertTrue(info.get('spawning'))
        self.assertEqual(self.env.step_count, steps_before)

    def test_set_stage_does_not_block_during_reset(self):
        self.session.reset_async()
        self.assertTrue(self.env.reset_started.wait(timeout=1))

        t0 = time.perf_counter()
        ack = self.session.handle('set_stage', {'food_reward': 9})
        self.assertLess(time.perf_counter() - t0, 0.05)
        self.assertEqual(ack, 'ok')
        self.assertFalse(hasattr(self.env, 'stage'))

        self.env.reset_release.set()
        spawned = False
        for _ in range(50):
            _, _, _, info = self.session.step(0)
            if info.get('spawned'):
                spawned = True
                break
            time.sleep(0.01)
        self.assertTrue(spawned)
        self.assertEqual(self.env.stage, {'food_reward': 9})

    def test_reset_sync_waits_for_observation(self):
        self.env.reset_release.set()
        obs = self.session.reset_sync()
        self.assertEqual(self.env.reset_count, 1)
        np.testing.assert_array_equal(obs['matrix'], self.env.alive_obs['matrix'])

    def test_join_reset_preserves_background_error(self):
        self.env.fail_reset = True
        self.session.reset_async()
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        self.env.reset_release.set()
        self.assertTrue(self.session._join_reset(timeout=1))
        self.assertIsInstance(self.session._reset_error, RuntimeError)
        with self.assertRaises(RuntimeError) as ctx:
            self.session.reset_sync()
        self.assertIn("reset exploded", str(ctx.exception))

    def test_reset_obs_with_spawning_does_not_mark_spawned(self):
        """Issue #16: mixed-mode first frame if we mark spawned before CDP is live."""
        orig_reset = self.env.reset

        def spawning_reset():
            obs = orig_reset()
            obs = dict(obs)
            obs['spawning'] = True
            return obs

        self.env.reset = spawning_reset
        self.session.step(99)
        self.assertTrue(self.env.reset_started.wait(timeout=1))
        self.env.reset_release.set()
        saw_spawned = False
        saw_spawning = False
        for _ in range(50):
            obs, reward, done, info = self.session.step(0)
            if info.get('spawned'):
                saw_spawned = True
                break
            if info.get('spawning') and self.env.reset_count:
                saw_spawning = True
                break
            time.sleep(0.01)
        self.assertTrue(saw_spawning)
        self.assertFalse(saw_spawned)
        self.assertTrue(obs.get('spawning'))


class TestDeathPacketNonBlocking(unittest.TestCase):
    def test_submit_returns_while_write_is_blocked(self):
        import slither_env

        writer = slither_env._DeathPacketWriter()
        started = threading.Event()
        released = threading.Event()

        def slow_write(*_args):
            started.set()
            released.wait(timeout=2)

        writer._write = slow_write
        try:
            t0 = time.perf_counter()
            writer.submit(
                np.zeros((3, 4, 4), dtype=np.float32),
                1.0,
                'Wall',
                {'self': {'x': 1, 'y': 2}},
                'circle',
            )
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.05)
            self.assertTrue(started.wait(timeout=1))
        finally:
            released.set()
            writer.close()

    def test_death_packet_filenames_are_unique_in_same_second(self):
        import slither_env

        writer = slither_env._DeathPacketWriter()
        captured = []

        def capture(*args):
            captured.append(args)

        writer._write = capture
        try:
            for _ in range(2):
                writer.submit(
                    np.zeros((3, 2, 2), dtype=np.float32),
                    0.0,
                    'Wall',
                    {},
                    'circle',
                )
            deadline = time.time() + 2
            while len(captured) < 2 and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(captured), 2)
            uniq_a, uniq_b = captured[0][-1], captured[1][-1]
            self.assertNotEqual(uniq_a, uniq_b)
            self.assertTrue(uniq_a)
            self.assertTrue(uniq_b)
        finally:
            writer.close()

    def test_unsafe_cause_is_sanitized_for_filenames(self):
        import slither_env

        self.assertEqual(slither_env._safe_filename_token('Wall'), 'Wall')
        token = slither_env._safe_filename_token('../etc/passwd')
        self.assertNotIn('..', token)
        self.assertNotIn('/', token)
        self.assertNotIn('\\', token)
        self.assertTrue(token)

    def test_close_stops_writer_when_queue_is_full(self):
        import slither_env

        writer = slither_env._DeathPacketWriter()
        in_write = threading.Event()
        allow_write = threading.Event()

        def slow_write(*_args):
            in_write.set()
            allow_write.wait(timeout=2)

        writer._write = slow_write
        try:
            writer.submit(np.zeros((3, 2, 2), dtype=np.float32), 0.0, 'Wall', {}, 'circle')
            self.assertTrue(in_write.wait(timeout=1))
            for _ in range(writer._MAX_QUEUE):
                writer.submit(np.zeros((3, 2, 2), dtype=np.float32), 0.0, 'Wall', {}, 'circle')
            allow_write.set()
            writer.close()
            writer._t.join(timeout=2)
            self.assertFalse(writer._t.is_alive())
        finally:
            allow_write.set()
            writer.close()


class TestVecFrameStackSpawn(unittest.TestCase):
    def setUp(self):
        from trainer import VecFrameStack

        class FakeVenv:
            def __init__(self):
                self.num_agents = 1
                self.payload = None

            def step(self, actions):
                return self.payload

            def reset_one(self, i):
                return self.reset_obs

        self.venv = FakeVenv()
        self.stack = VecFrameStack(self.venv, k=4)
        live = np.ones((3, 4, 4), dtype=np.float32)
        self.live_mat = live
        self.stack.frames[0].clear()
        for _ in range(4):
            self.stack.frames[0].append(live)

    def test_death_keeps_live_frames_until_spawned(self):
        dead = {
            'matrix': np.zeros((3, 4, 4), dtype=np.float32),
            'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
        }
        self.venv.payload = (
            [dead],
            [-15.0],
            [True],
            [{'terminal_observation': dead, 'cause': 'Wall'}],
        )
        _, _, dones, infos = self.stack.step([0])
        self.assertTrue(dones[0])
        self.assertIn('terminal_observation', infos[0])
        for frame in self.stack.frames[0]:
            np.testing.assert_array_equal(frame, self.live_mat)

        spawned_mat = np.full((3, 4, 4), 2.0, dtype=np.float32)
        spawned = {'matrix': spawned_mat, 'sectors': np.full(SECTOR_DIM, 2.0, dtype=np.float32)}
        self.venv.payload = ([spawned], [0.0], [False], [{'spawned': True}])
        self.stack.step([0])
        for frame in self.stack.frames[0]:
            np.testing.assert_array_equal(frame, spawned_mat)

    def test_reset_one_spawning_preserves_live_frames(self):
        zeros = np.zeros((3, 4, 4), dtype=np.float32)
        self.venv.reset_obs = {
            'matrix': zeros,
            'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
            'spawning': True,
        }
        self.stack.reset_one(0)
        for frame in self.stack.frames[0]:
            np.testing.assert_array_equal(frame, self.live_mat)


if __name__ == '__main__':
    unittest.main()
