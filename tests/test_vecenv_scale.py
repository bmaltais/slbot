import os
import sys
import threading
import time
import unittest

from multiprocessing import Pipe

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sector_layout import SECTOR_DIM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_process import READY_MSG, delayed_echo_worker, echo_worker, run_worker_loop
from worker_session import WorkerSession

from test_worker_session import FakeEnv


class TestWorkerBootLoop(unittest.TestCase):
    def test_step_returns_spawning_before_boot_finishes(self):
        release = threading.Event()
        parent, child = Pipe()

        def boot():
            release.wait(timeout=2)
            env = FakeEnv()
            env.reset_release.set()
            return WorkerSession(env, 8)

        t = threading.Thread(
            target=run_worker_loop, args=(child, 8, boot), daemon=True,
        )
        t.start()
        self.assertEqual(parent.recv(), READY_MSG)

        t0 = time.perf_counter()
        parent.send(('step', 0))
        obs, reward, done, info = parent.recv()
        self.assertLess(time.perf_counter() - t0, 0.1)
        self.assertTrue(info.get('spawning'))
        self.assertTrue(obs.get('spawning'))

        parent.send(('set_stage', {'food_reward': 3}))
        self.assertEqual(parent.recv(), 'ok')

        release.set()
        parent.send(('close', None))
        t.join(timeout=2)
        self.assertFalse(t.is_alive())

    def test_reset_waits_for_boot_then_returns_obs(self):
        release = threading.Event()
        parent, child = Pipe()

        def boot():
            release.wait(timeout=2)
            env = FakeEnv()
            env.reset_release.set()
            return WorkerSession(env, 8)

        t = threading.Thread(
            target=run_worker_loop, args=(child, 8, boot), daemon=True,
        )
        t.start()
        self.assertEqual(parent.recv(), READY_MSG)

        def do_reset():
            parent.send(('reset', None))
            return parent.recv()

        result = {}

        def run_reset():
            result['obs'] = do_reset()

        rt = threading.Thread(target=run_reset)
        rt.start()
        time.sleep(0.05)
        self.assertTrue(rt.is_alive(), "reset should wait until env boot finishes")
        release.set()
        rt.join(timeout=2)
        self.assertIn('obs', result)
        np.testing.assert_array_equal(result['obs']['matrix'], np.full((3, 8, 8), 1.0, dtype=np.float32))

        parent.send(('close', None))
        t.join(timeout=2)


class TestSubprocVecEnvScale(unittest.TestCase):
    def setUp(self):
        from trainer import SubprocVecEnv
        self.SubprocVecEnv = SubprocVecEnv

    def _make_env(self, n=1):
        return self.SubprocVecEnv(
            num_agents=n,
            matrix_size=8,
            frame_skip=1,
            worker_fn=echo_worker,
        )

    def test_add_agent_does_not_wait_for_worker_boot(self):
        env = self._make_env(1)
        try:
            t0 = time.perf_counter()
            obs = env.add_agent()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25, f"add_agent blocked for {elapsed:.2f}s")
            self.assertTrue(obs.get('spawning'))
            self.assertEqual(env.num_agents, 2)

            t0 = time.perf_counter()
            _, _, _, infos = env.step([0, 0])
            step_elapsed = time.perf_counter() - t0
            self.assertLess(step_elapsed, 0.25, f"step blocked on booting worker for {step_elapsed:.2f}s")
            self.assertTrue(infos[1].get('spawning'))
        finally:
            env.close()

    def test_remove_agent_does_not_wait_for_worker_close(self):
        env = self._make_env(1)
        try:
            env.add_agent()
            # Wait until the scaled worker is in the recv loop so close is processed.
            deadline = time.time() + 2
            while time.time() < deadline and not env._boot_ready[1]:
                time.sleep(0.02)
            self.assertTrue(env._boot_ready[1])

            t0 = time.perf_counter()
            self.assertTrue(env.remove_agent())
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25, f"remove_agent blocked for {elapsed:.2f}s")
            self.assertEqual(env.num_agents, 1)
        finally:
            env.close()

    PARENT_WORK_S = 0.08

    def _assert_step_overlapped(self, elapsed, info):
        """Serial send+parent+recv is ~worker_step_s + PARENT_WORK_S; overlap stays near worker_step_s."""
        worker_s = info.get('worker_step_s')
        self.assertIsNotNone(worker_s, "delayed_echo_worker must report worker_step_s")
        self.assertLess(
            elapsed,
            worker_s + self.PARENT_WORK_S * 0.5,
            f"no overlap: elapsed={elapsed:.3f}s worker={worker_s:.3f}s",
        )
        self.assertGreaterEqual(elapsed, worker_s * 0.5)

    def test_step_async_overlaps_parent_work(self):
        """Parent work between send and recv must not add to worker step time."""
        env = self.SubprocVecEnv(
            num_agents=1,
            matrix_size=8,
            frame_skip=1,
            worker_fn=delayed_echo_worker,
        )
        try:
            t0 = time.perf_counter()
            env.step_async([0])
            time.sleep(self.PARENT_WORK_S)  # stand-in for optimize_model()
            _, _, _, infos = env.step_wait()
            elapsed = time.perf_counter() - t0
            self._assert_step_overlapped(elapsed, infos[0])
            self.assertTrue(infos[0].get('spawning'))
        finally:
            env.close()

    def test_step_still_send_then_wait(self):
        env = self._make_env(1)
        try:
            _, _, _, infos = env.step([0])
            self.assertTrue(infos[0].get('spawning'))
        finally:
            env.close()

    def test_step_async_twice_raises(self):
        env = self._make_env(1)
        try:
            env.step_async([0])
            with self.assertRaises(RuntimeError):
                env.step_async([0])
            env.step_wait()
        finally:
            env.close()

    def test_step_wait_without_async_raises(self):
        env = self._make_env(1)
        try:
            with self.assertRaises(RuntimeError):
                env.step_wait()
        finally:
            env.close()

    def test_step_async_rejects_mismatched_action_count(self):
        env = self._make_env(1)
        try:
            with self.assertRaises(ValueError):
                env.step_async([0, 1])
            env.step_async([0])
            env.step_wait()
        finally:
            env.close()

    def test_vecframestack_step_async_overlaps_parent_work(self):
        from trainer import VecFrameStack
        raw = self.SubprocVecEnv(
            num_agents=1,
            matrix_size=8,
            frame_skip=1,
            worker_fn=delayed_echo_worker,
        )
        env = VecFrameStack(raw, k=4)
        try:
            t0 = time.perf_counter()
            env.step_async([0])
            time.sleep(self.PARENT_WORK_S)
            obs_list, _, _, infos = env.step_wait()
            elapsed = time.perf_counter() - t0
            self._assert_step_overlapped(elapsed, infos[0])
            self.assertEqual(obs_list[0]['matrix'].shape[0], 12)
            self.assertTrue(infos[0].get('spawning'))
        finally:
            env.close()

    def test_vecframestack_step_async_wait_stacks_frames(self):
        from trainer import VecFrameStack

        class FakeVenv:
            def __init__(self):
                self.num_agents = 1
                self.async_called = False
                self.mat = np.ones((3, 4, 4), dtype=np.float32)

            def step_async(self, actions):
                self.async_called = True

            def step_wait(self):
                self.assert_async = self.async_called
                obs = {'matrix': self.mat, 'sectors': np.zeros(SECTOR_DIM, dtype=np.float32)}
                info = {
                    'food_eaten': 0,
                    'pos': (0, 0),
                    'wall_dist': -1,
                    'enemy_dist': -1,
                    'length': 0,
                }
                return [obs], [0.0], [False], [info]

        venv = FakeVenv()
        stack = VecFrameStack(venv, k=4)
        stack._fill_frames(0, np.zeros((3, 4, 4), dtype=np.float32))
        stack.step_async([0])
        obs_list, rews, dones, infos = stack.step_wait()
        self.assertTrue(venv.async_called)
        self.assertTrue(venv.assert_async)
        self.assertEqual(obs_list[0]['matrix'].shape, (12, 4, 4))
        self.assertEqual(rews[0], 0.0)
        self.assertFalse(dones[0])

    def test_vecframestack_add_agent_marks_spawning(self):
        from trainer import VecFrameStack

        class FakeVenv:
            def __init__(self):
                self.num_agents = 1

            def add_agent(self):
                self.num_agents += 1
                return {
                    'matrix': np.zeros((3, 4, 4), dtype=np.float32),
                    'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
                    'spawning': True,
                }

        stack = VecFrameStack(FakeVenv(), k=4)
        t0 = time.perf_counter()
        obs = stack.add_agent()
        self.assertLess(time.perf_counter() - t0, 0.05)
        self.assertTrue(obs.get('spawning'))
        self.assertEqual(stack.num_agents, 2)
        self.assertEqual(len(stack.frames[1]), 4)


if __name__ == '__main__':
    unittest.main()
