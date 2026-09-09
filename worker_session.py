"""Worker-side env loop: death returns immediately, reset runs in the background.

SubprocVecEnv is lock-step — the parent waits for every worker before anyone
gets another action. If a dead snake's reset (browser reconnect, up to ~10s)
runs inside `step`, every other live snake flies with its last action.

This session starts reset on a background thread and answers subsequent `step`
commands with `spawning` / `spawned` until the new episode is ready.
"""

import threading

import numpy as np


SECTOR_DIM = 99


class WorkerSession:
    """Serialize env.step / env.reset and respawn without blocking the vecenv."""

    def __init__(self, env, matrix_size):
        self.env = env
        self.matrix_size = matrix_size
        self._reset_thread = None
        self._reset_obs = None
        self._reset_error = None
        self._env_lock = threading.Lock()

    def _dummy_obs(self):
        return {
            'matrix': np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32),
            'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
        }

    def _idle_info(self, **extra):
        info = {
            'food_eaten': 0,
            'pos': (0, 0),
            'wall_dist': -1,
            'enemy_dist': -1,
            'length': 0,
        }
        info.update(extra)
        return info

    def _do_reset(self):
        try:
            with self._env_lock:
                self._reset_obs = self.env.reset()
        except Exception as e:
            self._reset_error = e

    def _start_reset(self):
        self._reset_obs = None
        self._reset_error = None
        self._reset_thread = threading.Thread(
            target=self._do_reset, daemon=True, name="env-reset",
        )
        self._reset_thread.start()

    def _join_reset(self, timeout=30):
        if self._reset_thread is None:
            return
        self._reset_thread.join(timeout=timeout)
        self._reset_thread = None
        self._reset_obs = None
        self._reset_error = None

    def handle(self, cmd, data):
        if cmd == 'step':
            return self.step(data)
        if cmd == 'reset':
            return self.reset_sync()
        if cmd == 'reset_one':
            return self.reset_async()
        if cmd == 'set_stage':
            self.env.set_curriculum_stage(data)
            return 'ok'
        raise ValueError(f"Unknown worker command: {cmd}")

    def step(self, action):
        if self._reset_thread is not None:
            if self._reset_thread.is_alive():
                return (self._dummy_obs(), 0.0, False, self._idle_info(spawning=True))
            self._reset_thread.join(timeout=1)
            self._reset_thread = None
            if self._reset_error is not None:
                err = self._reset_error
                self._reset_error = None
                raise err
            obs = self._reset_obs if self._reset_obs is not None else self._dummy_obs()
            self._reset_obs = None
            return (obs, 0.0, False, self._idle_info(spawned=True))

        with self._env_lock:
            next_state, reward, done, info = self.env.step(action)
        if done:
            info['terminal_observation'] = next_state
            self._start_reset()
        return (next_state, reward, done, info)

    def reset_sync(self):
        """Startup / full reset — wait for a real observation."""
        self._join_reset()
        with self._env_lock:
            return self.env.reset()

    def reset_async(self):
        """Force-respawn without blocking the vecenv barrier (max-steps)."""
        if self._reset_thread is not None and self._reset_thread.is_alive():
            return self._dummy_obs()
        if self._reset_thread is not None:
            # Reset already finished between commands — return it.
            self._reset_thread.join(timeout=1)
            self._reset_thread = None
            if self._reset_error is not None:
                err = self._reset_error
                self._reset_error = None
                raise err
            obs = self._reset_obs if self._reset_obs is not None else self._dummy_obs()
            self._reset_obs = None
            return obs
        self._start_reset()
        return self._dummy_obs()

    def close(self):
        self._join_reset()
        self.env.close()
