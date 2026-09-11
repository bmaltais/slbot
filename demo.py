"""Human demonstration episodes: the on-disk format and how they replay.

``record_demo.py`` writes one file per episode with ``DemoEpisode``; the
trainer reads them back with ``load_demo`` and feeds them to the agent as
frame-stacked transitions via ``iter_stacked_transitions`` — the same shape
``VecFrameStack`` produces for live play, so a demonstration and a live step
are indistinguishable to the replay buffer and the loss.

File layout (``.npz``):
    frames   (T+1, 3, H, W) uint8   observation frames; frames[0] is reset
    sectors  (T+1, S)       float32 sector radar per frame
    actions  (T,)           int64   human action label per step
    rewards  (T,)           float32 reward the env computed for the step
    dones    (T,)           bool    True on the terminal step (death)
    meta     0-d str        JSON: peak_length, steps, total_reward, cause,
                             stage, style, frame_skip, matrix_size, ...
"""
import glob
import json
import os
import time

import numpy as np

DEMO_FORMAT_VERSION = 1
DEMO_SUFFIX = '.npz'
DEFAULT_DEMO_DIR = 'demos'


class DemoEpisode:
    """Accumulates one episode of human play, then saves it."""

    def __init__(self, first_obs):
        self.frames = [np.asarray(first_obs['matrix'], dtype=np.uint8)]
        self.sectors = [np.asarray(first_obs['sectors'], dtype=np.float32)]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.peak_length = 0
        self.started_at = time.time()

    def add(self, action, reward, next_obs, done, length=0):
        self.frames.append(np.asarray(next_obs['matrix'], dtype=np.uint8))
        self.sectors.append(np.asarray(next_obs['sectors'], dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.peak_length = max(self.peak_length, int(length or 0))

    def __len__(self):
        return len(self.actions)

    @property
    def total_reward(self):
        return float(sum(self.rewards))

    def save(self, out_dir, **meta):
        """Write the episode to ``out_dir``. Returns the file path."""
        if not self.actions:
            raise ValueError("cannot save an empty demo episode")
        os.makedirs(out_dir, exist_ok=True)
        info = {
            'format': DEMO_FORMAT_VERSION,
            'steps': len(self),
            'peak_length': self.peak_length,
            'total_reward': self.total_reward,
            'recorded_at': self.started_at,
            'duration_s': time.time() - self.started_at,
            'matrix_size': int(self.frames[0].shape[-1]),
        }
        info.update(meta)
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.started_at))
        name = f"demo_{stamp}_len{self.peak_length}_steps{len(self)}{DEMO_SUFFIX}"
        path = os.path.join(out_dir, name)
        np.savez_compressed(
            path,
            frames=np.stack(self.frames),
            sectors=np.stack(self.sectors),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=bool),
            meta=np.array(json.dumps(info)),
        )
        return path


def load_demo(path):
    """Read one demo file into a dict of arrays plus ``meta``."""
    with np.load(path, allow_pickle=False) as z:
        demo = {
            'frames': z['frames'],
            'sectors': z['sectors'],
            'actions': z['actions'],
            'rewards': z['rewards'],
            'dones': z['dones'],
            'meta': json.loads(str(z['meta'])),
        }
    t = len(demo['actions'])
    if demo['frames'].shape[0] != t + 1 or demo['sectors'].shape[0] != t + 1:
        raise ValueError(f"{path}: expected {t + 1} frames for {t} steps")
    return demo


def list_demos(demo_dir):
    """All demo files under ``demo_dir``, oldest first."""
    return sorted(glob.glob(os.path.join(demo_dir, f"*{DEMO_SUFFIX}")))


def read_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z['meta']))


def select_demos(paths, min_score=0):
    """Split demo paths into (kept, skipped) by peak snake length."""
    kept, skipped = [], []
    for p in paths:
        meta = read_meta(p)
        (kept if meta.get('peak_length', 0) >= min_score else skipped).append(p)
    return kept, skipped


def iter_stacked_transitions(demo, frame_stack):
    """Yield (state, action, reward, next_state, done) with frame-stacked obs.

    Mirrors VecFrameStack: the reset frame fills the whole stack, then each
    step appends the next frame. next_state of step t is the same object as
    state of step t+1, which DDQNAgent.remember_nstep relies on to store
    each frame once.
    """
    frames = demo['frames']
    sectors = demo['sectors']
    k = int(frame_stack)
    stack = [frames[0]] * k
    state = {'matrix': np.concatenate(stack, axis=0), 'sectors': sectors[0]}
    for t in range(len(demo['actions'])):
        stack = stack[1:] + [frames[t + 1]]
        next_state = {'matrix': np.concatenate(stack, axis=0), 'sectors': sectors[t + 1]}
        yield state, int(demo['actions'][t]), float(demo['rewards'][t]), next_state, bool(demo['dones'][t])
        state = next_state
