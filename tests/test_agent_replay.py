"""DDQNAgent replay path: n-step returns over deduplicated frames.

Drives remember_nstep() the way trainer.py + VecFrameStack do: stacked
observations built from a deque of per-step frames, the previous call's
next_state object passed back as the next call's state, and a terminal stack
on done.
"""
import os
import sys
from collections import deque

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import DDQNAgent
from config import Config


RES = 64
K = 4


@pytest.fixture
def agent():
    cfg = Config()
    cfg.env.resolution = (RES, RES)
    cfg.env.frame_stack = K
    cfg.opt.batch_size = 8
    cfg.buffer.capacity = 256
    a = DDQNAgent(cfg)
    a.current_gamma = 0.5
    return a


def _frame(seed):
    return np.random.default_rng(seed).random((3, RES, RES), dtype=np.float32)


class _Stacker:
    """Mirror of VecFrameStack for one agent."""

    def __init__(self):
        self.frames = deque(maxlen=K)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(K):
            self.frames.append(frame)
        return self.obs()

    def step(self, frame):
        self.frames.append(frame)
        return self.obs()

    def obs(self):
        return {'matrix': np.concatenate(list(self.frames), axis=0),
                'sectors': np.zeros(99, dtype=np.float32)}


def _q(mat):
    return (mat * 255).astype(np.uint8)


def _run_episode(agent, agent_id, n_steps, seed0, rewards=None):
    """Returns the list of stacked observations obs[0..n_steps] (obs[n] terminal)."""
    st = _Stacker()
    obs = [st.reset(_frame(seed0))]
    state = obs[0]
    for t in range(n_steps):
        nxt = st.step(_frame(seed0 + 1 + t))
        obs.append(nxt)
        r = rewards[t] if rewards else 1.0
        agent.remember_nstep(state, t % 14, r, nxt, t == n_steps - 1, agent_id=agent_id)
        state = nxt
    return obs


def test_one_frame_stored_per_step(agent):
    _run_episode(agent, 0, 20, seed0=100)
    # 1 frame for the reset stack (4 identical copies dedup to 1) + 20 steps
    assert agent.memory.frames.frame_count == 21
    assert len(agent.memory) == 20  # every step becomes a transition after flush


def test_sampled_stacks_match_stacked_observations(agent):
    obs = _run_episode(agent, 0, 12, seed0=200, rewards=[float(t) for t in range(12)])
    n = agent.n_step
    batch, idxs, _ = agent.memory.sample(len(agent.memory))
    slots = idxs - (agent.memory.capacity - 1)
    s_mat = batch['s_mat'].numpy()
    n_mat = batch['n_mat'].numpy()
    for row, slot in enumerate(slots):
        t = int(slot)  # transitions are pushed in step order for a single agent
        np.testing.assert_array_equal(s_mat[row], _q(obs[t]['matrix']))
        last = min(t + n, 12)
        np.testing.assert_array_equal(n_mat[row], _q(obs[last]['matrix']))
        expected_R = sum((0.5 ** i) * float(t + i) for i in range(last - t))
        assert batch['reward'][row] == pytest.approx(expected_R)
        assert batch['done'][row] == (1.0 if last == 12 else 0.0)
        assert batch['action'][row] == t % 14


def test_episode_boundary_restarts_stack(agent):
    _run_episode(agent, 0, 5, seed0=300)
    _run_episode(agent, 0, 5, seed0=400)
    assert agent.memory.frames.frame_count == 12  # (1 + 5) * 2
    assert len(agent.memory) == 10
    # No transition may mix frames from both episodes: seq ranges are disjoint.
    s = agent.memory.store.s_seq[:10]
    n = agent.memory.store.n_seq[:10]
    assert np.all((n.max(axis=1) < 6) | (s.min(axis=1) >= 6))


def test_interleaved_agents_keep_separate_stacks(agent):
    a = _Stacker()
    b = _Stacker()
    sa = a.reset(_frame(1))
    sb = b.reset(_frame(1000))
    for t in range(6):
        na = a.step(_frame(2 + t))
        nb = b.step(_frame(1001 + t))
        agent.remember_nstep(sa, 0, 1.0, na, t == 5, agent_id=0)
        agent.remember_nstep(sb, 1, 1.0, nb, t == 5, agent_id=1)
        sa, sb = na, nb
    assert len(agent.memory) == 12
    batch, _, _ = agent.memory.sample(12)
    # Agent 0 frames are seeds 1..7, agent 1 are 1000..1006; a stack must not mix them.
    s_mat = batch['s_mat'].numpy()
    for row in range(12):
        first = s_mat[row][:3]
        last = s_mat[row][-3:]
        seeds_a = [_q(_frame(i)) for i in range(1, 8)]
        in_a_first = any(np.array_equal(first, f) for f in seeds_a)
        in_a_last = any(np.array_equal(last, f) for f in seeds_a)
        assert in_a_first == in_a_last


def test_discontinuity_flushes_partial_trajectory(agent):
    """A new state object that is not the previous next_state (e.g. a re-added
    agent slot) must not be stitched onto the old n-step buffer."""
    st = _Stacker()
    s = st.reset(_frame(10))
    n1 = st.step(_frame(11))
    agent.remember_nstep(s, 0, 1.0, n1, False, agent_id=0)   # buffer has 1 entry, not yet pushed
    assert len(agent.memory) == 0
    st2 = _Stacker()
    s2 = st2.reset(_frame(50))
    n2 = st2.step(_frame(51))
    agent.remember_nstep(s2, 0, 1.0, n2, False, agent_id=0)  # discontinuity
    assert len(agent.memory) == 1                             # old entry flushed, bootstrapped
    assert agent.memory.store.done[0] == 0.0
    assert agent.memory.store.reward[0] == 1.0


def test_optimize_model_runs_on_dedup_buffer(agent):
    _run_episode(agent, 0, 30, seed0=700)
    metrics = agent.optimize_model()
    assert metrics is not None
    assert np.isfinite(metrics['loss'])


def test_quantize_clamps_out_of_range_values():
    q = DDQNAgent._quantize(np.array([-0.5, 0.0, 0.5, 1.0, 1.7], dtype=np.float32))
    np.testing.assert_array_equal(q, np.array([0, 0, 127, 255, 255], dtype=np.uint8))


def test_remember_full_stacks_still_works(agent):
    st = _Stacker()
    s = st.reset(_frame(1))
    n = st.step(_frame(2))
    agent.remember(s, 3, 2.0, n, False)
    assert len(agent.memory) == 1
    # reset stack dedups to 1; the next stack is stored independently (3 identical + 1 new -> 2)
    assert agent.memory.frames.frame_count == 3
    batch, _, _ = agent.memory.sample(1)
    np.testing.assert_array_equal(batch['s_mat'].numpy()[0], _q(s['matrix']))
    np.testing.assert_array_equal(batch['n_mat'].numpy()[0], _q(n['matrix']))
