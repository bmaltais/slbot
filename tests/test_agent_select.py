"""DDQNAgent.select_actions: batched greedy picks over uint8 observations."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ACTION_DIM, DDQNAgent
from config import Config

RES = 64  # smallest size the hybrid net's four conv layers accept
K = 4


def _agent(eps):
    cfg = Config()
    cfg.env.resolution = (RES, RES)
    cfg.env.frame_stack = K
    cfg.opt.eps_start = eps
    cfg.opt.eps_end = eps
    return DDQNAgent(cfg)


@pytest.fixture
def greedy():
    return _agent(0.0)


def _calm_sectors():
    s = np.zeros(99, dtype=np.float32)
    s[48:72] = -1.0
    s[96] = 1.0  # far from wall
    return s


def _obs(seed):
    rng = np.random.default_rng(seed)
    return {'matrix': rng.integers(0, 256, (3 * K, RES, RES), dtype=np.uint8),
            'sectors': _calm_sectors()}


def _spy(agent):
    """Wrap policy_net so the batch sizes it sees are recorded."""
    real = agent.policy_net
    sizes = []

    def net(m, s=None):
        sizes.append(int(m.shape[0]))
        return real(m, s) if s is not None else real(m)

    agent.policy_net = net
    return sizes


def test_batched_actions_match_per_agent_greedy(greedy):
    states = [_obs(i) for i in range(5)]
    single = [greedy.select_action(s, agent_id=i) for i, s in enumerate(states)]
    sizes = _spy(greedy)
    batched = greedy.select_actions(states, list(range(5)))
    assert batched == single
    assert sizes == [5]
    # and both equal a plain float32 forward
    with torch.no_grad():
        m = torch.from_numpy(np.stack([s['matrix'] for s in states])).to(greedy.device).float() / 255.0
        sec = torch.from_numpy(np.stack([s['sectors'] for s in states])).to(greedy.device)
        ref = greedy.policy_net(m, sec).argmax(1).tolist()
    assert batched == ref


def test_uint8_and_float_observations_agree(greedy):
    u8 = [_obs(i) for i in range(3)]
    f32 = [{'matrix': s['matrix'].astype(np.float32) / 255.0, 'sectors': s['sectors']} for s in u8]
    assert greedy.select_actions(u8) == greedy.select_actions(f32)


def test_reflex_agents_bypass_the_network(greedy):
    states = [_obs(0), _obs(1), _obs(2)]
    states[1]['sectors'][24] = 0.9  # obstacle dead ahead -> u-turn reflex
    sizes = _spy(greedy)
    actions = greedy.select_actions(states, [0, 1, 2])
    assert actions[1] in (9, 10)
    assert sizes == [2]
    stats = greedy.reflex_stats[1]
    assert stats['reflex_actions'] == 1 and stats['front'] == 1 and stats['total_actions'] == 1
    assert greedy.reflex_stats[0]['reflex_actions'] == 0


def test_random_epsilon_skips_the_network():
    agent = _agent(1.0)
    sizes = _spy(agent)
    actions = agent.select_actions([_obs(i) for i in range(4)])
    assert len(actions) == 4 and all(0 <= a < ACTION_DIM for a in actions)
    assert sizes == []


def test_empty_batch(greedy):
    assert greedy.select_actions([], []) == []


def test_agent_ids_default_to_positions(greedy):
    greedy.select_actions([_obs(0), _obs(1)])
    assert set(greedy.reflex_stats) >= {0, 1}


def test_matrices_to_device_scales_uint8(greedy):
    t = greedy._matrices_to_device(np.full((2, 3 * K, RES, RES), 255, dtype=np.uint8))
    assert t.dtype == torch.float32 and float(t.min()) == 1.0 and float(t.max()) == 1.0
    z = greedy._matrices_to_device(np.zeros((1, 3 * K, RES, RES), dtype=np.uint8))
    assert float(z.max()) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging is CUDA-only")
def test_staging_buffer_grows_to_largest_batch(greedy):
    for n in (1, 3, 2):
        greedy._matrices_to_device(np.zeros((n, 3 * K, RES, RES), dtype=np.uint8))
        torch.cuda.synchronize()
    assert greedy._act_staging.shape[0] == 3
    assert greedy._act_staging.is_pinned()


def test_quantize_passes_uint8_through():
    a = np.arange(4, dtype=np.uint8)
    assert DDQNAgent._quantize(a) is a
