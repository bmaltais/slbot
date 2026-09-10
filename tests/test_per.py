import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from per import PrioritizedReplayBuffer


FRAME = (3, 8, 8)
K = 4
SEC_DIM = 99


def _frame(seed):
    return np.random.default_rng(seed).integers(0, 256, FRAME, dtype=np.uint8)


def _sec(seed):
    return np.random.default_rng(seed).random(SEC_DIM, dtype=np.float32)


def _fill(buf, n_frames, n_transitions, hybrid=True):
    """Push n_frames frames, then transitions t -> stack [t..t+K-1] -> [t+1..t+K]."""
    for i in range(n_frames):
        assert buf.push_frame(_frame(i)) == i
    for t in range(n_transitions):
        s = list(range(t, t + K))
        n = list(range(t + 1, t + K + 1))
        buf.push(s, _sec(t) if hybrid else None, t % 14, float(t), n,
                 _sec(t + 1) if hybrid else None, t % 5 == 0, 0.9)


def test_sample_shapes_and_dtypes():
    buf = PrioritizedReplayBuffer(capacity=64)
    _fill(buf, 50, 40)
    assert len(buf) == 40

    batch, idxs, w = buf.sample(16)
    for key in ('s_mat', 'n_mat'):
        assert isinstance(batch[key], torch.Tensor)
        assert tuple(batch[key].shape) == (16, K * FRAME[0]) + FRAME[1:]
        assert batch[key].dtype == torch.uint8
    assert batch['s_sec'].shape == (16, SEC_DIM) and batch['s_sec'].dtype == np.float32
    assert batch['n_sec'].shape == (16, SEC_DIM) and batch['n_sec'].dtype == np.float32
    assert batch['action'].dtype == np.int64
    for k in ('reward', 'done', 'gamma'):
        assert batch[k].shape == (16,) and batch[k].dtype == np.float32
    assert idxs.shape == (16,) and w.shape == (16,) and w.dtype == np.float32
    assert w.max() == pytest.approx(1.0)


def test_stacks_rebuilt_from_shared_frames():
    """Every sampled row must be the exact concatenation of its referenced frames."""
    buf = PrioritizedReplayBuffer(capacity=32)
    _fill(buf, 40, 32)
    batch, idxs, _ = buf.sample(32)
    slots = idxs - (buf.capacity - 1)
    s_mat = batch['s_mat'].numpy()
    n_mat = batch['n_mat'].numpy()
    for row, slot in enumerate(slots):
        t = int(slot)
        np.testing.assert_array_equal(s_mat[row], np.concatenate([_frame(i) for i in range(t, t + K)]))
        np.testing.assert_array_equal(n_mat[row], np.concatenate([_frame(i) for i in range(t + 1, t + K + 1)]))
        np.testing.assert_array_equal(batch['s_sec'][row], _sec(t))
        np.testing.assert_array_equal(batch['n_sec'][row], _sec(t + 1))
        assert batch['action'][row] == t % 14
        assert batch['reward'][row] == float(t)
        assert batch['done'][row] == (1.0 if t % 5 == 0 else 0.0)
        assert batch['gamma'][row] == pytest.approx(0.9)


def test_memory_is_one_frame_per_transition():
    buf = PrioritizedReplayBuffer(capacity=1000)
    _fill(buf, 1003, 1000)
    frame_bytes = int(np.prod(FRAME))
    assert buf.frames.frame_count == 1003
    # Frame ring: capacity * (1 + headroom) + 64 frames, vs 2 * K frames per
    # transition in the old stacked layout.
    assert buf.frames.nbytes() <= (1000 * 1.125 + 64) * frame_bytes
    assert buf.frames.nbytes() < 1000 * 2 * K * frame_bytes / 4
    # Per-transition metadata (seqs, sectors, scalars) is under 1 KB.
    assert buf.store.nbytes() / 1000 < 1024


def test_frame_ring_wraps_and_transitions_wrap():
    buf = PrioritizedReplayBuffer(capacity=8, frame_capacity=16)
    _fill(buf, 12, 8)
    # 9th transition lands in slot 0; frames 12..15 fill the ring, 16 wraps to slot 0
    for i in range(12, 17):
        buf.push_frame(_frame(i))
    buf.push(list(range(13, 17)), _sec(0), 7, 123.0, list(range(14, 18)), _sec(1), False, 0.5)
    assert len(buf) == 8
    assert buf.store.reward[0] == 123.0
    np.testing.assert_array_equal(buf.frames.frames[0], _frame(16))


def test_stale_transitions_are_dropped_not_sampled():
    buf = PrioritizedReplayBuffer(capacity=16, frame_capacity=8, alpha=1.0)
    for i in range(8):
        buf.push_frame(_frame(i))
    buf.push([0, 1, 2, 3], _sec(0), 1, 1.0, [1, 2, 3, 4], _sec(1), False, 0.9)   # will go stale
    buf.push([4, 5, 6, 7], _sec(0), 2, 2.0, [5, 6, 7, 8], _sec(1), False, 0.9)   # stays valid
    # Overwrite frames 0..3 -> first transition references evicted frames
    for i in range(8, 12):
        buf.push_frame(_frame(i))
    batch, idxs, _ = buf.sample(4)
    assert np.all(batch['action'] == 2)
    assert buf.stale_dropped >= 1
    assert buf.tree.tree[buf.capacity - 1] == 0.0  # slot 0 priority zeroed


def test_priorities_steer_sampling():
    buf = PrioritizedReplayBuffer(capacity=16, alpha=1.0)
    _fill(buf, 20, 16)
    leaf0 = buf.capacity - 1
    buf.update_priorities([leaf0 + i for i in range(16)], np.zeros(16))
    buf.update_priorities([leaf0 + 3], np.array([1000.0]))
    batch, idxs, _ = buf.sample(8)
    assert np.all(idxs == leaf0 + 3)
    assert np.all(batch['reward'] == 3.0)


def test_legacy_transitions_without_sectors():
    buf = PrioritizedReplayBuffer(capacity=8)
    _fill(buf, 12, 8, hybrid=False)
    batch, _, _ = buf.sample(4)
    assert tuple(batch['s_mat'].shape) == (4, K * FRAME[0]) + FRAME[1:]
    assert batch['s_sec'] is None and batch['n_sec'] is None


def test_rejects_non_uint8_frames():
    buf = PrioritizedReplayBuffer(capacity=4)
    with pytest.raises(TypeError):
        buf.push_frame(np.zeros(FRAME, dtype=np.float32))


def test_staging_tensor_is_reused_and_pinned_when_cuda():
    buf = PrioritizedReplayBuffer(capacity=8)
    _fill(buf, 12, 8)
    first, _, _ = buf.sample(4)
    second, _, _ = buf.sample(4)
    assert first['s_mat'].data_ptr() == second['s_mat'].data_ptr()
    assert first['s_mat'].is_pinned() == torch.cuda.is_available()
