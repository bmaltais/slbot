import numpy as np
import random
import torch


class SumTree:
    """
    SumTree structure for efficient storage and sampling of prioritized experience.
    Leaf nodes store priorities. Internal nodes store sum of children.
    Leaf i (0-based data slot) lives at tree index i + capacity - 1.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree size is 2 * capacity - 1
        # Indices for leaves start at capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p):
        """Assign priority p to the next write slot. Returns the data slot written."""
        slot = self.write
        idx = slot + self.capacity - 1
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.count < self.capacity:
            self.count += 1
        return slot

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Returns (tree_idx, priority, data_slot) for cumulative priority s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], data_idx)


class FrameStore:
    """Ring of single observation frames, each stored once.

    A stacked observation (frame_stack * C, H, W) is not stored as one block.
    Each new frame (C, H, W) uint8 gets a monotonically increasing sequence
    number and transitions reference `stack` sequence numbers per state. With
    frame_stack=4 that is ~8x less memory than storing two full stacks per
    transition (601 KB -> ~78 KB at 160x160), and it removes the repeated
    uint8 conversion of the same frame in every stack position.

    Sequence s lives in slot s % capacity and stays valid while
    s >= frame_count - capacity. Transitions carry the minimum sequence they
    reference so the replay buffer can detect (and drop) ones whose frames
    have been overwritten.

    Stack batches are gathered with torch.index_select (multithreaded) into a
    reusable pinned staging tensor when CUDA is present, so the host->device
    copy can be asynchronous. The tensor returned by gather_stacks() is
    therefore valid only until the next call with the same key/batch size.
    """

    def __init__(self, capacity, frame, pin_memory=None):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            raise TypeError(f"replay frames must be uint8, got {frame.dtype}")
        self.capacity = int(capacity)
        self.frame_shape = tuple(frame.shape)
        # np.zeros is a lazy mmap on Linux: pages are only committed as slots
        # are written, so an empty buffer costs no resident memory.
        self.frames = np.zeros((self.capacity,) + self.frame_shape, dtype=np.uint8)
        self._frames_t = torch.from_numpy(self.frames)  # zero-copy view
        self.frame_count = 0
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self._pin_memory = bool(pin_memory)
        self._staging = {}  # key -> staging tensor (one per key)

    def push(self, frame):
        """Store one frame. Returns its sequence number."""
        seq = self.frame_count
        self.frames[seq % self.capacity] = frame
        self.frame_count += 1
        return seq

    def oldest_valid_seq(self):
        return self.frame_count - self.capacity

    def _out(self, key, n_frames):
        # One staging tensor per key, replaced when the batch size changes,
        # so the cache cannot grow with varying batch sizes.
        out = self._staging.get(key)
        if out is None or out.shape[0] != n_frames:
            out = torch.empty((n_frames,) + self.frame_shape, dtype=torch.uint8,
                              pin_memory=self._pin_memory)
            self._staging[key] = out
        return out

    def gather_stacks(self, seqs, key):
        """seqs: (B, k) int64 sequence numbers -> (B, k*C, H, W) uint8 tensor."""
        seqs = np.ascontiguousarray(seqs, dtype=np.int64)
        b, k = seqs.shape
        slots = torch.from_numpy((seqs % self.capacity).reshape(-1))
        out = torch.index_select(self._frames_t, 0, slots, out=self._out(key, b * k))
        c, h, w = self.frame_shape
        return out.view(b, k * c, h, w)

    def nbytes(self):
        return self.frames.nbytes


class _TransitionStore:
    """Preallocated, typed arrays for transitions (one row per replay slot).

    Matrices are not stored here: each state is `stack` frame sequence numbers
    into a FrameStore. Sector vectors are small and stored per transition.
    """

    def __init__(self, capacity, stack, s_sec):
        self.capacity = capacity
        self.stack = int(stack)
        self.hybrid = s_sec is not None
        self.s_seq = np.zeros((capacity, self.stack), dtype=np.int64)
        self.n_seq = np.zeros((capacity, self.stack), dtype=np.int64)
        # Minimum frame sequence referenced by the row; used for staleness.
        self.min_seq = np.full(capacity, -1, dtype=np.int64)
        if self.hybrid:
            sec = np.asarray(s_sec, dtype=np.float32)
            self.s_sec = np.zeros((capacity,) + sec.shape, dtype=np.float32)
            self.n_sec = np.zeros((capacity,) + sec.shape, dtype=np.float32)
        else:
            self.s_sec = None
            self.n_sec = None
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.gamma = np.zeros(capacity, dtype=np.float32)

    def write(self, slot, s_seq, s_sec, action, reward, n_seq, n_sec, done, gamma):
        self.s_seq[slot] = s_seq
        self.n_seq[slot] = n_seq
        self.min_seq[slot] = min(int(min(s_seq)), int(min(n_seq)))
        if self.hybrid:
            self.s_sec[slot] = s_sec
            self.n_sec[slot] = n_sec
        self.action[slot] = action
        self.reward[slot] = reward
        self.done[slot] = float(done)
        self.gamma[slot] = gamma

    def gather(self, slots, frames):
        """Batch dict for the given data slots.

        s_mat / n_mat are uint8 CPU torch tensors (pinned when CUDA is
        available) that are overwritten by the next gather(); everything else
        is a fresh numpy array.
        """
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        return {
            's_mat': frames.gather_stacks(self.s_seq[slots], 's'),
            'n_mat': frames.gather_stacks(self.n_seq[slots], 'n'),
            's_sec': self.s_sec[slots] if self.hybrid else None,
            'n_sec': self.n_sec[slots] if self.hybrid else None,
            'action': self.action[slots],
            'reward': self.reward[slots],
            'done': self.done[slots],
            'gamma': self.gamma[slots],
        }

    def nbytes(self):
        arrays = [self.s_seq, self.n_seq, self.min_seq, self.action, self.reward, self.done, self.gamma]
        if self.hybrid:
            arrays += [self.s_sec, self.n_sec]
        return sum(a.nbytes for a in arrays)


class PrioritizedReplayBuffer:
    # Frames per transition is ~1 in steady state (one new frame per env step)
    # plus a few extra at episode starts, so the frame ring gets this much
    # headroom over the transition ring before stale references can occur.
    FRAME_HEADROOM = 0.125

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000,
                 frame_capacity=None, pin_memory=None):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 # Start at 1 to avoid div by zero if logic changes
        if frame_capacity is None:
            frame_capacity = int(capacity * (1.0 + self.FRAME_HEADROOM)) + 64
        self.frame_capacity = int(frame_capacity)
        self._pin_memory = pin_memory
        self.frames = None  # FrameStore, allocated on first push_frame
        self.store = None   # _TransitionStore, allocated on first push
        self.stale_dropped = 0

    # --- frames -----------------------------------------------------------

    def push_frame(self, frame):
        """Store one (C, H, W) uint8 frame. Returns its sequence number."""
        if self.frames is None:
            self.frames = FrameStore(self.frame_capacity, frame, pin_memory=self._pin_memory)
        return self.frames.push(frame)

    # --- transitions ------------------------------------------------------

    def push(self, s_seq, s_sec, action, reward, n_seq, n_sec, done, gamma):
        """Save a transition whose states are frame sequence lists (len = frame_stack)."""
        if self.store is None:
            self.store = _TransitionStore(self.capacity, len(s_seq), s_sec)

        # Max priority for new items ensures they get replayed at least once
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0

        slot = self.tree.add(max_p)
        self.store.write(slot, s_seq, s_sec, action, reward, n_seq, n_sec, done, gamma)

    def _walk(self, batch_size):
        idxs = np.empty(batch_size, dtype=np.int64)
        slots = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, slot) = self.tree.get(s)

            priorities[i] = p
            idxs[i] = idx
            slots[i] = slot
        return idxs, slots, priorities

    def sample(self, batch_size, max_passes=64):
        """Returns (batch, tree_idxs, is_weights).

        batch is a dict: s_mat / n_mat (B, frame_stack*C, H, W) uint8 CPU
        torch tensors (reused by the next sample() call, so consume them
        first), s_sec / n_sec (B, S) float32 numpy or None, action int64,
        reward / done / gamma float32 numpy.

        Transitions whose frames were overwritten in the frame ring get
        priority 0 and the batch is re-drawn; this only happens if episodes
        are so short that frames outpace transitions by more than
        FRAME_HEADROOM. A batch is never returned with stale rows: every pass
        zeroes at least one stale priority, so the loop terminates, and it
        raises if the buffer runs out of valid transitions or max_passes.
        """
        if self.frames is None or self.store is None or len(self) == 0:
            raise ValueError("replay buffer is empty; push frames and transitions before sample()")

        # Calculate current beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += batch_size # Advance frame count

        oldest = self.frames.oldest_valid_seq()
        for _ in range(max_passes):
            idxs, slots, priorities = self._walk(batch_size)
            stale = self.store.min_seq[slots] < oldest
            if not stale.any():
                break
            for idx in idxs[stale]:
                self.tree.update(int(idx), 0.0)
            self.stale_dropped += int(stale.sum())
            if self.tree.total() <= 0:
                raise RuntimeError("replay buffer has no valid transitions left")
        else:
            raise RuntimeError(
                f"could not draw a batch without stale transitions in {max_passes} passes"
            )

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.total() * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return self.store.gather(slots, self.frames), idxs, is_weight.astype(np.float32)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + 1e-5) ** self.alpha
            self.tree.update(idx, p)

    def nbytes(self):
        total = 0
        if self.frames is not None:
            total += self.frames.nbytes()
        if self.store is not None:
            total += self.store.nbytes()
        return total

    def __len__(self):
        return self.tree.count
