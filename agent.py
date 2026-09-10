import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import logging
from collections import deque

from config import Config
from model import DuelingDQN, HybridDuelingDQN
from per import PrioritizedReplayBuffer

logger = logging.getLogger("slitherbot")
ACTION_DIM = 14

class DDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.reflex5_enabled = False  # Body encirclement reflex (off by default)
        self.saved_best_fitness = None
        self.saved_best_avg_reward = None

        # Device selection: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Agent running on device: {self.device}")

        if self.device.type == 'cuda':
            # Observation shapes are fixed (batch size varies only with the
            # number of live agents): let cuDNN benchmark conv kernels once
            # per shape instead of using its heuristic pick every call.
            torch.backends.cudnn.benchmark = True
        # Pinned host buffer for shipping uint8 observation batches to the
        # GPU in select_actions() (grown to the largest batch seen).
        self._act_staging = None

        # Calculate input channels (3 base channels * frame_stack)
        self.input_channels = 3 * config.env.frame_stack
        self.input_size = config.env.resolution
        self.use_hybrid = config.model.architecture == 'HybridDuelingDQN'

        if self.use_hybrid:
            self.policy_net = HybridDuelingDQN(
                input_channels=self.input_channels,
                action_dim=ACTION_DIM,
                input_size=self.input_size,
                sector_dim=config.model.sector_dim,
            ).to(self.device)
            self.target_net = HybridDuelingDQN(
                input_channels=self.input_channels,
                action_dim=ACTION_DIM,
                input_size=self.input_size,
                sector_dim=config.model.sector_dim,
            ).to(self.device)
        else:
            self.policy_net = DuelingDQN(
                input_channels=self.input_channels,
                action_dim=ACTION_DIM,
                input_size=self.input_size,
            ).to(self.device)
            self.target_net = DuelingDQN(
                input_channels=self.input_channels,
                action_dim=ACTION_DIM,
                input_size=self.input_size,
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config.opt.lr,
            weight_decay=config.opt.weight_decay
        )

        # LR scheduler removed — ReduceLROnPlateau is incompatible with RL
        # (rolling reward is too noisy to detect a real plateau early in training)

        if config.buffer.prioritized:
            self.memory = PrioritizedReplayBuffer(
                capacity=config.buffer.capacity,
                alpha=config.buffer.alpha,
                beta_start=config.buffer.beta_start,
                beta_frames=config.buffer.beta_frames
            )
        else:
            # Fallback to simple deque if prioritized is disabled (not implemented in per.py but keeping structure open)
            raise NotImplementedError("Only Prioritized Buffer is supported currently.")

        self.steps_done = 0

        # Dynamic gamma (set per curriculum stage)
        self.current_gamma = config.opt.gamma

        # N-step returns
        self.n_step = 3
        self.n_step_buffers = {}  # per-agent buffers: {agent_id: deque}

        # Frame dedup: each env step stores one new frame in the replay
        # buffer; a stacked state is `frame_stack` frame sequence numbers.
        # Per agent: the next_state object seen last and its frame seqs, so
        # the following remember_nstep() call can extend the stack by one
        # frame instead of storing all of them again.
        self.frame_stack = config.env.frame_stack
        self._stack_tracker = {}  # agent_id -> (next_state_obj, [seq] * frame_stack)

        # Reward Normalization Stats
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-5 # avoid div by zero
        self.reflex_stats = {}

    def get_epsilon(self):
        return self.config.opt.eps_end + (self.config.opt.eps_start - self.config.opt.eps_end) * \
            np.exp(-1. * self.steps_done / self.config.opt.eps_decay)

    def step_scheduler(self, metric):
        """No-op: LR scheduler removed (incompatible with RL)."""
        pass

    def boost_exploration(self, target_eps=0.5):
        """Resets steps_done to boost epsilon back to target_eps."""
        start = self.config.opt.eps_start
        end = self.config.opt.eps_end
        decay = self.config.opt.eps_decay

        # Clamp target to be valid
        target_eps = max(end + 0.01, min(start, target_eps))

        ratio = (target_eps - end) / (start - end)
        if ratio <= 0:
            new_steps = decay * 10
        else:
            new_steps = -decay * np.log(ratio)

        logger.info(f"  Boosting Exploration: Eps {self.get_epsilon():.3f} -> {target_eps:.3f} (Reset steps to {int(new_steps)})")
        self.steps_done = int(new_steps)

    def _stack_frames(self, frames):
        """
        Stacks list of frames into a single numpy array.
        Each frame is (3, H, W).
        Output is (12, H, W).
        """
        return np.concatenate(frames, axis=0)

    def _ensure_reflex_stats(self, agent_id):
        if agent_id not in self.reflex_stats:
            self.reflex_stats[agent_id] = {
                'total_actions': 0,
                'reflex_actions': 0,
                'front': 0,
                'wall': 0,
                'head': 0,
                'converge': 0,
                'encircle': 0,
            }
        return self.reflex_stats[agent_id]

    def consume_episode_stats(self, agent_id=0):
        stats = dict(self._ensure_reflex_stats(agent_id))
        self.reflex_stats[agent_id] = {
            'total_actions': 0,
            'reflex_actions': 0,
            'front': 0,
            'wall': 0,
            'head': 0,
            'converge': 0,
            'encircle': 0,
        }
        return stats

    def select_action(self, state, agent_id=0):
        """Single-observation form of select_actions()."""
        return self.select_actions([state], [agent_id])[0]

    def select_actions(self, states, agent_ids=None):
        """
        One action per observation, with a single batched forward pass.

        states: list of dicts {'matrix': (12, H, W) uint8 or float32 [0, 1],
        'sectors': (99,)}; the legacy (non-hybrid) model also accepts bare
        (12, H, W) arrays. agent_ids, when given, must be one per state.
        Reflexes and epsilon-random picks are resolved per agent on the CPU
        (cheap numpy on the sector vector); only the agents that fall through
        to the network are stacked, copied to the device as one batch and
        read back with one sync — instead of one 1.2 MB copy + sync per agent.

        Note: steps_done is incremented externally by the trainer (once per batch step)
        to avoid N× decay with N parallel agents.
        """
        n = len(states)
        if agent_ids is None:
            agent_ids = range(n)
        else:
            agent_ids = list(agent_ids)
            if len(agent_ids) != n:
                raise ValueError(
                    f"select_actions: {n} states but {len(agent_ids)} agent_ids"
                )
        if self.use_hybrid and any(not isinstance(s, dict) for s in states):
            raise ValueError(
                "select_actions: the hybrid model needs dict observations with 'sectors'"
            )
        actions = [None] * n
        net_idx = []
        eps_threshold = self.get_epsilon()

        for i, (state, agent_id) in enumerate(zip(states, agent_ids)):
            # --- REFLEX LAYER ---
            # Hardcoded survival reflexes that override the network when danger is imminent.
            # The network still learns from the outcomes — reflexes just keep the bot alive
            # long enough to generate useful training data.
            stats = self._ensure_reflex_stats(agent_id)
            stats['total_actions'] += 1

            if isinstance(state, dict) and 'sectors' in state:
                reflex_action, reflex_name = self._check_reflexes(state['sectors'])
                if reflex_action is not None:
                    stats['reflex_actions'] += 1
                    if reflex_name in stats:
                        stats[reflex_name] += 1
                    actions[i] = reflex_action
                    continue

            if random.random() > eps_threshold:
                net_idx.append(i)
            else:
                actions[i] = random.randrange(ACTION_DIM)

        if net_idx:
            for i, a in zip(net_idx, self._greedy_actions([states[i] for i in net_idx])):
                actions[i] = a
        return actions

    def _greedy_actions(self, states):
        """argmax_a Q(s, a) for a batch of observations: one forward, one sync."""
        mats = [s['matrix'] if isinstance(s, dict) else s for s in states]
        with torch.no_grad():
            mat_t = self._matrices_to_device(np.stack(mats))
            if self.use_hybrid:
                secs = np.stack([np.asarray(s['sectors'], dtype=np.float32) for s in states])
                q_values = self.policy_net(mat_t, torch.from_numpy(secs).to(self.device))
            else:
                q_values = self.policy_net(mat_t)
            return q_values.argmax(1).tolist()

    def _matrices_to_device(self, batch):
        """(B, C, H, W) numpy batch -> float32 [0, 1] tensor on the device.

        uint8 frames travel as uint8 (4x less host->device traffic; pinned
        and async on CUDA) and are scaled on the device. Float input is
        assumed to be in [0, 1] already.
        """
        if batch.dtype != np.uint8:
            return torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        src = torch.from_numpy(batch)
        if self.device.type == 'cuda':
            stage = self._act_staging
            if stage is None or stage.shape[0] < src.shape[0] or stage.shape[1:] != src.shape[1:]:
                stage = torch.empty(src.shape, dtype=torch.uint8, pin_memory=True)
                self._act_staging = stage
            # Safe to reuse every call: the caller syncs on the result before
            # the next batch is written here.
            src = stage[:src.shape[0]]
            src.copy_(torch.from_numpy(batch))
        return src.to(self.device, non_blocking=True).float().div_(255.0)

    def _check_reflexes(self, sectors):
        """
        Emergency reflexes based on sector vector. Returns (action, reflex_name) or (None, None).

        Sector layout (99 floats, alpha-5):
          [0..23]   food_score per sector (0=ahead, clockwise 15° each)
          [24..47]  obstacle_score per sector (1.0=touching, 0.0=clear)
          [48..71]  obstacle_type per sector (-1=none, 0=body/wall, 1=head)
          [72..95]  enemy_approach per sector (dot product, -1..+1)
          [96]      wall_dist_norm (dist_to_wall / 2000)
          [97]      snake_length_norm
          [98]      speed_norm

        Actions: 0=straight, 1/2=micro L/R, 3/4=gentle L/R, 5/6=medium L/R,
                 7/8=sharp L/R, 9/10=uturn L/R, 11=boost, 12/13=boost+micro L/R
        """
        ns = 24  # num_sectors
        obstacle = sectors[ns:ns*2]       # obstacle_score per sector
        obs_type = sectors[ns*2:ns*3]     # obstacle_type per sector
        # Globals start at index ns*4 (after 4 per-sector features)
        wall_norm = sectors[ns * 4]       # [96] wall_dist_norm

        # --- REFLEX 1: Obstacle directly ahead (sectors 0, 23 = front ±15°) ---
        # If something is close in front, turn away hard
        front_danger = max(obstacle[0], obstacle[23], obstacle[1])
        if front_danger > 0.55:  # lowered from 0.72 for earlier reaction
            # Pick the safer side — check left vs right obstacle density
            # Left = sectors 20-23 (−60° to 0°), Right = sectors 1-4 (0° to +60°)
            left_danger = sum(obstacle[20:24]) / 4.0
            right_danger = sum(obstacle[1:5]) / 4.0

            if front_danger > 0.85:  # lowered from 0.90 — U-turn sooner
                return (9 if left_danger <= right_danger else 10), 'front'
            else:  # Medium close — sharp turn
                return (7 if left_danger <= right_danger else 8), 'front'

        # --- REFLEX 2: Wall proximity emergency ---
        # wall_norm < 0.20 means within 400 units of wall (out of 2000 scope)
        if wall_norm < 0.20: # increased from 0.10 for safer margin
            # Turn toward center — check which side has more open space
            left_obs = sum(obstacle[18:24]) / 6.0
            right_obs = sum(obstacle[0:6]) / 6.0
            return (9 if left_obs <= right_obs else 10), 'wall'

        # --- REFLEX 3: Enemy head approaching from front ---
        # Enemy heads (type=1) in front sectors are the most dangerous
        for s_i in [0, 23, 1, 22]:  # front ±30°
            if obs_type[s_i] == 1 and obstacle[s_i] > 0.45:  # lowered from 0.55
                left_danger = sum(obstacle[20:24]) / 4.0
                right_danger = sum(obstacle[1:5]) / 4.0
                return (7 if left_danger <= right_danger else 8), 'head'

        # --- REFLEX 4: Converging trajectories (enemy approaching at angle) ---
        # Detects enemies in front-side sectors (±15°..60°) heading toward us.
        # enemy_approach > 0.5 = heading our way, obstacle > 0.3 = within ~1400 units
        # This catches the "slight angle collision" that REFLEX 1/3 miss.
        enemy_approach = sectors[ns*3:ns*4]  # [72..95]
        # Right side: sectors 2,3 (30°-60°), Left side: sectors 21,22 (300°-330°)
        for s_i in [2, 3, 21, 22]:
            if enemy_approach[s_i] > 0.7 and obstacle[s_i] > 0.45:
                # Enemy converging from this side — turn away
                if s_i <= 12:  # threat from right → turn left
                    return 5, 'converge'   # medium left
                else:          # threat from left → turn right
                    return 6, 'converge'   # medium right

        # --- REFLEX 5: Body encirclement (off by default, enable via reflex5_enabled) ---
        if self.reflex5_enabled:
            front_arc = list(range(0, 7)) + list(range(18, 24))  # ±90° (13 sectors)
            body_close_count = 0
            for s_i in front_arc:
                if obstacle[s_i] > 0.5 and obs_type[s_i] >= 0:
                    body_close_count += 1

            if body_close_count >= 4:
                best_sector = min(front_arc, key=lambda s: obstacle[s])
                best_obs = obstacle[best_sector]

                if best_obs < 0.3:  # Gap found — steer toward it
                    if best_sector <= 6:
                        if best_sector <= 1: return 0, 'encircle'
                        elif best_sector <= 3: return 6, 'encircle'
                        else: return 8, 'encircle'
                    else:
                        if best_sector >= 22: return 0, 'encircle'
                        elif best_sector >= 20: return 5, 'encircle'
                        else: return 7, 'encircle'
                else:  # No gap — U-turn
                    left_total = sum(obstacle[18:24])
                    right_total = sum(obstacle[0:7])
                    return (9 if left_total <= right_total else 10), 'encircle'

        return None, None  # No reflex triggered — let the network decide

    # --- replay storage helpers -------------------------------------------

    @staticmethod
    def _obs_parts(obs):
        """obs -> (matrix (k*C,H,W) float32, sectors float32 or None)."""
        if isinstance(obs, dict):
            sec = obs.get('sectors')
            return obs['matrix'], (None if sec is None else np.asarray(sec, dtype=np.float32))
        return obs, None

    def _split_stack(self, matrix):
        """(k*C, H, W) stacked matrix -> list of k (C, H, W) views, oldest first."""
        c = matrix.shape[0] // self.frame_stack
        return [matrix[i * c:(i + 1) * c] for i in range(self.frame_stack)]

    @staticmethod
    def _quantize(frame):
        """[0, 1] float frame -> uint8. Clipped so an out-of-range value can
        never wrap around during the cast. The env already emits uint8
        frames; those pass through untouched."""
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)

    def _store_stack(self, matrix):
        """Store every frame of a stack. Identical consecutive frames (the
        reset-filled stack at episode start) are stored once."""
        seqs = []
        prev = None
        for fr in self._split_stack(matrix):
            if prev is not None and np.array_equal(fr, prev):
                seqs.append(seqs[-1])
            else:
                seqs.append(self.memory.push_frame(self._quantize(fr)))
            prev = fr
        return seqs

    def _extend_stack(self, seqs, next_matrix):
        """Stack after one env step: drop the oldest seq, store the newest frame."""
        newest = self._split_stack(next_matrix)[-1]
        return seqs[1:] + [self.memory.push_frame(self._quantize(newest))]

    def remember(self, state, action, reward, next_state, done, gamma=None):
        """
        Stores a single transition from full stacked observations.
        state/next_state: dict {'matrix': (12,H,W) float32, 'sectors': (99,) float32}
        or a bare matrix (legacy). Every frame of both stacks is stored, so
        prefer remember_nstep(), which stores one frame per env step.
        gamma: the gamma used to compute n-step return (stored for consistency across stage changes).
        """
        if gamma is None:
            gamma = self.current_gamma

        s_mat, s_sec = self._obs_parts(state)
        n_mat, n_sec = self._obs_parts(next_state)
        s_seq = self._store_stack(s_mat)
        n_seq = self._store_stack(n_mat)
        self.memory.push(s_seq, s_sec, action, reward, n_seq, n_sec, done, gamma)

    def set_gamma(self, gamma):
        """Set gamma for current curriculum stage."""
        self.current_gamma = gamma
        logger.info(f"  Gamma set to {gamma} (effective n-step gamma: {gamma**self.n_step:.3f})")

    def remember_nstep(self, state, action, reward, next_state, done, agent_id=0):
        """
        N-step return buffer. Accumulates transitions and pushes
        n-step returns to PER when buffer is full or episode ends.
        """
        if agent_id not in self.n_step_buffers:
            self.n_step_buffers[agent_id] = deque(maxlen=self.n_step)

        buf = self.n_step_buffers[agent_id]
        s_mat, s_sec = self._obs_parts(state)
        n_mat, n_sec = self._obs_parts(next_state)

        # The trainer passes the previous call's next_state object back as
        # this call's state while an episode runs. If that chain breaks (new
        # episode, respawned/re-added agent), the partial trajectory in the
        # n-step buffer cannot be extended: flush it as truncated (bootstrapped)
        # transitions and store the new stack in full.
        tracked = self._stack_tracker.get(agent_id)
        if tracked is not None and tracked[0] is state:
            s_seq = tracked[1]
        else:
            if buf:
                self._flush_nstep(agent_id)
            s_seq = self._store_stack(s_mat)
        n_seq = self._extend_stack(s_seq, n_mat)
        self._stack_tracker[agent_id] = (next_state, n_seq)

        buf.append((s_seq, s_sec, action, reward, n_seq, n_sec, done))

        if done:
            # Flush all remaining transitions in buffer
            self._flush_nstep(agent_id)
            self._stack_tracker.pop(agent_id, None)
        elif len(buf) == self.n_step:
            # Buffer full: compute n-step return for oldest transition
            self._push_nstep_transition(agent_id)

    def _push_nstep_transition(self, agent_id):
        """Compute n-step return for oldest transition and push to PER."""
        buf = self.n_step_buffers[agent_id]
        if not buf:
            return

        # Oldest transition provides (state, action)
        s_seq_0, s_sec_0, action_0, _, _, _, _ = buf[0]

        # Snapshot gamma at write time — stored in PER for consistency
        gamma_used = self.current_gamma

        # Compute n-step discounted return: R = r1 + gamma*r2 + gamma^2*r3
        R = 0.0
        last_n_seq = None
        last_n_sec = None
        last_done = False
        for i, (_, _, _, r, n_seq, n_sec, d) in enumerate(buf):
            R += (gamma_used ** i) * r
            last_n_seq = n_seq
            last_n_sec = n_sec
            last_done = d
            if d:
                break

        self.memory.push(s_seq_0, s_sec_0, action_0, R, last_n_seq, last_n_sec, last_done, gamma_used)

    def _flush_nstep(self, agent_id):
        """Flush all remaining transitions at episode end."""
        buf = self.n_step_buffers[agent_id]
        while buf:
            self._push_nstep_transition(agent_id)
            buf.popleft()

    def optimize_model(self):
        if len(self.memory) < self.config.opt.batch_size:
            return None

        # Sample: contiguous uint8 / float32 arrays gathered from the replay store
        batch, idxs, is_weights = self.memory.sample(self.config.opt.batch_size)
        dev = self.device

        action_batch = torch.from_numpy(batch['action']).to(dev).unsqueeze(1)
        reward_batch = torch.from_numpy(batch['reward']).to(dev)
        done_batch = torch.from_numpy(batch['done']).to(dev)
        weights_batch = torch.from_numpy(is_weights).to(dev)
        gamma_batch = torch.from_numpy(batch['gamma']).to(dev)

        # Reward scaling (scale=1.0 preserves signal; clamp wide enough for S5/S6 long episodes)
        # Q-values in S5+ can legitimately reach ~200 (survival escalation + food over 4000 steps)
        # so the old ±100 clamp was killing gradients in best episodes. grad_clip handles divergence.
        reward_scale = max(self.config.opt.reward_scale, 1.0)
        norm_rewards = torch.clamp(reward_batch / reward_scale, -500.0, 500.0)

        # Ship matrices as uint8 from the buffer's pinned staging tensors (4x
        # less host->device traffic, async copy) and convert on the device.
        # Converting to float32 on the CPU first was ~95% of this method's
        # wall time (157 MB per batch at 160x160x12).
        s_matrices = batch['s_mat'].to(dev, non_blocking=True).float().div_(255.0)
        n_matrices = batch['n_mat'].to(dev, non_blocking=True).float().div_(255.0)

        if self.use_hybrid:
            s_sectors = torch.from_numpy(batch['s_sec']).to(dev)
            n_sectors = torch.from_numpy(batch['n_sec']).to(dev)

            q_values = self.policy_net(s_matrices, s_sectors).gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.policy_net(n_matrices, n_sectors).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(n_matrices, n_sectors).gather(1, next_actions).squeeze(1)
                next_q_values = torch.clamp(next_q_values, -500.0, 500.0)
                # Use per-transition gamma from PER (consistent with n-step return computation)
                gamma_n = gamma_batch ** self.n_step
                expected_q_values = torch.clamp((next_q_values * gamma_n * (1 - done_batch)) + norm_rewards, -500.0, 500.0)
        else:
            # Legacy: matrix-only model
            q_values = self.policy_net(s_matrices).gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.policy_net(n_matrices).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(n_matrices).gather(1, next_actions).squeeze(1)
                next_q_values = torch.clamp(next_q_values, -500.0, 500.0)
                gamma_n = gamma_batch ** self.n_step
                expected_q_values = torch.clamp((next_q_values * gamma_n * (1 - done_batch)) + norm_rewards, -500.0, 500.0)

        # TD Error for PER
        td_errors_raw = (q_values.squeeze(1) - expected_q_values).detach()
        td_errors = td_errors_raw.abs().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Loss with IS weights (Huber loss — robust to Q-value outliers, prevents loss explosion)
        loss = (weights_batch * F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1), reduction='none').squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — clip_grad_norm_ returns the TOTAL norm BEFORE clipping
        grad_norm_pre = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.opt.grad_clip)

        self.optimizer.step()

        # Collect training metrics
        with torch.no_grad():
            q_vals_np = q_values.squeeze(1).detach().cpu().numpy()
            metrics = {
                'loss': loss.item(),
                'q_mean': float(np.mean(q_vals_np)),
                'q_max': float(np.max(q_vals_np)),
                'td_error_mean': float(np.mean(td_errors)),
                'grad_norm': float(grad_norm_pre) if isinstance(grad_norm_pre, (int, float)) else float(grad_norm_pre.item()) if hasattr(grad_norm_pre, 'item') else float(grad_norm_pre),
            }

        return metrics

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load_matching_state(self, model, state_dict):
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        return missing, unexpected, skipped

    def save_checkpoint(self, filepath, episode, max_steps=None, supervisor_state=None, run_uid=None, parent_uid=None, best_fitness=None, best_avg_reward=None):
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # Saving memory is heavy, maybe skip or save separately?
            # For now skip saving memory to save disk/time
            'max_steps': max_steps,  # Curriculum state
            'supervisor_state': supervisor_state,
            'run_uid': run_uid,
            'parent_uid': parent_uid,
            'best_fitness': best_fitness,
            'best_avg_reward': best_avg_reward,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            return 0, 200, None, None  # episode, max_steps (default), supervisor_state, run_uid

        checkpoint = torch.load(filepath, map_location=self.device)
        # Load only matching-shape tensors so older checkpoints survive action-space/head changes.
        missing_p, unexpected_p, skipped_p = self._load_matching_state(self.policy_net, checkpoint['policy_net_state'])
        missing_t, unexpected_t, skipped_t = self._load_matching_state(self.target_net, checkpoint['target_net_state'])
        if missing_p or skipped_p:
            logger.info(f"  Checkpoint: Policy net - new/random layers: {len(missing_p) + len(skipped_p)}")
        if unexpected_p:
            logger.info(f"  Checkpoint: Policy net - dropped layers: {len(unexpected_p)}")
        if skipped_p or skipped_t:
            logger.info(f"  Checkpoint: skipped mismatched tensors due to architecture/action changes")
        # Only load optimizer if architectures match (no missing keys)
        if not missing_p and not unexpected_p and not skipped_p:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            logger.info(f"  Checkpoint: Architecture changed - optimizer reset to fresh state")
        self.steps_done = checkpoint['steps_done']

        max_steps = checkpoint.get('max_steps', 200)  # Default to 200 for old checkpoints
        run_uid = checkpoint.get('run_uid', None)
        self.saved_best_fitness = checkpoint.get('best_fitness')
        self.saved_best_avg_reward = checkpoint.get('best_avg_reward')
        return checkpoint['episode'], max_steps, checkpoint.get('supervisor_state'), run_uid
