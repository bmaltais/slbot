"""Learning from recorded human play: action labels, the demo file format,
the env's passive mode, and the agent's demo buffer + margin loss."""
import math
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_space import (
    ACTION_ANGLE_CHANGE,
    ACTION_BOOST,
    ACTION_BOOST_LEFT,
    ACTION_BOOST_RIGHT,
    ACTION_DIM,
    ACTION_STRAIGHT,
    BOOST_TURN_LIMIT,
    decode_action,
    label_human_action,
)
from agent import DDQNAgent
from browser_backend import FakeBackend
from config import Config
from demo import DemoEpisode, iter_stacked_transitions, list_demos, load_demo, select_demos
from sector_layout import SECTOR_DIM
from slither_env import BOOST_SPEED_THRESHOLD, SlitherEnv

RES = 64
K = 4


# --- action_space -----------------------------------------------------------

def test_decode_round_trips_every_action():
    for a in range(ACTION_DIM):
        angle, boost = decode_action(a)
        assert angle == ACTION_ANGLE_CHANGE[a]
        assert boost == (1 if a >= ACTION_BOOST else 0)
    with pytest.raises(ValueError):
        decode_action(ACTION_DIM)


@pytest.mark.parametrize("delta,expected", [
    (0.0, 0), (0.05, 0), (-0.05, 0),
    (0.18, 2), (-0.18, 1), (0.30, 4), (-0.40, 3),
    (0.61, 6), (-0.7, 5), (0.96, 8), (-1.1, 7),
    (1.57, 10), (-1.57, 9), (3.0, 10), (-3.0, 9),
])
def test_label_snaps_to_nearest_turn(delta, expected):
    heading = 1.0
    assert label_human_action(heading, heading + delta, boosting=False) == expected


def test_label_wraps_across_pi():
    # heading just below +pi, wanted just above -pi: a small right turn
    assert label_human_action(math.pi - 0.05, -math.pi + 0.05, boosting=False) == 2


def test_label_boost_keeps_micro_turns_and_drops_boost_on_hard_turns():
    assert label_human_action(0.0, 0.0, boosting=True) == ACTION_BOOST
    assert label_human_action(0.0, -0.15, boosting=True) == ACTION_BOOST_LEFT
    assert label_human_action(0.0, 0.2, boosting=True) == ACTION_BOOST_RIGHT
    assert label_human_action(0.0, BOOST_TURN_LIMIT + 0.05, boosting=True) == 4
    assert label_human_action(0.0, -1.5, boosting=True) == 9


def test_label_without_wanted_heading_is_straight():
    assert label_human_action(0.3, None, boosting=False) == ACTION_STRAIGHT
    assert label_human_action(0.3, None, boosting=True) == ACTION_BOOST


# --- demo files -------------------------------------------------------------

def _obs(seed, size=RES):
    rng = np.random.default_rng(seed)
    return {
        'matrix': rng.integers(0, 256, (3, size, size), dtype=np.uint8),
        'sectors': rng.random(SECTOR_DIM, dtype=np.float32),
    }


def _write_demo(tmp_path, steps, peak_length, seed=0, done=True):
    ep = DemoEpisode(_obs(seed))
    for t in range(steps):
        ep.add(action=(t % ACTION_DIM), reward=0.5 * t, next_obs=_obs(seed + t + 1),
               done=(done and t == steps - 1), length=peak_length if t == steps // 2 else 5)
    return ep.save(str(tmp_path), stage=1, style="test")


def test_demo_save_load_round_trip(tmp_path):
    path = _write_demo(tmp_path, steps=7, peak_length=42)
    assert os.path.basename(path).startswith("demo_") and "len42" in path and "steps7" in path
    d = load_demo(path)
    assert d['frames'].shape == (8, 3, RES, RES) and d['frames'].dtype == np.uint8
    assert d['sectors'].shape == (8, SECTOR_DIM)
    assert list(d['actions']) == [t % ACTION_DIM for t in range(7)]
    assert d['rewards'][3] == pytest.approx(1.5)
    assert d['dones'].tolist() == [False] * 6 + [True]
    assert d['meta']['peak_length'] == 42
    assert d['meta']['steps'] == 7
    assert d['meta']['stage'] == 1
    assert d['meta']['matrix_size'] == RES


def test_empty_episode_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):
        DemoEpisode(_obs(0)).save(str(tmp_path))


def test_list_and_select_by_peak_length(tmp_path):
    a = _write_demo(tmp_path, steps=3, peak_length=10, seed=1)
    b = _write_demo(tmp_path, steps=3, peak_length=80, seed=2)
    assert set(list_demos(str(tmp_path))) == {a, b}
    kept, skipped = select_demos([a, b], min_score=50)
    assert kept == [b] and skipped == [a]


def test_stacked_transitions_mirror_vecframestack(tmp_path):
    path = _write_demo(tmp_path, steps=5, peak_length=1)
    d = load_demo(path)
    trans = list(iter_stacked_transitions(d, K))
    assert len(trans) == 5
    s0 = trans[0][0]['matrix']
    assert s0.shape == (3 * K, RES, RES)
    # reset fills the stack with frame 0
    for i in range(K):
        assert np.array_equal(s0[3 * i:3 * i + 3], d['frames'][0])
    # after step t the newest slot is frame t+1, and next_state chains
    for t, (s, a, r, ns, done) in enumerate(trans):
        assert np.array_equal(ns['matrix'][-3:], d['frames'][t + 1])
        assert np.array_equal(ns['sectors'], d['sectors'][t + 1])
        if t + 1 < len(trans):
            assert trans[t + 1][0] is ns
    assert trans[-1][4] is True and trans[0][4] is False


# --- env passive mode -------------------------------------------------------

def _alive(ang=0.0, wang=None, sp=5.8, length=10):
    me = {'x': 21600, 'y': 21600, 'len': length, 'ang': ang, 'sp': sp}
    if wang is not None:
        me['wang'] = wang
    return {
        'dead': False, 'valid': True, 'self': me, 'enemies': [], 'foods': [],
        'map_radius': 21600, 'map_center_x': 21600, 'map_center_y': 21600,
        'view_radius': 500, 'gsc': 1.0,
    }


@pytest.fixture
def env():
    e = SlitherEnv(headless=True, nickname="T", matrix_size=RES, browser=FakeBackend())
    e.frame_skip = 0
    # Only the straight penalty and boost cost stay on so the label is visible in the reward.
    for name in ('survival_reward', 'survival_escalation', 'food_reward', 'food_shaping',
                 'cluster_eat_reward', 'length_bonus', 'wall_proximity_penalty',
                 'enemy_proximity_penalty', 'enemy_approach_penalty', 'mass_loss_penalty',
                 'starvation_penalty', 'contest_food_reward', 'kill_opportunity_reward',
                 'enemy_zone_control_reward', 'idle_food_penalty', 'boost_cluster_reward'):
        setattr(e, name, 0.0)
    e.straight_penalty = 0.5
    e.boost_penalty = 0.25
    return e


def _passive_step(env, pre, post):
    env._cached_data = pre
    env.prev_length = pre['self']['len']
    env.browser.send_action = MagicMock()
    env.browser.get_game_data = MagicMock(return_value=post)
    return env.step(None)


def test_passive_step_never_steers_and_labels_the_turn(env):
    pre = _alive(ang=0.0, wang=0.6)  # mouse ~35 deg to the right
    _obs, reward, done, info = _passive_step(env, pre, _alive(ang=0.3))
    env.browser.send_action.assert_not_called()
    assert not done
    assert info['human_action'] == 6
    assert reward == pytest.approx(0.0)  # a turn: no straight penalty, no boost cost


def test_passive_step_straight_gets_straight_penalty(env):
    _obs, reward, done, info = _passive_step(env, _alive(ang=1.0, wang=1.02), _alive(ang=1.0))
    assert info['human_action'] == 0
    assert reward == pytest.approx(-0.5)


def test_passive_step_boost_from_post_window_speed(env):
    post = _alive(ang=0.0, sp=BOOST_SPEED_THRESHOLD + 1)
    _obs, reward, done, info = _passive_step(env, _alive(ang=0.0, wang=0.0), post)
    assert info['human_action'] == ACTION_BOOST
    assert reward == pytest.approx(-0.25)


def test_passive_step_labels_death_from_last_live_frame(env):
    pre = _alive(ang=0.0, wang=-1.5)
    _obs, reward, done, info = _passive_step(env, pre, {'dead': True})
    assert done
    assert info['human_action'] == 9
    assert reward < 0


def test_active_step_has_no_human_label(env):
    env._cached_data = _alive()
    env.browser.send_action = MagicMock()
    env.browser.get_game_data = MagicMock(return_value=_alive())
    _obs, _r, _d, info = env.step(3)
    env.browser.send_action.assert_called_once()
    assert 'human_action' not in info


def test_human_control_requires_selenium():
    with pytest.raises(ValueError):
        SlitherEnv(backend="websocket", browser=FakeBackend(), human_control=True)


# --- agent: demo buffer, mixed batches, margin loss, pretraining ------------

@pytest.fixture
def agent():
    cfg = Config()
    cfg.env.resolution = (RES, RES)
    cfg.env.frame_stack = K
    cfg.opt.batch_size = 8
    cfg.opt.lr = 1e-3
    cfg.buffer.capacity = 64
    cfg.demo.ratio = 0.25
    a = DDQNAgent(cfg)
    a.current_gamma = 0.5
    return a


def test_load_demos_counts_and_filters(agent, tmp_path):
    _write_demo(tmp_path, steps=6, peak_length=5, seed=10)
    _write_demo(tmp_path, steps=9, peak_length=60, seed=20)
    _write_demo(tmp_path, steps=4, peak_length=70, seed=30, done=False)  # truncated
    summary = agent.load_demos(list_demos(str(tmp_path)), min_score=50)
    assert summary['episodes'] == 2 and summary['skipped'] == 1
    assert summary['steps'] == 13
    # one n-step transition per demo step, truncated episode flushed too
    assert summary['transitions'] == 13 == len(agent.demo_memory)
    assert summary['peak_length'] == 70
    assert agent.demo_memory.priority_eps == agent.config.demo.priority_eps
    assert len(agent.memory) == 0  # live buffer untouched


def test_load_demos_with_nothing_kept_leaves_no_buffer(agent, tmp_path):
    _write_demo(tmp_path, steps=3, peak_length=1)
    summary = agent.load_demos(list_demos(str(tmp_path)), min_score=99)
    assert summary['transitions'] == 0 and agent.demo_memory is None


def test_optimize_without_demos_is_unchanged(agent):
    assert agent.optimize_model() is None
    assert agent._demo_batch_size(8) == 0


def test_demo_only_batches_until_live_buffer_fills(agent, tmp_path):
    _write_demo(tmp_path, steps=20, peak_length=1)
    agent.load_demos(list_demos(str(tmp_path)))
    # no live data: the whole batch is demos
    assert agent._demo_batch_size(8) == 8
    m = agent.optimize_model()
    assert m is not None and m['demo_frac'] == 1.0 and m['margin_loss'] >= 0.0
    # enough live data: 25% demos
    s = _stacked(0)
    for t in range(1, 10):
        ns = _stacked(t)
        agent.remember_nstep(s, t % ACTION_DIM, 0.1, ns, t == 9, agent_id=0)
        s = ns
    assert len(agent.memory) >= 6
    assert agent._demo_batch_size(8) == 2
    m = agent.optimize_model()
    assert m['demo_frac'] == pytest.approx(0.25)
    m = agent.optimize_model(demo_only=True)
    assert m['demo_frac'] == 1.0


def _stacked(seed):
    o = _obs(seed)
    return {'matrix': np.concatenate([o['matrix']] * K, axis=0), 'sectors': o['sectors']}


def test_margin_loss_pushes_demo_actions_to_the_top(agent, tmp_path):
    torch.manual_seed(0)
    # One repeated observation, always labelled with action 7.
    ep = DemoEpisode(_obs(5))
    for t in range(16):
        ep.add(7, 0.0, _obs(5), done=(t == 15), length=1)
    ep.save(str(tmp_path))
    agent.load_demos(list_demos(str(tmp_path)))
    state = _stacked(5)

    def greedy():
        return agent._greedy_actions([state])[0]

    result = agent.pretrain_from_demos(60, log_every=0)
    assert result['metrics'] is not None and result['steps_done'] == 60 and not result['interrupted']
    assert greedy() == 7
    # target net was synced at the end of pretraining
    for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p, t)


def test_pretrain_needs_demos(agent):
    with pytest.raises(RuntimeError):
        agent.pretrain_from_demos(1)


def test_pretrain_needs_a_size(agent, tmp_path):
    _write_demo(tmp_path, steps=4, peak_length=1)
    agent.load_demos(list_demos(str(tmp_path)))
    with pytest.raises(ValueError):
        agent.pretrain_from_demos()


# --- record_demo loop -------------------------------------------------------

class _StubEnv:
    """Duck-typed env for record_episode: scripted step results."""

    def __init__(self, results):
        self.results = list(results)
        self.resets = 0

    def reset(self):
        self.resets += 1
        return _obs(100)

    def step(self, action):
        assert action is None
        return self.results.pop(0)


def _step_result(action, reward=1.0, done=False, length=12, cause=None, seed=200):
    info = {'length': length}
    if action is not None:
        info['human_action'] = action
    if cause is not None:
        info['cause'] = cause
    return _obs(seed), reward, done, info


def test_record_episode_skips_unlabelled_frames_and_stops_on_death(tmp_path):
    from record_demo import record_episode
    env = _StubEnv([
        _step_result(2, seed=201),
        _step_result(None, reward=0.0, seed=202),          # invalid frame: no label
        _step_result(ACTION_BOOST, length=30, seed=203),
        _step_result(9, reward=-15.0, done=True, cause='wall', seed=204),
        _step_result(0, seed=205),                          # never reached
    ])
    ep, cause, truncated, interrupted = record_episode(env, max_steps=100)
    assert env.resets == 1
    assert len(ep) == 3 and ep.actions == [2, ACTION_BOOST, 9]
    assert cause == 'wall' and not truncated and not interrupted
    assert ep.peak_length == 30
    assert ep.total_reward == pytest.approx(1.0 + 1.0 - 15.0)
    path = ep.save(str(tmp_path), cause=cause)
    assert load_demo(path)['meta']['cause'] == 'wall'


def test_record_episode_step_cap_is_truncated():
    from record_demo import record_episode
    env = _StubEnv([_step_result(1, seed=s) for s in range(300, 310)])
    ep, cause, truncated, interrupted = record_episode(env, max_steps=4)
    assert len(ep) == 4 and truncated and not interrupted and cause is None


def test_record_episode_ctrl_c_returns_partial_episode():
    from record_demo import record_episode

    class _Interrupting(_StubEnv):
        def step(self, action):
            if not self.results:
                raise KeyboardInterrupt
            return super().step(action)

    env = _Interrupting([_step_result(3, seed=400), _step_result(4, seed=401)])
    ep, cause, truncated, interrupted = record_episode(env, max_steps=50)
    assert len(ep) == 2 and truncated and interrupted


# --- pretraining terminal view ----------------------------------------------

def _update(step=200, steps=1000, agree=0.5):
    return {
        'step': step, 'steps': steps, 'elapsed': 12.0, 'eta': 48.0, 'lr': 5e-5,
        'metrics': {'loss': 0.5 / step, 'margin_loss': 0.2 / step, 'td_error_mean': 0.3,
                    'q_mean': 1.5, 'q_max': 3.0, 'grad_norm': 0.4, 'demo_frac': 1.0},
        'agreement': {'agreement': agree, 'n': 64,
                      'human_counts': np.bincount([0] * 50 + [2] * 10 + [11] * 4, minlength=ACTION_DIM),
                      'policy_counts': np.bincount([0] * 60 + [2] * 4, minlength=ACTION_DIM)},
    }


def test_sparkline_and_helpers():
    from pretrain_view import action_mix, fmt_duration, progress_bar, sparkline
    assert sparkline([]) == ""
    assert sparkline([1, 1, 1]) == "▄▄▄"
    s = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
    assert s[0] == "▁" and s[-1] == "█" and len(s) == 8
    assert len(sparkline(range(100), width=10)) == 10
    assert fmt_duration(5) == "5s" and fmt_duration(65) == "1m05s" and fmt_duration(3700) == "1h01m"
    assert progress_bar(0.5, width=4) == "██░░"
    mix = action_mix(np.bincount([0] * 8 + [11] * 2, minlength=ACTION_DIM))
    assert mix.startswith("FWD 80%") and "BST 20%" in mix
    assert action_mix(np.zeros(ACTION_DIM)) == "-"


def test_progress_line_mentions_the_numbers_that_matter():
    from pretrain_view import progress_line
    line = progress_line(_update(step=250, steps=1000, agree=0.73))
    assert "250/1000 (25%)" in line and "agree 73%" in line
    assert "margin" in line and "left" in line


def test_monitor_plain_mode_prints_one_line_per_update():
    import io
    from pretrain_view import PretrainMonitor
    out = io.StringIO()
    with PretrainMonitor(use_rich=False, stream=out) as mon:
        mon.update(_update(step=100))
        mon.update(_update(step=200, agree=0.9))
    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 2 and lines[1].startswith("[Pretrain] 200/1000")
    assert mon.agree_hist == [0.5, 0.9] and len(mon.loss_hist) == 2


def test_monitor_rich_panel_renders_agreement_and_mixes():
    pytest.importorskip("rich")
    import io
    from rich.console import Console
    from pretrain_view import PretrainMonitor
    console = Console(file=io.StringIO(), width=120, force_terminal=False)
    with PretrainMonitor(use_rich=True, console=console) as mon:
        mon.update(_update(step=300, agree=0.66))
    console.print(mon.render())
    text = console.file.getvalue()
    assert "Demo pretraining" in text and "66%" in text
    assert "your actions" in text and "policy picks" in text
    assert "FWD 78%" in text and "300/1000" in text


def test_demo_table_lines_show_episode_outcomes(agent, tmp_path):
    from pretrain_view import demo_table_lines
    _write_demo(tmp_path, steps=5, peak_length=40, seed=50)
    _write_demo(tmp_path, steps=6, peak_length=3, seed=60, done=False)
    summary = agent.load_demos(list_demos(str(tmp_path)), min_score=10)
    lines = demo_table_lines(summary)
    assert len(lines) == 2
    assert "5 steps" in lines[0] and "peak   40" in lines[0]
    assert "skipped" in lines[1]


def test_demo_agreement_reaches_one_after_pretraining(agent, tmp_path):
    torch.manual_seed(1)
    ep = DemoEpisode(_obs(7))
    for t in range(12):
        ep.add(4, 0.0, _obs(7), done=(t == 11), length=1)
    ep.save(str(tmp_path))
    agent.load_demos(list_demos(str(tmp_path)))
    before = agent.demo_agreement(8)
    assert before['n'] == 8 and before['human_counts'][4] == 8
    assert before['policy_counts'].sum() == 8
    updates = []
    agent.pretrain_from_demos(60, log_every=20, on_progress=updates.append)
    assert [u['step'] for u in updates] == [20, 40, 60]
    assert set(updates[0]) >= {'step', 'steps', 'elapsed', 'eta', 'lr', 'metrics', 'agreement'}
    assert updates[-1]['agreement']['agreement'] == 1.0
    assert agent.demo_agreement(12)['agreement'] == 1.0


def test_trainer_does_not_shadow_resource_monitor():
    """The pretraining view must not be bound to `monitor`, which the main
    loop uses for the ResourceMonitor (regression: AttributeError record_step)."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), '..', 'trainer.py')).read()
    assert not re.search(r"PretrainMonitor\(\)\s+as\s+monitor\b", src)


# --- epoch mode and Ctrl+C ---------------------------------------------------

def test_iter_epoch_visits_every_transition_once(agent, tmp_path):
    _write_demo(tmp_path, steps=11, peak_length=1, seed=70)
    _write_demo(tmp_path, steps=6, peak_length=1, seed=80)
    agent.load_demos(list_demos(str(tmp_path)))
    buf = agent.demo_memory
    assert len(buf) == 17 and buf.epoch_batches(8) == 3
    seen = []
    sizes = []
    for batch, idxs, weights in buf.iter_epoch(8):
        sizes.append(len(batch['action']))
        assert len(idxs) == len(batch['action']) == len(weights)
        assert (weights == 1.0).all()
        seen.extend((idxs - (buf.capacity - 1)).tolist())
    assert sizes == [8, 8, 1]
    assert sorted(seen) == list(range(17))
    # tree indices are real: priority updates on them must not raise
    buf.update_priorities(np.arange(17) + (buf.capacity - 1), np.ones(17))


def test_epoch_mode_step_count_and_progress(agent, tmp_path):
    _write_demo(tmp_path, steps=20, peak_length=1, seed=90)
    agent.load_demos(list_demos(str(tmp_path)))
    updates = []
    result = agent.pretrain_from_demos(epochs=2, log_every=1, on_progress=updates.append, eval_states=4)
    per_epoch = agent.demo_memory.epoch_batches(agent.config.opt.batch_size)
    assert per_epoch == 3  # 20 transitions, batch 8
    assert result['steps'] == 6 == result['steps_done'] == len(updates)
    assert result['epochs_done'] == 2 and not result['interrupted']
    assert [u['epoch'] for u in updates] == [1, 1, 1, 2, 2, 2]
    assert updates[0]['epochs'] == 2
    assert all(u['metrics']['demo_frac'] == 1.0 for u in updates)


def test_epochs_win_over_steps(agent, tmp_path):
    _write_demo(tmp_path, steps=20, peak_length=1, seed=91)
    agent.load_demos(list_demos(str(tmp_path)))
    result = agent.pretrain_from_demos(steps=500, epochs=1, log_every=0)
    assert result['steps'] == 3


def test_ctrl_c_during_pretraining_keeps_progress_and_syncs_target(agent, tmp_path):
    _write_demo(tmp_path, steps=20, peak_length=1, seed=92)
    agent.load_demos(list_demos(str(tmp_path)))
    before = [p.detach().clone() for p in agent.policy_net.parameters()]

    def interrupt_at_third(update):
        if update['step'] == 3:
            raise KeyboardInterrupt

    result = agent.pretrain_from_demos(steps=50, log_every=1, on_progress=interrupt_at_third, eval_states=0)
    assert result['interrupted'] and result['steps_done'] == 3
    # weights moved, and were copied into the target net on the way out
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.policy_net.parameters()))
    for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p, t)


def test_progress_line_and_panel_show_epochs():
    from pretrain_view import PretrainMonitor, progress_line
    u = _update(step=6, steps=12)
    u.update(epoch=1, epochs=2)
    assert "epoch 1/2" in progress_line(u)
    pytest.importorskip("rich")
    import io
    from rich.console import Console
    console = Console(file=io.StringIO(), width=120, force_terminal=False)
    mon = PretrainMonitor(use_rich=True, console=console)
    mon.update(u)
    console.print(mon.render())
    assert "epoch 1/2" in console.file.getvalue()
