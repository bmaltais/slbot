"""The best-fitness and best-avg-reward bars are only meaningful within the
stage they were earned in (fitness weighs avg_steps, which is capped
non-monotonically per curriculum stage; reward shaping and gamma also vary
per stage), so checkpoints must carry the stage alongside each value and
trainer.py must discard a bar on a stage mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import DDQNAgent
from config import Config
from death_cause import Cause
from trainer import CurriculumManager, resolve_best_bar

RES = 64  # smallest size the hybrid net's four conv layers accept


def _agent():
    cfg = Config()
    cfg.env.resolution = (RES, RES)
    cfg.env.frame_stack = 4
    return DDQNAgent(cfg)


def test_new_agent_has_no_saved_fitness_stage():
    agent = _agent()
    assert agent.saved_best_fitness is None
    assert agent.saved_best_fitness_stage is None
    assert agent.saved_best_avg_reward is None
    assert agent.saved_best_avg_reward_stage is None


def test_checkpoint_round_trips_fitness_stage(tmp_path):
    path = str(tmp_path / "ckpt.pth")
    saver = _agent()
    saver.save_checkpoint(
        path, episode=42,
        best_fitness=1234.5, best_fitness_stage=3,
        best_avg_reward=7.5, best_avg_reward_stage=3,
    )

    loader = _agent()
    loader.load_checkpoint(path)
    assert loader.saved_best_fitness == 1234.5
    assert loader.saved_best_fitness_stage == 3
    assert loader.saved_best_avg_reward == 7.5
    assert loader.saved_best_avg_reward_stage == 3


def test_old_checkpoint_without_stage_tag_loads_as_unknown(tmp_path):
    """A checkpoint saved before this fix has no *_stage keys."""
    path = str(tmp_path / "old_ckpt.pth")
    saver = _agent()
    saver.save_checkpoint(path, episode=1, best_fitness=500.0, best_avg_reward=5.0)  # no *_stage kwargs

    loader = _agent()
    loader.load_checkpoint(path)
    assert loader.saved_best_fitness == 500.0
    assert loader.saved_best_fitness_stage is None
    assert loader.saved_best_avg_reward == 5.0
    assert loader.saved_best_avg_reward_stage is None


# --- supervisor_state must survive torch.load's weights_only=True default ---
# (PyTorch >= 2.6): a Cause instance pickled into the checkpoint (e.g. via
# CurriculumManager.get_state()'s cause_history) would otherwise make
# load_checkpoint() raise UnpicklingError on any checkpoint saved mid-episode.

def test_checkpoint_with_curriculum_state_loads_with_weights_only_default(tmp_path):
    path = str(tmp_path / "ckpt.pth")
    curriculum = CurriculumManager()
    curriculum.record_episode(food_eaten=3, steps=120, cause=Cause.WALL, peak_length=20)
    curriculum.record_episode(food_eaten=5, steps=200, cause=Cause.SNAKE_COLLISION, peak_length=30)

    saver = _agent()
    saver.save_checkpoint(path, episode=7, supervisor_state=curriculum.get_state())

    loader = _agent()
    # No map_location override here: this exercises agent.load_checkpoint's
    # real torch.load call, weights_only default included.
    episode, _max_steps, supervisor_state, _run_uid = loader.load_checkpoint(path)
    assert episode == 7
    assert supervisor_state["cause_history"] == ["Wall", "SnakeCollision"]
    # Every entry is a plain str, not a pickled Cause instance.
    assert all(type(c) is str for c in supervisor_state["cause_history"])


def test_old_checkpoint_with_pickled_cause_still_loads(tmp_path):
    """A checkpoint saved before get_state() plain-stringified cause_history
    has a raw Cause instance pickled in. agent.py registers Cause as a torch
    safe global specifically so these keep loading under weights_only=True
    instead of raising UnpicklingError, the way a real checkpoint saved on
    this branch before that fix did."""
    path = str(tmp_path / "old_format_ckpt.pth")
    saver = _agent()
    saver.save_checkpoint(
        path, episode=9,
        supervisor_state={"stage": 2, "cause_history": [Cause.WALL, Cause.MAX_STEPS]},
    )

    loader = _agent()
    episode, _max_steps, supervisor_state, _run_uid = loader.load_checkpoint(path)
    assert episode == 9
    assert supervisor_state["cause_history"] == [Cause.WALL, Cause.MAX_STEPS]


# --- resolve_best_bar: the stage-mismatch decision trainer.py restores with ---

def test_resolve_best_bar_restores_on_matching_stage():
    assert resolve_best_bar(1234.5, 3, current_stage=3, label="fitness") == 1234.5


def test_resolve_best_bar_discards_on_stage_mismatch():
    assert resolve_best_bar(1234.5, 5, current_stage=1, label="fitness") == -float('inf')


def test_resolve_best_bar_restores_untagged_checkpoint_best_effort():
    """No stage tag (pre-fix checkpoint) means we can't verify it, so trust it."""
    assert resolve_best_bar(1234.5, None, current_stage=1, label="fitness") == 1234.5


def test_resolve_best_bar_with_no_saved_value_starts_fresh():
    assert resolve_best_bar(None, None, current_stage=1, label="fitness") == -float('inf')
