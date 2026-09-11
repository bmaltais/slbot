"""The best-fitness bar is only meaningful within the stage it was earned in
(fitness weighs avg_steps, and max_steps varies non-monotonically per
curriculum stage), so the checkpoint must carry the stage alongside the
fitness value and trainer.py must discard the bar on a stage mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import DDQNAgent
from config import Config

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


def test_checkpoint_round_trips_fitness_stage(tmp_path):
    path = str(tmp_path / "ckpt.pth")
    saver = _agent()
    saver.save_checkpoint(path, episode=42, best_fitness=1234.5, best_fitness_stage=3)

    loader = _agent()
    loader.load_checkpoint(path)
    assert loader.saved_best_fitness == 1234.5
    assert loader.saved_best_fitness_stage == 3


def test_old_checkpoint_without_stage_tag_loads_as_unknown(tmp_path):
    """A checkpoint saved before this fix has no best_fitness_stage key."""
    path = str(tmp_path / "old_ckpt.pth")
    saver = _agent()
    saver.save_checkpoint(path, episode=1, best_fitness=500.0)  # no best_fitness_stage kwarg

    loader = _agent()
    loader.load_checkpoint(path)
    assert loader.saved_best_fitness == 500.0
    assert loader.saved_best_fitness_stage is None
