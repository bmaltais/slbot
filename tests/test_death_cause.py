import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from death_cause import Cause
from trainer import cause_cell


class TestCauseLabels(unittest.TestCase):
    """Every member's label/short/str, and CSV compatibility with the
    historical string literals it replaces (see death_cause.py)."""

    LABELS = {
        Cause.WALL: ("Wall", "Wall"),
        Cause.SNAKE_COLLISION: ("SnakeCollision", "Snake"),
        Cause.BROWSER_ERROR: ("BrowserError", "Browser"),
        Cause.INVALID_FRAME: ("InvalidFrame", "Invalid"),
        Cause.SPAWN_WAIT: ("SpawnWait", "SpawnWait"),
        Cause.MAX_STEPS: ("MaxSteps", "MaxSteps"),
    }

    def test_every_member_has_a_label_and_short_form(self):
        for cause, (label, short) in self.LABELS.items():
            self.assertEqual(cause.label, label)
            self.assertEqual(cause.short, short)

    def test_str_is_the_bare_label_not_the_enum_repr(self):
        for cause, (label, _short) in self.LABELS.items():
            self.assertEqual(str(cause), label)
            self.assertEqual(f"{cause}", label)

    def test_equals_and_hashes_like_the_historical_string_literal(self):
        # seed_history() loads old training_stats.csv rows as plain strings;
        # they must still compare and hash equal to the matching member.
        for cause, (label, _short) in self.LABELS.items():
            self.assertEqual(cause, label)
            self.assertEqual(label, cause)
            self.assertEqual(hash(cause), hash(label))
            self.assertEqual({label: 1}.get(cause), 1)
            self.assertEqual({cause: 1}.get(label), 1)

    def test_short_forms_are_distinct(self):
        shorts = [short for _label, short in self.LABELS.values()]
        self.assertEqual(len(shorts), len(set(shorts)))


class TestDashboardCauseStyling(unittest.TestCase):
    """The "Last Death" cell's colour and text, per cause — this is the
    exact spot the truncate-then-compare bug lived: agent_last_cause used to
    store cause_label[:10] ("SnakeColli") and compare it against "Snake",
    which can never match."""

    def test_wall_is_red(self):
        style, text = cause_cell(Cause.WALL)
        self.assertEqual(style, "red")
        self.assertEqual(text, "Wall")

    def test_snake_collision_is_yellow(self):
        style, text = cause_cell(Cause.SNAKE_COLLISION)
        self.assertEqual(style, "yellow")
        self.assertEqual(text, "Snake")

    def test_other_causes_are_dim(self):
        for cause in (Cause.BROWSER_ERROR, Cause.INVALID_FRAME, Cause.SPAWN_WAIT, Cause.MAX_STEPS):
            style, _text = cause_cell(cause)
            self.assertEqual(style, "dim", f"{cause} should render dim")

    def test_no_death_yet_sentinel_is_dim_and_passed_through(self):
        style, text = cause_cell('—')
        self.assertEqual(style, "dim")
        self.assertEqual(text, '—')

    def test_plain_string_cause_still_gets_its_short_form(self):
        # A raw "SnakeCollision" string (e.g. from a source that predates
        # Cause) must still render "Snake", not the full string.
        style, text = cause_cell("SnakeCollision")
        self.assertEqual(style, "yellow")
        self.assertEqual(text, "Snake")

        style, text = cause_cell("Wall")
        self.assertEqual(style, "red")
        self.assertEqual(text, "Wall")


if __name__ == '__main__':
    unittest.main()
