"""Cause: why an episode ended.

One enum is the single place a death/episode-end reason is spelled, so
curriculum checks, the super-pattern optimizer, death stats, the CSV writer
and the dashboard all compare against the same values instead of each
re-typing a matching string literal.

Cause mixes in `str` so the historical CSV/log text is unchanged: every
member's value is exactly the string literal it replaces (`str(Cause.WALL)
== "Wall"`), and old rows already on disk (loaded as plain strings by
seed_history) still compare equal and hash equal to the matching member.
"""
from enum import Enum


class Cause(str, Enum):
    WALL = "Wall"
    SNAKE_COLLISION = "SnakeCollision"
    BROWSER_ERROR = "BrowserError"
    INVALID_FRAME = "InvalidFrame"
    SPAWN_WAIT = "SpawnWait"
    MAX_STEPS = "MaxSteps"

    def __str__(self):
        # Without this, str,Enum.__str__ prints "Cause.WALL" on some Python
        # versions — every f-string and CSV write needs the bare value.
        return self.value

    @property
    def label(self):
        """Full display label — identical to the CSV/log value."""
        return self.value

    @property
    def short(self):
        """Label that fits an 11-column-wide dashboard cell."""
        return _SHORT_LABELS[self]


_SHORT_LABELS = {
    Cause.WALL: "Wall",
    Cause.SNAKE_COLLISION: "Snake",
    Cause.BROWSER_ERROR: "Browser",
    Cause.INVALID_FRAME: "Invalid",
    Cause.SPAWN_WAIT: "SpawnWait",
    Cause.MAX_STEPS: "MaxSteps",
}
