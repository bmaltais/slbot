"""The Backend interface: what SlitherEnv needs from a browser adapter.

Two real adapters satisfy this today, both instances of
``browser_engine.SlitherBrowser``: plain Selenium (``backend="selenium"``)
and Selenium-plus-CDP (``backend="websocket"``, ``use_cdp=True``). See
``slither_env._create_browser`` for how a backend string picks one.

``FakeBackend`` below is a third adapter, kept in memory, used by tests to
drive ``SlitherEnv`` through this interface instead of a real browser.
"""
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Everything a caller must know to drive a game through SlitherEnv.

    Every method here is called on ``self.browser`` somewhere in
    ``slither_env.SlitherEnv``. ``cdp_is_active``, ``try_activate_cdp`` and
    ``cdp_stats`` are only exercised when the env's ``backend`` is
    ``"websocket"``; ``inject_view_plus_overlay`` and
    ``update_view_plus_overlay`` only when the env was constructed with
    ``view_plus=True``. Every other method is called on every backend.
    """

    def get_game_data(self) -> Optional[Dict[str, Any]]:
        """Return one snapshot of game state, or None/falsy if unavailable."""
        ...

    def send_action(self, angle: float, boost: int) -> None:
        """Steer towards `angle` (radians), boosting if `boost` is truthy.

        Fire-and-forget: the env reads the result with a later
        get_game_data() call, not from this call's return value.
        """
        ...

    def force_restart(self) -> None:
        """Respawn / reload the page for a new episode."""
        ...

    def scan_game_variables(self) -> Optional[Dict[str, Any]]:
        """One-shot debug scan of in-page globals, called once per browser."""
        ...

    def inject_view_plus_overlay(self) -> None:
        """Inject the optional on-page debug overlay."""
        ...

    def update_view_plus_overlay(self, matrix=None, gsc=None, view_radius=None,
                                  debug_info=None) -> None:
        """Refresh the optional on-page debug overlay with the latest state."""
        ...

    def cdp_is_active(self) -> bool:
        """Whether the CDP packet interceptor is currently armed."""
        ...

    def try_activate_cdp(self) -> bool:
        """Attempt to arm the CDP packet interceptor; return the new state."""
        ...

    def cdp_stats(self) -> Dict[str, Any]:
        """Diagnostics about CDP arming, folded into env step/reset info."""
        ...

    def close(self) -> None:
        """Release browser resources. Must tolerate being called more than once."""
        ...


class FakeBackend:
    """Scripted, in-memory Backend adapter for tests.

    ``get_game_data()`` replays frames from a caller-supplied script, one per
    call. Once only one frame remains it is returned on every further call,
    so a test can set up either a genuine multi-frame sequence or a single
    constant frame. ``send_action`` calls are recorded in ``actions`` so a
    test can assert on what the env steered towards.
    """

    def __init__(self, frames=None, cdp_active=True):
        self._frames = list(frames) if frames else []
        # Controls cdp_is_active(), try_activate_cdp() and cdp_stats():
        # a single knob for "CDP is armed" in tests that need it.
        self.cdp_active = cdp_active
        self.actions = []          # [(angle, boost), ...]
        self.overlay_updates = []  # [{'gsc':..., 'view_radius':..., 'debug_info':...}, ...]
        self.restart_count = 0
        self.scan_result = {}
        self.closed = False

    # -- scripting, for tests -------------------------------------------
    def push_frame(self, frame):
        """Queue one more frame to be returned once the current script is exhausted."""
        self._frames.append(frame)

    def set_frame(self, frame):
        """Replace the script with a single frame, returned on every future call."""
        self._frames = [frame]

    # -- Backend interface ------------------------------------------------
    def get_game_data(self):
        if not self._frames:
            return None
        if len(self._frames) == 1:
            return self._frames[0]
        return self._frames.pop(0)

    def send_action(self, angle, boost):
        self.actions.append((angle, boost))

    def force_restart(self):
        self.restart_count += 1

    def scan_game_variables(self):
        return self.scan_result

    def inject_view_plus_overlay(self):
        pass

    def update_view_plus_overlay(self, matrix=None, gsc=None, view_radius=None,
                                  debug_info=None):
        self.overlay_updates.append(
            {'gsc': gsc, 'view_radius': view_radius, 'debug_info': debug_info}
        )

    def cdp_is_active(self):
        return self.cdp_active

    def try_activate_cdp(self):
        return self.cdp_active

    def cdp_stats(self):
        return {'cdp_active': self.cdp_active, 'rearm_ms': None, 'fallback_ticks': 0}

    def close(self):
        self.closed = True
