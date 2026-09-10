"""Tests for --backend websocket survival (issue #14).

CDP steering, SlitherBrowser.close, one-shot CDP re-arm, and auto-scale
must not die after the first death.
"""
import math
import os
import sys
import threading
import unittest
from multiprocessing import Pipe
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from browser_engine import SlitherBrowser
from cdp_intercept import CDPInterceptor, steering_js
from slither_env import SlitherEnv
from trainer import ResourceMonitor
from worker_process import READY_MSG, run_worker_loop
from worker_session import WorkerSession

from test_worker_session import FakeEnv


class TestCdpSteeringJs(unittest.TestCase):
    def test_matches_selenium_offsets_from_center(self):
        angle = math.pi / 4
        js = steering_js(angle, 0)
        self.assertIn(f"xm=Math.cos({angle})*500", js)
        self.assertIn(f"ym=Math.sin({angle})*500", js)
        self.assertNotIn("c.width/2", js)
        self.assertNotIn("c.height/2", js)
        self.assertNotIn("*300", js)
        self.assertNotIn("slither.wang", js)
        self.assertIn("window.accelerating=false", js)
        self.assertIn("setAcceleration(0)", js)

    def test_boost_sets_same_globals_as_selenium(self):
        js = steering_js(0.0, 1)
        self.assertIn("window.accelerating=true", js)
        self.assertIn("setAcceleration(1)", js)
        self.assertIn("xm=Math.cos(0.0)*500", js)

    def test_cdp_send_action_evaluates_steering_js(self):
        interceptor = object.__new__(CDPInterceptor)
        interceptor._cdp_send_fire_and_forget = MagicMock()
        interceptor.send_action(1.25, 0)
        interceptor._cdp_send_fire_and_forget.assert_called_once()
        args, kwargs = interceptor._cdp_send_fire_and_forget.call_args
        self.assertEqual(args[0], 'Runtime.evaluate')
        self.assertEqual(args[1]['expression'], steering_js(1.25, 0))


class TestSlitherBrowserClose(unittest.TestCase):
    def _bare_browser(self):
        browser = object.__new__(SlitherBrowser)
        browser._use_cdp = True
        browser._cdp = MagicMock()
        browser.driver = MagicMock()
        return browser

    def test_close_stops_cdp_and_quits_driver(self):
        browser = self._bare_browser()
        cdp = browser._cdp
        driver = browser.driver
        browser.close()
        cdp.stop.assert_called_once()
        driver.quit.assert_called_once()
        self.assertIsNone(browser._cdp)
        self.assertIsNone(browser.driver)

    def test_close_is_idempotent_and_swallows_failures(self):
        browser = self._bare_browser()
        browser._cdp.stop.side_effect = RuntimeError("cdp hung")
        browser.driver.quit.side_effect = RuntimeError("quit hung")
        browser.close()
        browser.close()  # second call after fields cleared

    def test_env_close_tolerates_missing_browser_close(self):
        env = object.__new__(SlitherEnv)
        env._death_writer = None
        env.browser = object()  # no close()
        env.close()

    def test_env_close_tolerates_browser_close_raising(self):
        class Boom:
            def close(self):
                raise RuntimeError("quit hung")

        env = object.__new__(SlitherEnv)
        env._death_writer = None
        env.browser = Boom()
        env.close()

    def test_worker_close_survives_env_close_error(self):
        class BoomEnv(FakeEnv):
            def close(self):
                self.reset_release.set()
                self.closed = True
                raise RuntimeError("quit hung")

        parent, child = Pipe()

        def boot():
            env = BoomEnv()
            env.reset_release.set()
            return WorkerSession(env, 8)

        t = threading.Thread(
            target=run_worker_loop, args=(child, 8, boot), daemon=True,
        )
        t.start()
        self.assertEqual(parent.recv(), READY_MSG)
        parent.send(('close', None))
        t.join(timeout=2)
        self.assertFalse(t.is_alive(), "worker must exit even if env.close raises")


class TestCdpRearm(unittest.TestCase):
    def _bare_browser(self):
        browser = object.__new__(SlitherBrowser)
        browser._use_cdp = True
        browser._cdp = MagicMock()
        browser._cdp.active = False
        browser._cdp._frames_received = 80
        browser._cdp.state = MagicMock()
        browser._cdp.state.playing = False
        browser.driver = MagicMock()
        browser.driver.execute_script.return_value = {"dead": True}
        browser._fast_getstate_injected = True
        return browser

    def test_get_game_data_does_not_try_activate(self):
        browser = self._bare_browser()
        browser.get_game_data()
        browser._cdp.try_activate.assert_not_called()
        browser.driver.execute_script.assert_called()

    def test_get_game_data_uses_cdp_when_active(self):
        browser = self._bare_browser()
        browser._cdp.active = True
        browser._cdp.get_game_data.return_value = {"dead": False, "self": {}}
        data = browser.get_game_data()
        self.assertEqual(data["dead"], False)
        browser._cdp.try_activate.assert_not_called()
        browser.driver.execute_script.assert_not_called()

    def test_rearm_resets_then_starts_cdp_once(self):
        browser = self._bare_browser()
        browser._start_cdp_if_enabled = MagicMock()
        browser._rearm_cdp()
        browser._cdp.reset.assert_called_once()
        browser._start_cdp_if_enabled.assert_called_once()

    def test_force_restart_rearms_cdp_once(self):
        browser = self._bare_browser()
        browser.driver.execute_script.return_value = False  # not playing
        browser._handle_login = MagicMock()
        browser.inject_override_script = MagicMock()
        browser.inject_fast_getstate = MagicMock()
        browser._rearm_cdp = MagicMock()
        with patch('browser_engine.time.sleep'):
            browser.force_restart()
        browser._handle_login.assert_called_once_with(wait_cdp=False)
        browser._rearm_cdp.assert_called_once()
        browser._cdp.try_activate.assert_not_called()


class TestWebsocketAutoscale(unittest.TestCase):
    def test_tick_is_active_requires_every_agent_playing(self):
        live = {'food_eaten': 0}
        self.assertTrue(ResourceMonitor.tick_is_active(
            [False, False], [live, live],
        ))
        self.assertFalse(ResourceMonitor.tick_is_active(
            [True, False], [live, live],
        ))
        self.assertFalse(ResourceMonitor.tick_is_active(
            [False, False], [live, {'spawning': True}],
        ))
        self.assertFalse(ResourceMonitor.tick_is_active(
            [False, False], [{'spawned': True}, live],
        ))

    def test_websocket_does_not_scale_down_on_212ms_death_tick(self):
        """Pre-fix: one 212ms death sample vs 100ms websocket down threshold."""
        monitor = ResourceMonitor(check_interval=0, cooldown_up=0, cooldown_down=0)
        monitor.last_check = 0
        metrics = {
            'cpu_percent': 40,
            'ram_free_mb': 8000,
            'ram_percent': 20,
            'avg_step_ms': 212,
        }
        self.assertEqual(monitor.recommend(2, metrics, backend="websocket"), 0)

    def test_websocket_still_scales_up_when_fast(self):
        monitor = ResourceMonitor(check_interval=0, cooldown_up=0, cooldown_down=0)
        monitor.last_check = 0
        metrics = {
            'cpu_percent': 40,
            'ram_free_mb': 8000,
            'ram_percent': 20,
            'avg_step_ms': 42,
        }
        self.assertEqual(monitor.recommend(1, metrics, backend="websocket"), 1)


if __name__ == '__main__':
    unittest.main()
