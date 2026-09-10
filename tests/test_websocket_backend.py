"""Tests for --backend websocket (issues #14 and #16).

#14: CDP steering, SlitherBrowser.close, one-shot CDP re-arm, auto-scale.
#16: no mixed-mode spawn, no missed game WS id, no selenium fallback.
"""
import math
import os
import sys
import threading
import time
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
        self.assertIsNone(env.browser)

    def test_env_close_tolerates_browser_close_raising(self):
        class Boom:
            def close(self):
                raise RuntimeError("quit hung")

        env = object.__new__(SlitherEnv)
        env._death_writer = None
        env.browser = Boom()
        env.close()
        self.assertIsNone(env.browser)

    def test_env_close_clears_browser_after_success(self):
        env = object.__new__(SlitherEnv)
        env._death_writer = None
        browser = MagicMock()
        env.browser = browser
        env.close()
        browser.close.assert_called_once()
        self.assertIsNone(env.browser)

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
        browser._cdp.state.dead = False
        browser.driver = MagicMock()
        browser.driver.execute_script.return_value = {"dead": True}
        browser._fast_getstate_injected = True
        browser._cdp_rearm_t0 = None
        browser._cdp_rearm_ms = None
        browser._cdp_fallback_ticks = 0
        return browser

    def test_get_game_data_does_not_try_activate_or_selenium(self):
        browser = self._bare_browser()
        data = browser.get_game_data()
        browser._cdp.try_activate.assert_not_called()
        browser.driver.execute_script.assert_not_called()
        self.assertTrue(data.get('spawning'))
        self.assertFalse(data.get('dead'))

    def test_get_game_data_dead_when_ws_closed(self):
        browser = self._bare_browser()
        browser._cdp.state.dead = True
        data = browser.get_game_data()
        self.assertTrue(data.get('dead'))
        self.assertFalse(data.get('spawning', False))
        browser.driver.execute_script.assert_not_called()

    def test_selenium_backend_still_uses_execute_script(self):
        browser = self._bare_browser()
        browser._use_cdp = False
        browser.get_game_data()
        browser.driver.execute_script.assert_called()

    def test_get_game_data_uses_cdp_when_active(self):
        browser = self._bare_browser()
        browser._cdp.active = True
        browser._cdp.get_game_data.return_value = {
            "dead": False, "self": {"x": 21600, "y": 21600},
        }
        data = browser.get_game_data()
        self.assertEqual(data["dead"], False)
        browser._cdp.try_activate.assert_not_called()
        browser.driver.execute_script.assert_not_called()
        browser._cdp.get_game_data.assert_called_once()

    def test_send_action_does_not_selenium_when_unarmed(self):
        browser = self._bare_browser()
        browser.send_action(0.5, 0)
        browser._cdp.send_action.assert_not_called()
        browser.driver.execute_script.assert_not_called()

    def test_send_action_uses_cdp_when_active(self):
        browser = self._bare_browser()
        browser._cdp.active = True
        browser.send_action(1.25, 1)
        browser._cdp.send_action.assert_called_once_with(1.25, 1)
        browser.driver.execute_script.assert_not_called()

    def test_rearm_does_not_reset_or_poll(self):
        browser = self._bare_browser()
        browser._cdp.try_activate.return_value = False
        with patch('browser_engine.time.sleep') as sleep:
            browser._rearm_cdp()
        browser._cdp.reset.assert_not_called()
        browser._cdp.try_activate.assert_called_once()
        sleep.assert_not_called()

    def test_force_restart_resets_cdp_before_connect_not_after(self):
        browser = self._bare_browser()
        browser.driver.execute_script.return_value = False  # not playing
        browser._handle_login = MagicMock()
        browser.inject_override_script = MagicMock()
        browser.inject_fast_getstate = MagicMock()
        order = []
        browser._cdp.reset.side_effect = lambda *a, **k: order.append('reset')
        browser._rearm_cdp = MagicMock(side_effect=lambda: order.append('rearm'))
        with patch('browser_engine.time.sleep'):
            browser.force_restart()
        self.assertEqual(order, ['reset', 'rearm'])
        browser._cdp.reset.assert_called_once_with(clear_ws_id=True)
        browser._handle_login.assert_called_once_with(wait_cdp=False)
        browser._cdp.try_activate.assert_not_called()

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
        self.assertTrue(ResourceMonitor.tick_is_active(
            [False, False], [live, None],
        ))
        self.assertFalse(ResourceMonitor.tick_is_active(
            [False, True], [live, None],
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

    def test_does_not_scale_up_on_zero_ms_spawning_ticks(self):
        monitor = ResourceMonitor(check_interval=0, cooldown_up=0, cooldown_down=0)
        monitor.last_check = 0
        metrics = {
            'cpu_percent': 4,
            'ram_free_mb': 65000,
            'ram_percent': 10,
            'avg_step_ms': 0,
        }
        self.assertEqual(monitor.recommend(1, metrics, backend="websocket"), 0)


class TestCdpInterceptorReset(unittest.TestCase):
    def test_reset_preserves_game_ws_id_by_default(self):
        cdp = CDPInterceptor(MagicMock())
        cdp._game_ws_request_id = 'ws-keep'
        cdp._frames_received = 50
        cdp.reset()
        self.assertEqual(cdp._game_ws_request_id, 'ws-keep')
        self.assertEqual(cdp._frames_received, 0)

    def test_reset_clear_ws_id_drops_stale_socket(self):
        cdp = CDPInterceptor(MagicMock())
        cdp._game_ws_request_id = 'ws-old'
        cdp.reset(clear_ws_id=True)
        self.assertIsNone(cdp._game_ws_request_id)

    def test_websocket_created_binds_new_request_id(self):
        cdp = CDPInterceptor(MagicMock())
        cdp._game_ws_request_id = 'ws-old'
        cdp._handle_cdp_event({
            'method': 'Network.webSocketCreated',
            'params': {'url': 'ws://1.2.3.4:444/slither', 'requestId': 'ws-new'},
        })
        self.assertEqual(cdp._game_ws_request_id, 'ws-new')
        self.assertEqual(cdp._frames_received, 0)

    def test_try_activate_uses_chrome_snake_even_if_packets_are_garbage(self):
        from ws_engine import Snake
        cdp = CDPInterceptor(MagicMock())
        cdp._running = True
        cdp.state.my_id = 7
        cdp.state.snakes[7] = Snake(id=7, x=-548101, y=716830)
        cdp.driver.execute_script.return_value = {
            'x': 21600, 'y': 21600, 'id': 42,
        }
        self.assertTrue(cdp.try_activate())
        self.assertEqual(cdp.state.my_id, 42)
        self.assertTrue(cdp.state.playing)
        self.assertTrue(cdp.active)

    def test_try_activate_false_when_chrome_has_no_snake(self):
        cdp = CDPInterceptor(MagicMock())
        cdp._running = True
        cdp.driver.execute_script.return_value = None
        self.assertFalse(cdp.try_activate())
        self.assertFalse(cdp.state.playing)

    def test_cdp_does_not_assume_first_snake_add(self):
        cdp = CDPInterceptor(MagicMock())
        self.assertFalse(cdp._packet_handler._assume_first_snake)

    def test_try_activate_does_not_wait_on_packet_frames(self):
        cdp = CDPInterceptor(MagicMock())
        cdp._running = True
        cdp._frames_received = 0
        cdp.driver.execute_script.return_value = {
            'x': 25000, 'y': 18000, 'id': 3,
        }
        self.assertTrue(cdp.try_activate())
        self.assertTrue(cdp.active)


class TestCdpSpawnGate(unittest.TestCase):
    def _alive(self):
        return {
            'dead': False,
            'valid': True,
            'self': {'x': 21600, 'y': 21600, 'len': 10, 'ang': 0.0},
            'enemies': [],
            'foods': [],
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'view_radius': 500,
            'gsc': 1.0,
        }

    def _env(self):
        with patch('slither_env._create_browser', return_value=MagicMock()):
            env = SlitherEnv(headless=True, nickname="TestBot", backend="websocket")
        env.browser = MagicMock()
        env.browser.force_restart = MagicMock()
        env.browser.try_activate_cdp = MagicMock(return_value=False)
        env.browser.cdp_is_active = MagicMock(return_value=False)
        env.browser.get_game_data = MagicMock(
            return_value={'dead': False, 'spawning': True},
        )
        env.browser.send_action = MagicMock()
        env.browser.cdp_stats = MagicMock(return_value={
            'cdp_active': False, 'rearm_ms': None, 'fallback_ticks': 0,
        })
        env.browser.update_view_plus_overlay = MagicMock()
        return env

    def test_reset_returns_spawning_until_cdp_ready(self):
        env = self._env()
        env._wait_for_playable_data = MagicMock(
            return_value={'dead': False, 'spawning': True},
        )
        obs = env.reset()
        self.assertTrue(obs.get('spawning'))
        self.assertEqual(env.browser.force_restart.call_count, 2)

    def test_reset_returns_playable_when_cdp_ready(self):
        env = self._env()
        env.browser.cdp_is_active.return_value = True
        env.browser.cdp_stats.return_value = {
            'cdp_active': True, 'rearm_ms': 120.0, 'fallback_ticks': 0,
        }
        env._wait_for_playable_data = MagicMock(return_value=self._alive())
        obs = env.reset()
        self.assertFalse(obs.get('spawning', False))
        self.assertIn('matrix', obs)

    def test_step_stays_spawning_while_cdp_unarmed(self):
        env = self._env()
        obs, reward, done, info = env.step(0)
        self.assertTrue(info.get('spawning'))
        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        env.browser.send_action.assert_not_called()
        self.assertTrue(obs.get('spawning'))

    def test_step_reconnects_on_unarmed_dead_or_timeout(self):
        env = self._env()
        env.browser.get_game_data.return_value = {'dead': True}
        obs, reward, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info.get('spawning'))
        self.assertEqual(info.get('cause'), 'SpawnWait')
        env.browser.send_action.assert_not_called()

        env2 = self._env()
        env2._cdp_spawn_wait_t0 = time.time() - 9
        obs, reward, done, info = env2.step(0)
        self.assertTrue(done)
        self.assertTrue(info.get('spawning'))

    def test_step_emits_spawned_when_cdp_just_armed(self):
        env = self._env()
        env._cdp_spawn_wait_t0 = time.time()
        env.browser.cdp_is_active.return_value = True
        env.browser.try_activate_cdp.return_value = True
        env.browser.cdp_stats.return_value = {
            'cdp_active': True, 'rearm_ms': 80.0, 'fallback_ticks': 0,
        }
        env.browser.get_game_data.return_value = self._alive()
        obs, reward, done, info = env.step(0)
        self.assertFalse(done)
        self.assertTrue(info.get('spawned'))
        self.assertFalse(info.get('spawning', False))
        env.browser.send_action.assert_not_called()
        self.assertIsNone(env._cdp_spawn_wait_t0)

    def test_step_plays_when_cdp_active(self):
        env = self._env()
        env.browser.cdp_is_active.return_value = True
        env.browser.try_activate_cdp.return_value = True
        env.browser.cdp_stats.return_value = {
            'cdp_active': True, 'rearm_ms': 80.0, 'fallback_ticks': 0,
        }
        alive = self._alive()
        env.browser.get_game_data.return_value = alive
        env.frame_skip = 0
        with patch('slither_env.time.sleep'):
            obs, reward, done, info = env.step(0)
        self.assertFalse(done)
        self.assertFalse(info.get('spawning', False))
        self.assertTrue(info.get('cdp_active'))
        env.browser.send_action.assert_called_once()


if __name__ == '__main__':
    unittest.main()
