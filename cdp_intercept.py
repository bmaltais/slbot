"""
CDP WebSocket Interceptor for slither.io.

Connects to Chrome DevTools Protocol via a direct WebSocket to localhost.
Intercepts game WebSocket frames, parses them in Python, and maintains
a GameState object. Actions are sent via CDP Runtime.evaluate.

Architecture:
    slither.io server ↔ Chrome (handles anti-bot) ↔ CDP WS ↔ Python

This gives us:
- get_game_data(): instant (reads from Python memory, no execute_script)
- send_action(): ~1-2ms via CDP Runtime.evaluate (vs ~10-15ms Selenium)
- Real-time state updates pushed via CDP events (no polling)
"""

import json
import time
import base64
import threading
import logging
import math
import urllib.request

import websocket as ws_lib

from ws_engine import GameState, SlitherWSClient
from ws_protocol import (
    PacketReader, TWO_PI,
    PACKET_INIT, PACKET_SNAKE_ADD, PACKET_PREINIT, PACKET_PONG,
    PACKET_MOVE_ABS, PACKET_MOVE_ABS2, PACKET_MOVE_GROW, PACKET_MOVE_GROW2,
    PACKET_ROTATE_E, PACKET_ROTATE_E2, PACKET_ROTATE_3, PACKET_ROTATE_4, PACKET_ROTATE_5,
    PACKET_FOOD_ADD, PACKET_FOOD_ADD_B, PACKET_FOOD_ADD_F,
    PACKET_FOOD_EAT, PACKET_FAM_UPDATE, PACKET_TAIL_REMOVE,
    PACKET_DEATH, PACKET_SECTOR_ON, PACKET_SECTOR_OFF,
    PACKET_MINIMAP, PACKET_LEADERBOARD, PACKET_SNAKE_REMOVE_DEAD,
    ROTATION_PACKETS, MOVEMENT_PACKETS, FOOD_ADD_PACKETS,
)

logger = logging.getLogger(__name__)


def log(msg):
    print(msg, flush=True)


def steering_js(angle, boost):
    """JS that sets heading the same way as selenium SlitherBrowser.send_action.

    xm/ym are offsets from screen center; the game does `wang = atan2(ym, xm)`.
    Adding canvas pixel center warps heading toward the bottom-right.
    """
    is_boost = 1 if boost > 0.5 else 0
    if is_boost:
        boost_js = (
            "window.accelerating=true;"
            "if(window.setAcceleration)window.setAcceleration(1);"
        )
    else:
        boost_js = (
            "window.accelerating=false;"
            "if(window.setAcceleration)window.setAcceleration(0);"
        )
    return (
        f"xm=Math.cos({angle})*500;ym=Math.sin({angle})*500;"
        f"window._botTargetAng={angle};{boost_js}"
    )


class CDPInterceptor:
    """
    Intercepts slither.io WebSocket traffic via Chrome DevTools Protocol.

    Usage:
        interceptor = CDPInterceptor(selenium_driver)
        interceptor.start()  # Starts listening for game WS frames

        data = interceptor.get_game_data()  # Instant, from memory
        interceptor.send_action(angle, boost)  # Via CDP Runtime.evaluate
    """

    MAX_FOODS = 300
    MAX_ENEMIES = 50
    MAX_BODY_PTS = 150

    def __init__(self, driver, nickname="CDPBot"):
        """
        Args:
            driver: Selenium WebDriver instance (Chrome, must have
                    --remote-debugging-port and --remote-allow-origins=*)
            nickname: Bot name; used to identify our snake from SNAKE_ADD.
        """
        self.driver = driver
        self.state = GameState()
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cdp_ws = None
        self._listener_thread = None
        self._running = False
        self._game_ws_request_id = None
        self._cdp_id_counter = 0
        self._want_etm_s = True  # slither.io uses timestamp mode
        self._packet_handler = SlitherWSClient.__new__(SlitherWSClient)
        # Initialize the handler's state to point to OUR state
        self._packet_handler.state = self.state
        self._packet_handler._lock = self._lock
        self._packet_handler._want_etm_s = True
        self._packet_handler._login_sent = False
        self._packet_handler._init_received = threading.Event()
        self._packet_handler._spawn_received = threading.Event()
        self._packet_handler._connected_event = threading.Event()
        self._packet_handler._close_requested = False
        self._packet_handler.nickname = nickname
        # Other snakes are added on the same WS; first SNAKE_ADD is not us.
        self._packet_handler._assume_first_snake = False
        # CDP is passive — Chrome handles WS sends, so _send_binary is a no-op
        self._packet_handler._send_binary = lambda data: None
        self._packet_handler.ws = None
        self._init_received = self._packet_handler._init_received
        self._frames_received = 0

    def _next_id(self):
        self._cdp_id_counter += 1
        return self._cdp_id_counter

    def start(self):
        """Connect to Chrome DevTools and start intercepting WS frames."""
        if self._running:
            return

        # Get Chrome debugger address
        debugger_addr = self.driver.capabilities.get(
            'goog:chromeOptions', {}
        ).get('debuggerAddress', '')

        if not debugger_addr:
            log("[CDP] ERROR: No debuggerAddress in driver capabilities. "
                "Add --remote-debugging-port=0 --remote-allow-origins=* to Chrome options.")
            return

        # Get page target's DevTools WS URL
        try:
            resp = urllib.request.urlopen(f'http://{debugger_addr}/json').read()
            targets = json.loads(resp)
            page_target = next(
                (t for t in targets if t.get('type') == 'page'), None
            )
            if not page_target:
                log("[CDP] ERROR: No page target found")
                return
            ws_url = page_target['webSocketDebuggerUrl']
        except Exception as e:
            log(f"[CDP] ERROR: Failed to get DevTools URL: {e}")
            return

        # Connect to Chrome DevTools via WebSocket
        try:
            self._cdp_ws = ws_lib.WebSocket()
            self._cdp_ws.connect(ws_url)
            log(f"[CDP] Connected to Chrome DevTools")
        except Exception as e:
            log(f"[CDP] ERROR: Failed to connect to DevTools WS: {e}")
            return

        # Enable Network domain to capture WS frames
        self._cdp_send('Network.enable', {})
        log("[CDP] Network monitoring enabled")

        # Start listener thread
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="cdp-listener",
        )
        self._listener_thread.start()
        log("[CDP] Listener thread started")

    def stop(self):
        """Stop the interceptor."""
        self._running = False
        if self._cdp_ws:
            try:
                self._cdp_ws.close()
            except Exception:
                pass
            self._cdp_ws = None
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)

    def _reset_parsed_state(self, *, clear_ws_id=False):
        """Rebuild GameState and frame counters. Caller may already hold `_lock`."""
        old_connected = self.state.connected
        self.state = GameState()
        self.state.connected = old_connected
        self._packet_handler.state = self.state
        if clear_ws_id:
            self._game_ws_request_id = None
        self._frames_received = 0
        self._init_received.clear()
        spawn_ev = getattr(self._packet_handler, '_spawn_received', None)
        if spawn_ev is not None:
            spawn_ev.clear()

    def reset(self, *, clear_ws_id=False):
        """Reset parsed game state for a new round.

        Does not clear `_game_ws_request_id` unless `clear_ws_id=True`.
        `connect()` often binds the new game WS before re-arm; wiping the
        id then drops incoming frames and CDP never activates.
        """
        with self._lock:
            self._reset_parsed_state(clear_ws_id=clear_ws_id)

    def _cdp_send(self, method, params=None):
        """Send a CDP command and return the response."""
        if not self._cdp_ws:
            return None
        msg_id = self._next_id()
        msg = {'id': msg_id, 'method': method, 'params': params or {}}
        try:
            with self._io_lock:
                self._cdp_ws.send(json.dumps(msg))
                self._cdp_ws.settimeout(5.0)
                while True:
                    resp = json.loads(self._cdp_ws.recv())
                    if resp.get('id') == msg_id:
                        return resp.get('result')
                    self._handle_cdp_event(resp)
        except Exception as e:
            logger.debug(f"[CDP] Send error: {e}")
            return None
        finally:
            try:
                if self._cdp_ws:
                    self._cdp_ws.settimeout(0.02)
            except Exception:
                pass

    def _cdp_send_fire_and_forget(self, method, params=None):
        """Send a CDP command without waiting for response."""
        if not self._cdp_ws:
            return
        msg_id = self._next_id()
        msg = {'id': msg_id, 'method': method, 'params': params or {}}
        try:
            with self._io_lock:
                self._cdp_ws.send(json.dumps(msg))
        except Exception:
            pass

    def _listen_loop(self):
        """Background thread: listen for CDP events."""
        self._cdp_ws.settimeout(0.02)
        while self._running:
            try:
                with self._io_lock:
                    if not self._cdp_ws:
                        break
                    raw = self._cdp_ws.recv()
                if raw:
                    msg = json.loads(raw)
                    self._handle_cdp_event(msg)
            except ws_lib.WebSocketTimeoutException:
                continue
            except ws_lib.WebSocketConnectionClosedException:
                log("[CDP] DevTools connection closed")
                self._running = False
                break
            except Exception as e:
                if self._running:
                    logger.debug(f"[CDP] Listener error: {e}")
                continue

    def _handle_cdp_event(self, msg):
        """Process a CDP event message."""
        method = msg.get('method', '')

        if method == 'Network.webSocketCreated':
            url = msg['params'].get('url', '')
            rid = msg['params'].get('requestId', '')
            # Detect the game WebSocket (port 444 or /slither path)
            if '/slither' in url or ':444' in url:
                # Bind the new id only. Do not wipe playing/my_id — Chrome may
                # already have spawned, and env.reset waits on that gate.
                with self._lock:
                    self._game_ws_request_id = rid
                    self._frames_received = 0
                log(f"[CDP] Game WebSocket detected: {url} (rid={rid})")

        elif method == 'Network.webSocketFrameReceived':
            rid = msg['params'].get('requestId', '')
            if rid == self._game_ws_request_id:
                response = msg['params'].get('response', {})
                opcode = response.get('opcode', 2)
                payload_data = response.get('payloadData', '')
                if payload_data:
                    try:
                        if opcode == 2:
                            # Binary frame — base64 encoded
                            raw = base64.b64decode(payload_data)
                        else:
                            # Text frame — raw UTF-8 bytes
                            raw = payload_data.encode('utf-8')
                        self._handle_game_frame(raw)
                        with self._lock:
                            self._frames_received += 1
                            nframes = self._frames_received
                        if nframes <= 3:
                            log(f"[CDP] Frame #{nframes}: {len(raw)}B opcode={opcode} "
                                f"first_bytes={list(raw[:8])}")
                    except Exception as e:
                        logger.debug(f"[CDP] Frame decode error: {e}")

        elif method == 'Network.webSocketClosed':
            rid = msg['params'].get('requestId', '')
            if rid == self._game_ws_request_id:
                log(f"[CDP] Game WebSocket closed (rid={rid}, frames={self._frames_received})")
                with self._lock:
                    self.state.dead = True
                    self.state.playing = False

    def _handle_game_frame(self, data: bytes):
        """Parse a raw game WebSocket frame using the existing protocol handler."""
        if len(data) < 1:
            return

        # Reuse SlitherWSClient's message parser (handles framing + dispatch)
        try:
            self._packet_handler._handle_message(data)
        except Exception as e:
            logger.debug(f"[CDP] Packet parse error: {e}")

        # Track that init has been received (snake identification happens on main thread)

    def try_activate(self):
        """Activate when Chrome actually has a live snake.

        Packet parse is not trustworthy (wrong snake / garbage coords), so do
        not wait on init/frames or position-match parsed snakes. One-shot JS.
        """
        if self.state.playing and not self.state.dead:
            return True
        try:
            result = self.driver.execute_script(
                "if(!window.slither || typeof window.slither.xx !== 'number') return null;"
                "return {x: window.slither.xx, y: window.slither.yy, id: window.slither.id};"
            )
            if not result or result.get('x') is None:
                return False
            sx, sy = float(result['x']), float(result['y'])
            if abs(sx) <= 1000 and abs(sy) <= 1000:
                return False
            if abs(sx) > 80000 or abs(sy) > 80000:
                return False
            sid = result.get('id')
            with self._lock:
                if sid is not None:
                    try:
                        self.state.my_id = int(sid)
                    except (TypeError, ValueError):
                        pass
                self.state.dead = False
                self.state.playing = True
                self.state.connected = True
            log(f"[CDP] Game state active — chrome snake id={sid} "
                f"pos=({sx:.0f},{sy:.0f}) frames={self._frames_received}")
            return True
        except Exception as e:
            logger.debug(f"[CDP] Could not identify snake: {e}")
        return False

    # ─── Public API (browser_engine compatible) ─────────────────────

    @property
    def active(self):
        """True when DevTools is up and Chrome's snake is live."""
        return bool(self._running and self.state.playing and not self.state.dead)

    def _js_game_data(self):
        """Read Chrome game state via CDP Runtime.evaluate (not Selenium)."""
        result = self._cdp_send('Runtime.evaluate', {
            'expression': 'window._botGetState ? window._botGetState() : null',
            'returnByValue': True,
        })
        if not result:
            return None
        inner = result.get('result', result) if isinstance(result, dict) else None
        if not isinstance(inner, dict):
            return None
        value = inner.get('value')
        return value if isinstance(value, dict) else None

    def get_game_data(self):
        """Prefer Chrome JS state via CDP. Packet parse is a last resort."""
        js = self._js_game_data()
        if js:
            return js
        with self._lock:
            return self._packet_handler._build_game_data()

    def send_action(self, angle, boost):
        """
        Send steering command via CDP Runtime.evaluate.
        Much faster than Selenium execute_script (~1-2ms vs ~10-15ms).
        Must set the same xm/ym as selenium (offsets from screen center).
        """
        self._cdp_send_fire_and_forget('Runtime.evaluate', {
            'expression': steering_js(angle, boost),
            'returnByValue': False,
        })

    def send_action_get_data(self, angle, boost):
        """Combined send + read. Action via CDP, state from memory."""
        self.send_action(angle, boost)
        return self.get_game_data()
