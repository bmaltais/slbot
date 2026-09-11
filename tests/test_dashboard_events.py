import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trainer
from trainer import CTRL_E, TrainingDashboard


class TestDashboardEvents(unittest.TestCase):
    def setUp(self):
        self.dash = TrainingDashboard()

    def test_store_keeps_more_than_the_visible_cap(self):
        self.assertGreater(self.dash.events.maxlen, self.dash.EVENTS_MAX_VISIBLE)
        self.assertEqual(self.dash.events.maxlen, self.dash.EVENTS_STORE_MAX)

    def test_wrap_rows_counts_full_lines(self):
        self.assertEqual(self.dash._wrap_rows("abc", 10), 1)
        self.assertEqual(self.dash._wrap_rows("abcdefghij", 5), 2)
        self.assertEqual(self.dash._wrap_rows("", 10), 1)

    def test_fit_events_keeps_the_newest(self):
        events = [f"e{i}" for i in range(10)]
        shown = self.dash._fit_events(events, max_rows=3, inner_width=40)
        self.assertEqual(shown, ["e7", "e8", "e9"])

    def test_fit_events_empty(self):
        self.assertEqual(self.dash._fit_events([], 5, 40), [])

    def test_panel_grows_with_event_count(self):
        empty, _a, empty_h, empty_bottom, _s = self.dash._events_bottom_geom(
            [], [], term_width=200, term_height=50,
        )
        five, _a, five_h, five_bottom, _s = self.dash._events_bottom_geom(
            [f"event {i}" for i in range(5)], [],
            term_width=200, term_height=50,
        )
        self.assertEqual(empty, [])
        self.assertEqual(len(five), 5)
        self.assertGreater(five_h, empty_h)
        self.assertGreater(five_bottom, empty_bottom)
        self.assertEqual(five_h, 5 + self.dash.EVENTS_PANEL_CHROME)

    def test_visible_cap_on_tall_terminal(self):
        events = [f"event {i}" for i in range(50)]
        shown, _a, events_h, bottom_h, stacked = self.dash._events_bottom_geom(
            events, [], term_width=200, term_height=80,
        )
        self.assertFalse(stacked)
        self.assertEqual(len(shown), self.dash.EVENTS_MAX_VISIBLE)
        self.assertEqual(shown[0], "event 30")
        self.assertEqual(shown[-1], "event 49")
        self.assertLessEqual(len(shown), bottom_h - self.dash.EVENTS_PANEL_CHROME)
        self.assertEqual(events_h, self.dash.EVENTS_MAX_VISIBLE + self.dash.EVENTS_PANEL_CHROME)

    def test_short_terminal_caps_below_max_visible(self):
        events = [f"event {i}" for i in range(50)]
        shown, _a, events_h, bottom_h, _s = self.dash._events_bottom_geom(
            events, [], term_width=200, term_height=36,
        )
        self.assertLess(len(shown), self.dash.EVENTS_MAX_VISIBLE)
        self.assertGreaterEqual(len(shown), 1)
        self.assertLessEqual(bottom_h, self.dash._max_bottom_height(36))
        self.assertLessEqual(events_h, bottom_h)

    def test_stacked_layout_leaves_room_for_agents(self):
        events = [f"event {i}" for i in range(20)]
        agents = [{"idx": i} for i in range(8)]
        shown, agents_h, events_h, bottom_h, stacked = self.dash._events_bottom_geom(
            events, agents, term_width=100, term_height=50,
        )
        self.assertTrue(stacked)
        self.assertEqual(bottom_h, agents_h + events_h)
        self.assertLessEqual(bottom_h, self.dash._max_bottom_height(50))
        self.assertLess(len(shown), self.dash.EVENTS_MAX_VISIBLE)

    def test_wide_bar_uses_agent_height_to_show_more_history(self):
        events = [f"event {i}" for i in range(12)]
        agents = [{"idx": i} for i in range(8)]
        shown, agents_h, events_h, bottom_h, stacked = self.dash._events_bottom_geom(
            events, agents, term_width=200, term_height=50,
        )
        self.assertFalse(stacked)
        self.assertEqual(bottom_h, max(agents_h, events_h))
        # Agents board is taller than 12 event lines, so fill the extra rows.
        self.assertEqual(len(shown), 12)

    def test_build_layout_uses_fitted_events(self):
        for i in range(30):
            self.dash.events.append(f"14:00:{i:02d} msg {i}")
        with patch.object(self.dash, "_term_width", return_value=200), \
             patch.object(self.dash, "_term_height", return_value=50):
            layout = self.dash._build_layout()
        events_renderable = layout["events"].renderable
        text = str(events_renderable.renderable)
        self.assertIn("msg 29", text)
        self.assertNotIn("msg 0", text)
        self.assertIsNotNone(layout["bottom_bar"].size)
        self.assertLessEqual(layout["bottom_bar"].size, self.dash._max_bottom_height(50))
        footer = str(layout["footer"].renderable.renderable)
        self.assertIn("Ctrl+E", footer)

    def test_ctrl_e_requests_graceful_shutdown(self):
        prev = trainer._shutdown_requested
        trainer._shutdown_requested = False
        try:
            self.dash._on_key(CTRL_E)
            self.assertTrue(trainer._shutdown_requested)
            self.assertTrue(any("Ctrl+E" in e for e in self.dash.events))
        finally:
            trainer._shutdown_requested = prev


if __name__ == "__main__":
    unittest.main()
