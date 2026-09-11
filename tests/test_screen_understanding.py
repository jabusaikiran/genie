"""Tests for Phase 2 Step 7: Screen Understanding & Visual Context Reliability.

Verifies:
- Conversational target extraction and scoring in ai.hybrid_pointer
- Deictic query detection and monitor lookup in tutor
- Active monitor and cursor context formatting in companion_manager
- Tolerant POINT_RE parsing and multi-monitor coordinate denormalization
- Multi-monitor drawing tag support
- Screen capture defaults and DPI query
"""

import unittest

from ai.hybrid_pointer import _extract_target, _score_match
from screen.capture import ScreenShot, capture_all_screens, query_monitor_dpi
from tutor import (
    find_monitor_for_point,
    is_deictic,
)
from companion_manager import POINT_RE, CompanionManager, _build_system_prompt


class TestHybridPointerConversationalMatching(unittest.TestCase):
    """Verify natural-language target extraction and scoring."""

    def test_extract_target_phrases(self):
        self.assertEqual(_extract_target("Where do I click to export this?"), "export")
        self.assertEqual(_extract_target("Show me the Save button"), "save")
        self.assertEqual(_extract_target("How do I find settings?"), "settings")
        self.assertEqual(_extract_target("click on options"), "options")
        self.assertEqual(_extract_target("press Submit"), "submit")

    def test_score_match_conversational(self):
        # Conversational query against UI element name
        score_export = _score_match("Where do I click to export this?", "Export")
        self.assertGreaterEqual(score_export, 0.85)

        score_save = _score_match("Show me the Save button", "Save")
        self.assertGreaterEqual(score_save, 0.85)

        score_settings = _score_match("How do I find settings?", "Settings")
        self.assertGreaterEqual(score_settings, 0.85)

    def test_score_match_exact_and_unrelated(self):
        # Exact match
        self.assertEqual(_score_match("Save", "Save"), 1.0)

        # Completely unrelated
        score_unrelated = _score_match("Tell me a bedtime story", "Export")
        self.assertEqual(score_unrelated, 0.0)


class TestTutorDeicticAndSpatial(unittest.TestCase):
    """Verify deictic detection and spatial monitor lookup in tutor."""

    def test_is_deictic(self):
        self.assertTrue(is_deictic("what is this button?"))
        self.assertTrue(is_deictic("what does this icon do?"))
        self.assertTrue(is_deictic("explain that error"))
        self.assertTrue(is_deictic("explain that icon"))
        self.assertTrue(is_deictic("what is here?"))
        self.assertTrue(is_deictic("what am i looking at"))
        self.assertFalse(is_deictic("write a python script to sort a list"))
        self.assertFalse(is_deictic("what is the weather in Tokyo?"))

    def test_find_monitor_for_point(self):
        s1 = ScreenShot(
            index=1,
            width=1920,
            height=1080,
            base64_jpeg="",
            physical_width=1920,
            physical_height=1080,
            physical_left=0,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=0,
            logical_top=0,
        )
        s2 = ScreenShot(
            index=2,
            width=1920,
            height=1080,
            base64_jpeg="",
            physical_width=1920,
            physical_height=1080,
            physical_left=1920,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=1920,
            logical_top=0,
        )
        screens = [s1, s2]

        # Point on monitor 1
        idx1 = find_monitor_for_point((500, 400), screens)
        self.assertEqual(idx1, 1)

        # Point on monitor 2
        idx2 = find_monitor_for_point((2500, 400), screens)
        self.assertEqual(idx2, 2)

        # Point out of bounds defaults to monitor 1
        idx_oob = find_monitor_for_point((5000, 5000), screens)
        self.assertEqual(idx_oob, 1)


class TestCompanionPromptContext(unittest.TestCase):
    """Verify system prompt construction includes active window and cursor context."""

    def test_single_monitor_no_screen_suffix_on_active_window(self):
        prompt = _build_system_prompt(
            window_title="Notepad",
            active_screen_idx=1,
            screen_count=1,
            cursor_info="CURSOR: Screen 1, x=500, y=500",
        )
        self.assertIn('ACTIVE WINDOW: "Notepad"', prompt)
        self.assertNotIn('(on Screen 1)', prompt)
        self.assertIn("CURSOR: Screen 1, x=500, y=500", prompt)

    def test_multi_monitor_screen_suffix_on_active_window(self):
        prompt = _build_system_prompt(
            window_title="Visual Studio Code",
            active_screen_idx=2,
            screen_count=2,
            cursor_info='CURSOR: Screen 2, x=200, y=300\nELEMENT UNDER CURSOR: "Run" (Button)',
        )
        self.assertIn('ACTIVE WINDOW: "Visual Studio Code" (on Screen 2)', prompt)
        self.assertIn('CURSOR: Screen 2, x=200, y=300', prompt)
        self.assertIn('ELEMENT UNDER CURSOR: "Run" (Button)', prompt)


class TestPointParsingAndDenormalization(unittest.TestCase):
    """Verify tolerant POINT_RE regex and coordinate mapping."""

    def setUp(self):
        self.cm = CompanionManager()
        s1 = ScreenShot(
            index=1,
            width=1920,
            height=1080,
            base64_jpeg="",
            physical_width=1920,
            physical_height=1080,
            physical_left=0,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=0,
            logical_top=0,
        )
        s2 = ScreenShot(
            index=2,
            width=2560,
            height=1440,
            base64_jpeg="",
            physical_width=2560,
            physical_height=1440,
            physical_left=1920,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=1920,
            logical_top=0,
        )
        self.cm._screens_ctx = [s1, s2]
        self.cm._active_screen_idx = 2

    def test_point_re_matches(self):
        m1 = POINT_RE.search("Look here [POINT:500,300:Save Button]")
        self.assertIsNotNone(m1)
        self.assertEqual(m1.group(1), "500")
        self.assertEqual(m1.group(2), "300")
        self.assertEqual(m1.group(3), "Save Button")
        self.assertIsNone(m1.group(4))

        m2 = POINT_RE.search("Click [POINT:250,750:Settings:screen1]")
        self.assertIsNotNone(m2)
        self.assertEqual(m2.group(1), "250")
        self.assertEqual(m2.group(2), "750")
        self.assertEqual(m2.group(3), "Settings")
        self.assertEqual(m2.group(4), "1")

    def test_parse_points_active_monitor_default(self):
        # When screen is omitted, targets active monitor (_active_screen_idx = 2)
        emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: emitted.append((x, y, lbl)))

        text = "Click here [POINT:500,500:Target]"
        self.cm._parse_points(text)
        self.assertEqual(len(emitted), 1)
        px, py, label = emitted[0]
        self.assertEqual(label, "Target")
        # Screen 2: left=1920, top=0, log_w=2560, log_h=1440
        # 500 norm -> 1920 + 0.5 * 2560 = 3200
        # 500 norm -> 0 + 0.5 * 1440 = 720
        self.assertEqual(px, 3200.0)
        self.assertEqual(py, 720.0)

    def test_parse_points_explicit_screen(self):
        # Explicit screen1 targeting
        emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: emitted.append((x, y, lbl)))

        text = "Click here [POINT:500,500:Target:screen1]"
        self.cm._parse_points(text)
        self.assertEqual(len(emitted), 1)
        px, py, label = emitted[0]
        self.assertEqual(label, "Target")
        # Screen 1: left=0, top=0, log_w=1920, log_h=1080
        # 500 norm -> 0 + 0.5 * 1920 = 960
        # 500 norm -> 0 + 0.5 * 1080 = 540
        self.assertEqual(px, 960.0)
        self.assertEqual(py, 540.0)

    def test_drawing_tags_explicit_and_active_screen(self):
        # Test shape_from_tag for RECT targeting screen1 explicitly
        shape_s1 = self.cm._shape_from_tag("[RECT:100,100,200,200:red:screen1]")
        self.assertIsNotNone(shape_s1)
        # Screen 1 rect: left=0, top=0, log_w=1920, log_h=1080
        # x1 = 0 + 100/1000 * 1920 = 192, y1 = 0 + 100/1000 * 1080 = 108
        # x2 = 0 + 200/1000 * 1920 = 384, y2 = 0 + 200/1000 * 1080 = 216
        self.assertEqual(shape_s1["x1"], 192.0)
        self.assertEqual(shape_s1["y1"], 108.0)
        self.assertEqual(shape_s1["x2"], 384.0)
        self.assertEqual(shape_s1["y2"], 216.0)

        # Test shape_from_tag for RECT without screen tag -> active screen (screen2)
        shape_s2 = self.cm._shape_from_tag("[RECT:100,100,200,200:blue]")
        self.assertIsNotNone(shape_s2)
        # Screen 2 rect: left=1920, top=0, log_w=2560, log_h=1440
        # x1 = 1920 + 100/1000 * 2560 = 2176, y1 = 0 + 100/1000 * 1440 = 144
        # x2 = 1920 + 200/1000 * 2560 = 2432, y2 = 0 + 200/1000 * 1440 = 288
        self.assertEqual(shape_s2["x1"], 2176.0)
        self.assertEqual(shape_s2["y1"], 144.0)
        self.assertEqual(shape_s2["x2"], 2432.0)
        self.assertEqual(shape_s2["y2"], 288.0)


class TestScreenCaptureDefaults(unittest.TestCase):
    """Verify capture defaults and DPI query helper."""

    def test_capture_all_screens_signature(self):
        import inspect
        sig = inspect.signature(capture_all_screens)
        self.assertEqual(sig.parameters["max_width"].default, 1920)
        self.assertEqual(sig.parameters["quality"].default, 85)

    def test_query_monitor_dpi(self):
        dpi = query_monitor_dpi(0, 0, 1920, 1080)
        self.assertIsInstance(dpi, float)
        self.assertGreaterEqual(dpi, 1.0)


if __name__ == "__main__":
    unittest.main()
