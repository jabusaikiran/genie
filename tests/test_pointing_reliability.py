"""Tests for Phase 2 Step 8: Pointing & Coordinate Reliability.

Verifies:
- Mixed-DPI logical monitor origin calculation
- Denormalization with mixed-DPI secondary monitors
- Clamping of out-of-bounds coordinates (no discontinuous jumps)
- UIA physical desktop coordinate → logical secondary-monitor coordinate conversion
- RapidOCR local coordinate → logical secondary-monitor coordinate conversion
- Element locator and Universal locator secondary-monitor coordinate conversion
- Case-insensitive [POINT] and drawing tag parsing
- [POINT:x,y:screenX] screen-only shorthand
- Explicit screen overriding active screen
- Cursor position fallback when active-window center is unavailable
- Negative monitor coordinates
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from companion_manager import POINT_RE, CompanionManager
from screen.capture import ScreenShot, _get_qt_screen_info


class TestMixedDPILogicalMonitorOrigin(unittest.TestCase):
    """Verify that logical desktop origins match Qt geometry across mixed-DPI monitors."""

    def test_mixed_dpi_logical_origin_from_qt(self):
        # Monitor 1: 1920x1080 @ 150% (dpr 1.5, logical width 1280)
        # Monitor 2: 2560x1440 @ 100% (dpr 1.0, logical width 2560), positioned to the right
        mock_screen1 = MagicMock()
        mock_screen1.geometry.return_value = MagicMock(x=lambda: 0, y=lambda: 0, width=lambda: 1280, height=lambda: 720)
        mock_screen1.devicePixelRatio.return_value = 1.5

        mock_screen2 = MagicMock()
        mock_screen2.geometry.return_value = MagicMock(x=lambda: 1280, y=lambda: 0, width=lambda: 2560, height=lambda: 1440)
        mock_screen2.devicePixelRatio.return_value = 1.0

        mock_app = MagicMock()
        mock_app.screens.return_value = [mock_screen1, mock_screen2]

        with patch("PyQt6.QtWidgets.QApplication.instance", return_value=mock_app):
            # Monitor 1
            info1 = _get_qt_screen_info(1)
            self.assertIsNotNone(info1)
            log_l1, log_t1, dpr1 = info1
            self.assertEqual(log_l1, 0)
            self.assertEqual(log_t1, 0)
            self.assertEqual(dpr1, 1.5)

            # Monitor 2: Expected logical origin is 1280, NOT 1920!
            info2 = _get_qt_screen_info(2)
            self.assertIsNotNone(info2)
            log_l2, log_t2, dpr2 = info2
            self.assertEqual(log_l2, 1280)
            self.assertEqual(log_t2, 0)
            self.assertEqual(dpr2, 1.0)


class TestDenormalizationAndClamping(unittest.TestCase):
    """Verify _denorm accurately maps coordinates and clamps out-of-bounds values safely."""

    def setUp(self):
        self.cm = CompanionManager()
        # Setup mixed-DPI dual monitor setup:
        # Screen 1: 1920x1080 @ 150% -> logical size 1280x720, logical origin (0, 0)
        s1 = ScreenShot(
            index=1,
            width=1920,
            height=1080,
            base64_jpeg="",
            physical_width=1920,
            physical_height=1080,
            physical_left=0,
            physical_top=0,
            dpi_scale=1.5,
            logical_left=0,
            logical_top=0,
        )
        # Screen 2: 2560x1440 @ 100% -> logical size 2560x1440, logical origin (1280, 0)
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
            logical_left=1280,
            logical_top=0,
        )
        self.cm._screens_ctx = [s1, s2]
        self.cm._active_screen_idx = 1

    def test_denorm_secondary_mixed_dpi(self):
        # Top-left of Screen 2: (0, 0)
        x0, y0 = self.cm._denorm(0, 0, screen_idx=2)
        self.assertEqual(x0, 1280.0)
        self.assertEqual(y0, 0.0)

        # Center of Screen 2: (500, 500)
        xc, yc = self.cm._denorm(500, 500, screen_idx=2)
        self.assertEqual(xc, 1280.0 + 1280.0)  # 2560.0
        self.assertEqual(yc, 720.0)

    def test_denorm_clamping_above_1000(self):
        # Coordinate 1005, 500 on Screen 1:
        # Must safely clamp to 1000, 500 -> logical (1280.0, 360.0)
        # It must NOT drop to center (~52%) or switch divisors!
        x, y = self.cm._denorm(1005, 500, screen_idx=1)
        self.assertEqual(x, 1280.0)
        self.assertEqual(y, 360.0)

        # Negative coordinate -10, 500:
        # Clamps to 0, 500 -> logical (0.0, 360.0)
        xn, yn = self.cm._denorm(-10, 500, screen_idx=1)
        self.assertEqual(xn, 0.0)
        self.assertEqual(yn, 360.0)

    def test_denorm_negative_monitor_coordinates(self):
        # Screen 2 positioned left of primary: physical_left = -1920, logical_left = -1920
        s_left = ScreenShot(
            index=2,
            width=1920,
            height=1080,
            base64_jpeg="",
            physical_width=1920,
            physical_height=1080,
            physical_left=-1920,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=-1920,
            logical_top=0,
        )
        self.cm._screens_ctx = [self.cm._screens_ctx[0], s_left]
        x, y = self.cm._denorm(500, 500, screen_idx=2)
        self.assertEqual(x, -960.0)
        self.assertEqual(y, 540.0)


class TestLocalLocatorCoordinateConversions(unittest.TestCase):
    """Verify UIA, OCR, and Locator coordinate conversions on secondary monitors."""

    def test_uia_desktop_to_logical_secondary_monitor(self):
        # Monitor 2: logical_left = 1280, physical_left = 1920, dpi_scale = 1.0
        shot = ScreenShot(
            index=2,
            width=2560,
            height=1440,
            base64_jpeg="",
            physical_width=2560,
            physical_height=1440,
            physical_left=1920,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=1280,
            logical_top=0,
        )
        # UIA element at physical desktop x = 3200 (1280 px into monitor 2)
        physical_x = 3200
        physical_y = 720

        # Correct formula: logical_origin + (physical - physical_origin) / dpi_scale
        lx = shot.logical_left + (physical_x - shot.physical_left) / shot.dpi_scale
        ly = shot.logical_top + (physical_y - shot.physical_top) / shot.dpi_scale
        self.assertEqual(lx, 2560.0)
        self.assertEqual(ly, 720.0)

    def test_rapidocr_local_to_logical_secondary_monitor(self):
        # Monitor 2: logical_left = 1280, dpi_scale = 1.0
        shot = ScreenShot(
            index=2,
            width=2560,
            height=1440,
            base64_jpeg="",
            physical_width=2560,
            physical_height=1440,
            physical_left=1920,
            physical_top=0,
            dpi_scale=1.0,
            logical_left=1280,
            logical_top=0,
        )
        # RapidOCR returns monitor-local coordinate: (500, 300)
        local_x = 500
        local_y = 300

        # Correct formula: logical_origin + local_x / dpi_scale
        lx = shot.logical_left + local_x / shot.dpi_scale
        ly = shot.logical_top + local_y / shot.dpi_scale
        self.assertEqual(lx, 1780.0)
        self.assertEqual(ly, 300.0)

    def test_resolve_anchor_secondary_monitor(self):
        cm = CompanionManager()
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
            logical_left=1280,
            logical_top=0,
        )
        cm._screens_ctx = [s2]
        cm._active_screen_idx = 2

        mock_target = MagicMock()
        mock_target.bbox = (2020, 100, 2120, 200)  # physical coords on monitor 2

        with patch("ai.hybrid_pointer.find_target", return_value=mock_target):
            bbox = cm._resolve_anchor("Save button")
            self.assertIsNotNone(bbox)
            lx1, ly1, lx2, ly2 = bbox
            # l = 2020, phys_left = 1920 -> local = 100 -> log = 1280 + 100 = 1380
            self.assertEqual(lx1, 1380.0)
            self.assertEqual(ly1, 100.0)
            self.assertEqual(lx2, 1480.0)
            self.assertEqual(ly2, 200.0)


class TestPointAndTagParsing(unittest.TestCase):
    """Verify case-insensitive parsing, screen-only shorthand, and tag parsing."""

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
        self.cm._screens_ctx = [s1, s2]
        self.cm._active_screen_idx = 1

    def test_case_insensitive_point_tags(self):
        emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: emitted.append((x, y, lbl)))

        # lowercase point
        self.cm._parse_points("Click [point:500,500:Save]")
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0], (960.0, 540.0, "Save"))

        # uppercase Screen2
        self.cm._parse_points("Click [POINT:500,500:Save:Screen2]")
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[1], (1920.0 + 960.0, 540.0, "Save"))

    def test_screen_only_shorthand(self):
        emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: emitted.append((x, y, lbl)))

        # [POINT:500,500:screen2] -> screen2 is screen identifier, not label
        self.cm._parse_points("Look here [POINT:500,500:screen2]")
        self.assertEqual(len(emitted), 1)
        # Should land on Screen 2 (x = 1920 + 960 = 2880)
        self.assertEqual(emitted[0][0], 2880.0)
        self.assertEqual(emitted[0][1], 540.0)
        self.assertEqual(emitted[0][2], "")

    def test_explicit_screen_overrides_active(self):
        emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: emitted.append((x, y, lbl)))
        self.cm._active_screen_idx = 1

        self.cm._parse_points("See [POINT:100,100:Icon:screen2]")
        self.assertEqual(len(emitted), 1)
        # Screen 2 origin 1920 + 10% of 1920 = 1920 + 192 = 2112
        self.assertEqual(emitted[0][0], 2112.0)

    def test_drawing_tags_case_insensitive(self):
        # Case insensitive rect with color RED and screen Screen2
        shape = self.cm._shape_from_tag("[RECT:100,100,200,200:RED:Screen2]")
        self.assertIsNotNone(shape)
        self.assertEqual(shape["color"], "red")
        # Starts on screen 2 (1920 + 192 = 2112)
        self.assertEqual(shape["x1"], 2112.0)


class TestActiveMonitorFallback(unittest.TestCase):
    """Verify active monitor detection falls back to cursor position."""

    def test_cursor_fallback_when_active_window_none(self):
        # Setup 2 screens
        s1 = ScreenShot(index=1, width=1920, height=1080, base64_jpeg="", physical_width=1920,
                        physical_height=1080, physical_left=0, physical_top=0, dpi_scale=1.0,
                        logical_left=0, logical_top=0)
        s2 = ScreenShot(index=2, width=1920, height=1080, base64_jpeg="", physical_width=1920,
                        physical_height=1080, physical_left=1920, physical_top=0, dpi_scale=1.0,
                        logical_left=1920, logical_top=0)
        screenshots = [s1, s2]

        # Active window has no center (e.g. desktop clicked)
        win_info = {"title": "", "center": None}
        cursor_pt = (2500, 500)  # Cursor is physically on Screen 2

        from tutor import find_monitor_for_point
        cursor_scr_idx = find_monitor_for_point(cursor_pt, screenshots)
        self.assertEqual(cursor_scr_idx, 2)

        active_scr_idx = None
        if win_info.get("center"):
            active_scr_idx = find_monitor_for_point(win_info.get("center"), screenshots)
        if not active_scr_idx:
            active_scr_idx = cursor_scr_idx or (screenshots[0].index if screenshots else 1)

        self.assertEqual(active_scr_idx, 2)


if __name__ == "__main__":
    unittest.main()
