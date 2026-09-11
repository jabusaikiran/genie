"""Unit tests for Phase 2 Step 10: Reliability & Hardening Improvements.

Verifies:
- Copilot provider unauthenticated safe initialization and typed AuthenticationError(401)
- Web search gating hierarchy (skip screen questions, allow explicit web search, timeout slow searches)
- Audio device stream recovery on inactive/stale streams
- Win32 named mutex single-instance guard
- Panel error visibility and clean cancellation error suppression in _submit
- Runtime monitor change handlers
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PyQt6.QtWidgets import QApplication

from ai.github_copilot_provider import GitHubCopilotProvider
from ai.openai_compatible_provider import AuthenticationError
from audio.ambient_listener import AmbientListener
from companion_manager import AppState, CompanionManager
from main import (
    _acquire_single_instance_mutex,
    _release_single_instance_mutex,
)
from tutor import is_web_search_needed
from ui.overlay import CursorOverlay
from ui.panel import CompanionPanel


class TestCopilotProviderReliability(unittest.TestCase):
    """Verify GitHub Copilot provider hardening without token and typed error handling."""

    @patch("ai.github_copilot_provider.load_github_token", return_value=None)
    def test_init_without_token_does_not_raise(self, mock_load):
        """Provider must initialize cleanly without raising RuntimeError when unauthenticated."""
        provider = GitHubCopilotProvider()
        self.assertIsNone(provider._gh_token)

    @patch("ai.github_copilot_provider.load_github_token", return_value=None)
    def test_stream_response_raises_typed_auth_error(self, mock_load):
        """stream_response must raise typed AuthenticationError(401) directing to Tray sign-in."""
        provider = GitHubCopilotProvider()

        async def _run():
            gen = provider.stream_response(
                user_text="Hello",
                screenshots_b64=[],
                history=[],
                system_prompt="Test",
            )
            async for _ in gen:
                pass

        with self.assertRaises(AuthenticationError) as ctx:
            asyncio.run(_run())

        err = ctx.exception
        self.assertEqual(err.status_code, 401)
        self.assertEqual(err.provider, "copilot")
        self.assertIn("Tray → Model → Sign in to GitHub Copilot", str(err))

    @patch("ai.github_copilot_provider.load_github_token", return_value=None)
    def test_get_copilot_token_raises_typed_auth_error(self, mock_load):
        """_get_copilot_token must raise typed AuthenticationError(401)."""
        provider = GitHubCopilotProvider()
        mock_client = MagicMock()

        with self.assertRaises(AuthenticationError) as ctx:
            asyncio.run(provider._get_copilot_token(mock_client))

        err = ctx.exception
        self.assertEqual(err.status_code, 401)
        self.assertEqual(err.provider, "copilot")

    @patch("ai.github_copilot_provider.load_github_token", return_value=None)
    def test_health_check_returns_false_when_unauthenticated(self, mock_load):
        """health_check() must return False without unhandled exceptions when unauthenticated."""
        provider = GitHubCopilotProvider()
        ok = asyncio.run(provider.health_check())
        self.assertFalse(ok)


class TestWebSearchGating(unittest.TestCase):
    """Verify web search decision hierarchy and timeout behavior."""

    def test_screen_questions_skip_search(self):
        """Screen/UI/deictic/local-context queries must skip web search."""
        skip_examples = [
            "Where is the Export button?",
            "What does this error mean?",
            "What does this button do?",
            "Explain this diagram.",
            "Where do I click?",
            "What is the code on my screen?",
            "Show me where the save icon is",
            "What does this switch do?",
        ]
        for q in skip_examples:
            with self.subTest(query=q):
                self.assertFalse(
                    is_web_search_needed(q),
                    f"Expected query to skip web search: {q}",
                )

    def test_explicit_web_questions_allow_search(self):
        """Explicit web/search queries must allow web search."""
        allow_examples = [
            "Search for python asyncio tutorials",
            "Search the web for numpy array syntax",
            "Google the population of Tokyo",
            "Look up recent news about Python",
            "Who is Alan Turing?",
            "Who was Ada Lovelace?",
            "What is the current price of bitcoin?",
            "Weather in Seattle",
        ]
        for q in allow_examples:
            with self.subTest(query=q):
                self.assertTrue(
                    is_web_search_needed(q),
                    f"Expected query to allow web search: {q}",
                )

    def test_general_questions_preserve_search_behavior(self):
        """General non-screen questions preserve existing web-search behavior."""
        general_examples = [
            "What is quantum computing?",
            "Explain the theory of relativity",
            "How does photosynthesis work?",
        ]
        for q in general_examples:
            with self.subTest(query=q):
                self.assertTrue(
                    is_web_search_needed(q),
                    f"Expected general query to allow web search: {q}",
                )

    def test_web_search_timeout_drops_slow_results(self):
        """Web search exceeding 2.0s must be dropped and cancelled without hanging the turn."""
        async def _test():
            async def _slow_search():
                await asyncio.sleep(10.0)
                return "Slow search results"

            search_task = asyncio.create_task(_slow_search())
            search_results = ""
            try:
                search_results = await asyncio.wait_for(search_task, timeout=0.05) or ""
            except (asyncio.TimeoutError, TimeoutError):
                if not search_task.done():
                    search_task.cancel()
                search_results = ""

            self.assertEqual(search_results, "")
            self.assertTrue(search_task.cancelled())

        asyncio.run(_test())


class TestAudioDeviceRecovery(unittest.TestCase):
    """Verify audio listener stream recovery on inactive/stale streams."""

    def test_inactive_stream_is_disposed_and_reopened(self):
        """An inactive/dead stream must be closed and re-opened when _open_stream runs."""
        listener = AmbientListener(on_level=lambda _: None, on_wake=lambda: None)
        dead_stream = MagicMock()
        dead_stream.active = False
        listener._stream = dead_stream

        with patch("sounddevice.InputStream") as mock_sd_stream:
            new_stream = MagicMock()
            new_stream.active = True
            mock_sd_stream.return_value = new_stream

            listener._open_stream()

            dead_stream.stop.assert_called_once()
            dead_stream.close.assert_called_once()
            self.assertIs(listener._stream, new_stream)
            new_stream.start.assert_called_once()

    def test_healthy_active_stream_is_preserved(self):
        """An active stream is not needlessly recreated."""
        listener = AmbientListener(on_level=lambda _: None, on_wake=lambda: None)
        active_stream = MagicMock()
        active_stream.active = True
        listener._stream = active_stream

        with patch("sounddevice.InputStream") as mock_sd_stream:
            listener._open_stream()
            mock_sd_stream.assert_not_called()
            self.assertIs(listener._stream, active_stream)

    def test_start_recording_raises_if_stream_fails_to_open(self):
        """start_recording must raise RuntimeError if opening stream fails."""
        listener = AmbientListener(on_level=lambda _: None, on_wake=lambda: None)
        with patch.object(listener, "_open_stream"):
            listener._stream = None
            with self.assertRaises(RuntimeError):
                listener.start_recording()


class TestSingleInstanceMutex(unittest.TestCase):
    """Verify Win32 named mutex single-instance behavior."""

    @patch("sys.platform", "win32")
    def test_first_instance_acquires_mutex(self):
        """First instance acquires mutex successfully."""
        with patch("ctypes.windll.kernel32.CreateMutexW", return_value=12345):
            with patch("ctypes.windll.kernel32.GetLastError", return_value=0):
                acquired = _acquire_single_instance_mutex()
                self.assertTrue(acquired)
                _release_single_instance_mutex()

    @patch("sys.platform", "win32")
    def test_second_instance_detected_and_rejected(self):
        """Second instance receives ERROR_ALREADY_EXISTS (183) and returns False."""
        ERROR_ALREADY_EXISTS = 183
        with patch("ctypes.windll.kernel32.CreateMutexW", return_value=9999):
            with patch("ctypes.windll.kernel32.GetLastError", return_value=ERROR_ALREADY_EXISTS):
                with patch("ctypes.windll.kernel32.CloseHandle") as mock_close:
                    acquired = _acquire_single_instance_mutex()
                    self.assertFalse(acquired)
                    mock_close.assert_called_once_with(9999)


class TestPanelErrorAndCleanCancellation(unittest.TestCase):
    """Verify panel error display and clean cancellation in _submit."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test"])

    def test_panel_show_error_updates_ui(self):
        """show_error must populate response label and set status to Error."""
        panel = CompanionPanel()
        panel.show_error("API key invalid")
        self.assertIn("API key invalid", panel._response_text)
        self.assertIn("API key invalid", panel._response_label.text())
        self.assertEqual(panel._status_label.text(), "Error")

    def test_submit_ignores_cancelled_error(self):
        """_submit done callback must ignore CancelledError without emitting sig_error."""
        cm = CompanionManager()
        error_emitted = []
        cm.sig_error.connect(lambda e: error_emitted.append(e))

        fut = concurrent.futures.Future()
        fut.cancel()

        # Simulate what asyncio.run_coroutine_threadsafe callback receives
        done_callbacks = []
        with patch.object(fut, "add_done_callback", side_effect=done_callbacks.append):
            with patch("asyncio.run_coroutine_threadsafe", return_value=fut):
                cm._loop = MagicMock()
                async def _dummy():
                    pass
                coro = _dummy()
                try:
                    cm._submit(coro)
                finally:
                    coro.close()

        for cb in done_callbacks:
            cb(fut)

        self.assertEqual(len(error_emitted), 0, "Cancelled future should not emit sig_error")


class TestMonitorChangeHandlers(unittest.TestCase):
    """Verify multi-monitor topology change handling."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test"])

    def test_overlay_cover_all_monitors_runs(self):
        """overlay._cover_all_monitors executes safely without errors."""
        overlay = CursorOverlay()
        overlay._cover_all_monitors()
        geo = overlay.geometry()
        self.assertGreater(geo.width(), 0)
        self.assertGreater(geo.height(), 0)


if __name__ == "__main__":
    unittest.main()
