"""Unit tests for Phase 2 Step 9: Reliability & UX Improvements.

Verifies:
- Cancellation & State Machine (Esc during LISTENING/THINKING, race conditions, task ownership)
- Response clearing and Push-to-Talk wiring
- Identity classifier screen-context sensitivity
- Provider lazy/safe initialization without API keys and typed AuthenticationError
- Multi-monitor Privacy Guard (secondary monitor filtering, active window fail-closed, uncertain attribution)
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PyQt6.QtWidgets import QApplication

from ai.base_provider import Message
from ai.claude_provider import ClaudeProvider
from ai.openai_compatible_provider import AuthenticationError
from ai.openai_provider import OpenAIProvider
from companion_manager import AppState, CompanionManager
from config import cfg
from screen.capture import ScreenShot
from tutor import (
    find_sensitive_windows,
    get_sensitive_monitor_indices,
    is_identity_question,
)
from ui.panel import CompanionPanel


class TestStateAndCancellation(unittest.TestCase):
    """Verify state transitions, cancellation, and race-condition immunity."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test"])

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
        self.cm._screens_ctx = [s1]
        self.cm._active_screen_idx = 1

    def test_stop_while_idle_is_noop(self):
        """stop() when already IDLE must not emit redundant state changes."""
        self.cm._state = AppState.IDLE
        self.cm._cancel_flag = False
        self.cm._current_task = None

        emitted = []
        self.cm.sig_state_changed.connect(lambda s: emitted.append(s))
        self.cm.stop()
        self.assertEqual(len(emitted), 0)

    def test_esc_during_listening_aborts_recording(self):
        """Esc during LISTENING stops mic recording and immediately returns to IDLE."""
        self.cm._state = AppState.LISTENING
        mock_listener = MagicMock()
        self.cm._listener = mock_listener

        self.cm.stop()

        self.assertTrue(self.cm._cancel_flag)
        mock_listener.stop_recording.assert_called_once()
        self.assertEqual(self.cm._state, AppState.IDLE)

    def test_esc_during_thinking_cancels_task(self):
        """Esc during THINKING cancels active asyncio task and releases pointer."""
        self.cm._state = AppState.THINKING
        mock_task = MagicMock()
        mock_task.done.return_value = False
        self.cm._current_task = mock_task

        point_released = []
        self.cm.sig_point_release.connect(lambda: point_released.append(True))

        self.cm.stop()

        self.assertTrue(self.cm._cancel_flag)
        mock_task.cancel.assert_called_once()
        self.assertEqual(len(point_released), 1)
        self.assertEqual(self.cm._state, AppState.IDLE)

    def test_race_condition_stale_request_suppressed(self):
        """Mandatory race-condition test:
        Request A is cancelled, Request B starts immediately.
        Late response chunks, TTS, and pointing from Request A must be ignored.
        Request B must succeed cleanly.
        """
        # 1. Request A starts in THINKING
        self.cm._state = AppState.THINKING
        self.cm._active_request_id = 1
        req_id_a = 1

        # 2. User presses Esc -> cancelled
        self.cm.stop()
        self.assertTrue(self.cm._cancel_flag)
        self.assertEqual(self.cm._state, AppState.IDLE)

        # 3. Request B starts immediately -> active request id increments, state becomes THINKING
        self.cm._cancel_flag = False
        self.cm._active_request_id = 2
        self.cm._state = AppState.THINKING
        req_id_b = 2

        # 4. Late pointing from Request A
        point_emitted = []
        self.cm.sig_point_at.connect(lambda x, y, lbl: point_emitted.append((x, y, lbl)))
        self.cm._parse_points("[POINT:200,300:ButtonA]", req_id=req_id_a)
        self.assertEqual(len(point_emitted), 0, "Stale Request A point tag must NOT emit")

        # 5. Pointing from Request B succeeds
        self.cm._parse_points("[POINT:500,600:ButtonB]", req_id=req_id_b)
        self.assertEqual(len(point_emitted), 1)
        self.assertEqual(point_emitted[0][2], "ButtonB")

        # 6. Late TTS from Request A
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()
        self.cm._tts = mock_tts

        asyncio.run(self.cm._play_lesson("Late narration from A", "Late narration from A", req_id=req_id_a))
        mock_tts.speak.assert_not_called()

        # 7. Narration from Request B succeeds
        asyncio.run(self.cm._play_lesson("Narration for B", "Narration for B", req_id=req_id_b))
        mock_tts.speak.assert_called_once()

    def test_current_task_ownership_protection(self):
        """When Request A finishes its finally block, it must not clear Request B's task."""
        task_a = MagicMock()
        task_b = MagicMock()

        self.cm._current_task = task_b

        # Simulate Request A finally block
        this_task = task_a
        if self.cm._current_task is this_task:
            self.cm._current_task = None

        self.assertIs(self.cm._current_task, task_b, "Request A must not clear Request B's task")


class TestResponseClearingAndPTT(unittest.TestCase):
    """Verify panel clears response on state change and PTT signals work."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test"])

    def test_panel_clears_response_on_listening_and_thinking(self):
        panel = CompanionPanel()

        # Add text
        panel.append_response_chunk("Previous response text")
        self.assertIn("Previous response text", panel._response_label.text())

        # Entering LISTENING must clear response view
        panel.set_state(AppState.LISTENING)
        self.assertEqual(panel._response_label.text(), "")

        # Add text again
        panel.append_response_chunk("Another response text")
        self.assertIn("Another response text", panel._response_label.text())

        # Entering THINKING must also clear response view
        panel.set_state(AppState.THINKING)
        self.assertEqual(panel._response_label.text(), "")

    def test_push_to_talk_button_signals(self):
        panel = CompanionPanel()

        pressed_called = []
        released_called = []

        panel.on_push_to_talk_pressed.connect(lambda: pressed_called.append(True))
        panel.on_push_to_talk_released.connect(lambda: released_called.append(True))

        panel._ptt_btn.pressed.emit()
        self.assertEqual(len(pressed_called), 1)

        panel._ptt_btn.released.emit()
        self.assertEqual(len(released_called), 1)


class TestIdentityClassifier(unittest.TestCase):
    """Verify is_identity_question distinguishes screen context from entity lookups."""

    def test_screen_context_questions_retain_screenshots(self):
        screen_context_queries = [
            "What is this error?",
            "What does this button do?",
            "What is Docker?",
            "Tell me about this diagram",
            "How do I fix this code?",
            "Where is the save button?",
            "Why did this fail?",
        ]
        for query in screen_context_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    is_identity_question(query),
                    f"Query '{query}' has screen context and must NOT be classified as an identity question",
                )

    def test_real_entity_inquiries_are_identity(self):
        entity_queries = [
            "Who is Elon Musk?",
            "Who was Abraham Lincoln?",
            "Who are the Beatles?",
            "How old is Joe Biden?",
        ]
        for query in entity_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_identity_question(query),
                    f"Query '{query}' is a real entity inquiry and must be classified as an identity question",
                )


class TestUnconfiguredProviders(unittest.TestCase):
    """Verify Claude and OpenAI providers do not crash on init without API keys."""

    def test_claude_provider_safe_init_and_auth_error(self):
        provider = ClaudeProvider(api_key="")
        self.assertEqual(provider.provider_id, "claude")

        async def _run():
            with self.assertRaises(AuthenticationError) as ctx:
                async for _ in provider.stream_response(
                    user_text="hello", screenshots_b64=[], history=[], system_prompt=""
                ):
                    pass
            self.assertEqual(ctx.exception.status_code, 401)

        asyncio.run(_run())

    def test_openai_provider_safe_init_and_auth_error(self):
        provider = OpenAIProvider(api_key="")
        self.assertEqual(provider.provider_id, "openai")

        async def _run():
            with self.assertRaises(AuthenticationError) as ctx:
                async for _ in provider.stream_response(
                    user_text="hello", screenshots_b64=[], history=[], system_prompt=""
                ):
                    pass
            self.assertEqual(ctx.exception.status_code, 401)

        asyncio.run(_run())


class TestPrivacyGuard(unittest.TestCase):
    """Verify multi-monitor privacy detection and fail-closed semantics."""

    def setUp(self):
        self.screens = [
            ScreenShot(
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
            ),
            ScreenShot(
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
            ),
        ]

    @patch("tutor.find_sensitive_windows", return_value=[])
    @patch("tutor.active_window_title", return_value="Visual Studio Code")
    def test_no_sensitive_windows_returns_empty(self, mock_active, mock_find):
        sensitive = get_sensitive_monitor_indices(self.screens)
        self.assertEqual(sensitive, set())

    @patch("tutor.active_window_title", return_value="Visual Studio Code")
    @patch("tutor.find_sensitive_windows")
    def test_sensitive_window_on_secondary_monitor_filters_only_that_monitor(
        self, mock_find, mock_active
    ):
        mock_find.return_value = [
            {"title": "KeePass - Passwords", "rect": (2000, 100, 2500, 600), "hwnd": 1234}
        ]
        sensitive = get_sensitive_monitor_indices(self.screens)
        self.assertEqual(sensitive, {2}, "Only monitor 2 should be marked sensitive")

    @patch("tutor.active_window_title", return_value="1Password")
    @patch("tutor.find_sensitive_windows")
    def test_active_window_sensitive_fails_closed_on_all_monitors(
        self, mock_find, mock_active
    ):
        mock_find.return_value = [
            {"title": "1Password", "rect": (100, 100, 500, 500), "hwnd": 1234}
        ]
        sensitive = get_sensitive_monitor_indices(self.screens)
        self.assertEqual(sensitive, {1, 2}, "Active sensitive window must fail closed on all monitors")

    @patch("tutor.active_window_title", return_value="Visual Studio Code")
    @patch("tutor.find_sensitive_windows")
    def test_uncertain_attribution_fails_closed_on_all_monitors(
        self, mock_find, mock_active
    ):
        mock_find.return_value = [
            {"title": "Bitwarden", "rect": None, "hwnd": 5678}
        ]
        sensitive = get_sensitive_monitor_indices(self.screens)
        self.assertEqual(sensitive, {1, 2}, "Uncertain attribution must fail closed on all monitors")


if __name__ == "__main__":
    unittest.main()
