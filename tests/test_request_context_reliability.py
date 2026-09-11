"""
Unit tests for AI Request & Context Reliability (Phase 2, Step 6).

Verifies:
- Multi-monitor screenshot labeling across OpenAI-compatible, Gemini, and Claude
- Defensive vision gating in Claude, Ollama, and Copilot
- Error translation for Claude and Ollama
- Copilot payload reuse with build_openai_messages
- History sanitization (removing [POINT:...], [RECT:...], etc.)
- Context-window budgeting for small and large models
- Async screen capture offloading via asyncio.to_thread
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from ai.base_provider import Message, ModelInfo
from ai.claude_provider import ClaudeProvider, translate_anthropic_error
from ai.gemini_provider import build_gemini_payload
from ai.github_copilot_provider import GitHubCopilotProvider
from ai.ollama_provider import OllamaProvider, format_ollama_error
from ai.openai_compatible_provider import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NotFoundError,
    OpenAIProviderError,
    RateLimitError,
    ServerError,
    build_openai_messages,
)
from companion_manager import (
    CompanionManager,
    _budget_context,
    _sanitize_history_text,
)


class _MockStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class _MockAsyncClient:
    def __init__(self, response):
        self.response = response
        self.last_method = None
        self.last_url = None
        self.last_json = None

    def stream(self, method, url, **kwargs):
        self.last_method = method
        self.last_url = url
        self.last_json = kwargs.get("json")
        return _MockStreamContext(self.response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class _MockStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class TestRequestContextReliability(unittest.TestCase):

    # ── 1. Multi-Monitor Screenshot Labeling ──────────────────────────────────

    def test_openai_compatible_multi_monitor_labels(self):
        """Verify multiple screenshots in build_openai_messages get Screen 1 (Primary):, Screen 2: labels."""
        # 2 screenshots -> labeled
        msgs = build_openai_messages(
            system_prompt="sys",
            history=[],
            user_text="what is on my screens?",
            screenshots_b64=["base64_shot1", "base64_shot2"],
            supports_vision=True,
        )
        user_msg = msgs[-1]
        parts = user_msg["content"]

        self.assertEqual(parts[0], {"type": "text", "text": "Screen 1 (Primary):"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,base64_shot1")

        self.assertEqual(parts[2], {"type": "text", "text": "Screen 2:"})
        self.assertEqual(parts[3]["type"], "image_url")
        self.assertEqual(parts[3]["image_url"]["url"], "data:image/jpeg;base64,base64_shot2")

        self.assertEqual(parts[4], {"type": "text", "text": "what is on my screens?"})

        # 1 screenshot -> no multi-monitor label
        msgs_single = build_openai_messages(
            system_prompt="sys",
            history=[],
            user_text="what is on my screen?",
            screenshots_b64=["base64_shot1"],
            supports_vision=True,
        )
        parts_single = msgs_single[-1]["content"]
        self.assertEqual(len(parts_single), 2)
        self.assertEqual(parts_single[0]["type"], "image_url")
        self.assertEqual(parts_single[1], {"type": "text", "text": "what is on my screen?"})

    def test_gemini_multi_monitor_labels(self):
        """Verify multiple screenshots in build_gemini_payload get Screen 1 (Primary):, Screen 2: labels."""
        payload = build_gemini_payload(
            user_text="what is on my screens?",
            screenshots_b64=["base64_shot1", "base64_shot2"],
            history=[],
            system_prompt="sys",
            supports_vision=True,
        )
        parts = payload["contents"][0]["parts"]

        self.assertEqual(parts[0], {"text": "Screen 1 (Primary):"})
        self.assertEqual(parts[1]["inline_data"]["data"], "base64_shot1")
        self.assertEqual(parts[2], {"text": "Screen 2:"})
        self.assertEqual(parts[3]["inline_data"]["data"], "base64_shot2")
        self.assertEqual(parts[4], {"text": "what is on my screens?"})

        # Single screenshot -> no label
        payload_single = build_gemini_payload(
            user_text="what is on my screen?",
            screenshots_b64=["base64_shot1"],
            history=[],
            system_prompt="sys",
            supports_vision=True,
        )
        parts_single = payload_single["contents"][0]["parts"]
        self.assertEqual(len(parts_single), 2)
        self.assertEqual(parts_single[0]["inline_data"]["data"], "base64_shot1")
        self.assertEqual(parts_single[1], {"text": "what is on my screen?"})

    # ── 2. Claude Vision Gating, Multi-Monitor & Error Translation ───────────

    def test_claude_multi_monitor_and_vision_gating(self):
        """Verify ClaudeProvider gates vision and adds multi-monitor labels."""
        provider = ClaudeProvider()

        # A: Vision supported with 2 screenshots
        with patch.object(provider, "supports_vision", return_value=True):
            mock_stream_ctx = MagicMock()
            mock_stream = AsyncMock()

            async def _fake_stream():
                yield "hello"

            mock_stream.text_stream = _fake_stream()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

            provider._client = MagicMock()
            provider._client.messages.stream = MagicMock(return_value=mock_stream_ctx)

            async def run_call():
                chunks = []
                async for ch in provider.stream_response(
                    user_text="check screens",
                    screenshots_b64=["shot1", "shot2"],
                    history=[],
                    system_prompt="sys",
                    model="claude-sonnet-4-6",
                ):
                    chunks.append(ch)
                return chunks

            res = asyncio.run(run_call())
            self.assertEqual(res, ["hello"])

            call_kwargs = provider._client.messages.stream.call_args[1]
            sent_content = call_kwargs["messages"][0]["content"]
            self.assertEqual(sent_content[0], {"type": "text", "text": "Screen 1 (Primary):"})
            self.assertEqual(sent_content[1]["type"], "image")
            self.assertEqual(sent_content[2], {"type": "text", "text": "Screen 2:"})
            self.assertEqual(sent_content[3]["type"], "image")
            self.assertEqual(sent_content[4], {"type": "text", "text": "check screens"})

        # B: Vision NOT supported -> screenshots omitted
        with patch.object(provider, "supports_vision", return_value=False):
            provider._client.messages.stream = MagicMock(return_value=mock_stream_ctx)

            async def run_blind():
                chunks = []
                async for ch in provider.stream_response(
                    user_text="check screens",
                    screenshots_b64=["shot1", "shot2"],
                    history=[],
                    system_prompt="sys",
                    model="text-only-model",
                ):
                    chunks.append(ch)
                return chunks

            asyncio.run(run_blind())
            call_kwargs = provider._client.messages.stream.call_args[1]
            sent_content = call_kwargs["messages"][0]["content"]
            self.assertEqual(len(sent_content), 1)
            self.assertEqual(sent_content[0], {"type": "text", "text": "check screens"})

    def test_claude_error_translation(self):
        """Verify Claude SDK exceptions translate into typed provider errors."""
        class MockAnthropicAuthError(Exception):
            pass
        class MockAnthropicRateLimitError(Exception):
            pass
        class MockAnthropicConnError(Exception):
            pass
        class MockAnthropicServerError(Exception):
            pass

        auth_err = translate_anthropic_error(MockAnthropicAuthError("invalid api key"))
        self.assertIsInstance(auth_err, AuthenticationError)
        self.assertEqual(auth_err.status_code, 401)

        rate_err = translate_anthropic_error(MockAnthropicRateLimitError("too many requests"))
        self.assertIsInstance(rate_err, RateLimitError)
        self.assertEqual(rate_err.status_code, 429)

        conn_err = translate_anthropic_error(MockAnthropicConnError("APIConnectionError: connection dropped"))
        self.assertIsInstance(conn_err, ConnectionError)

        server_err = translate_anthropic_error(MockAnthropicServerError("InternalServerError: upstream down"))
        self.assertIsInstance(server_err, ServerError)

    # ── 3. Ollama Vision Gating & Error Translation ──────────────────────────

    def test_ollama_vision_gating(self):
        """Verify OllamaProvider attaches images only when supports_vision is True."""
        provider = OllamaProvider()

        # Vision False -> no images attached in user_msg
        with patch.object(provider, "supports_vision", return_value=False):
            mock_resp = _MockStreamResponse([
                json.dumps({"message": {"content": "ok"}, "done": True})
            ])
            mock_client = _MockAsyncClient(mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                async def run_ollama():
                    chunks = []
                    async for ch in provider.stream_response(
                        user_text="hi",
                        screenshots_b64=["shot_b64"],
                        history=[],
                        system_prompt="sys",
                        model="llama3.2:3b",
                    ):
                        chunks.append(ch)
                    return chunks

                asyncio.run(run_ollama())

            post_json = mock_client.last_json
            user_msg = post_json["messages"][-1]
            self.assertNotIn("images", user_msg)
            self.assertEqual(user_msg["content"], "hi")

        # Vision True -> images attached
        with patch.object(provider, "supports_vision", return_value=True):
            mock_client = _MockAsyncClient(mock_resp)
            with patch("httpx.AsyncClient", return_value=mock_client):
                async def run_ollama_vision():
                    chunks = []
                    async for ch in provider.stream_response(
                        user_text="hi",
                        screenshots_b64=["shot_b64"],
                        history=[],
                        system_prompt="sys",
                        model="qwen2-vl:7b",
                    ):
                        chunks.append(ch)
                    return chunks

                asyncio.run(run_ollama_vision())

            post_json = mock_client.last_json
            user_msg = post_json["messages"][-1]
            self.assertIn("images", user_msg)
            self.assertEqual(user_msg["images"], ["shot_b64"])

    def test_ollama_error_translation(self):
        """Verify format_ollama_error maps HTTP codes to typed provider errors."""
        err_404 = format_ollama_error(404, "model not found", model="qwen:7b")
        self.assertIsInstance(err_404, NotFoundError)
        self.assertIn("ollama pull", str(err_404))

        err_429 = format_ollama_error(429, "rate limited")
        self.assertIsInstance(err_429, RateLimitError)

        err_400 = format_ollama_error(400, "invalid options", model="qwen:7b")
        self.assertIsInstance(err_400, BadRequestError)

        err_500 = format_ollama_error(500, "internal crash")
        self.assertIsInstance(err_500, ServerError)

    # ── 4. Copilot Payload Reuse & Vision Gating ─────────────────────────────

    @patch("ai.github_copilot_provider.load_github_token", return_value="fake_gh_tok")
    def test_copilot_payload_reuse_and_vision_gating(self, mock_tok):
        """Verify GitHubCopilotProvider reuses build_openai_messages and respects vision gating."""
        provider = GitHubCopilotProvider()
        provider._get_copilot_token = AsyncMock(return_value="fake_copilot_tok")

        mock_resp = _MockStreamResponse([
            "data: " + json.dumps({"choices": [{"delta": {"content": "copilot reply"}}]}),
            "data: [DONE]",
        ])
        mock_client = _MockAsyncClient(mock_resp)

        # Vision True with 2 screens
        with patch.object(provider, "supports_vision", return_value=True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                async def run_copilot():
                    chunks = []
                    async for ch in provider.stream_response(
                        user_text="explain",
                        screenshots_b64=["s1", "s2"],
                        history=[],
                        system_prompt="sys",
                        model="gpt-4o",
                    ):
                        chunks.append(ch)
                    return chunks

                asyncio.run(run_copilot())

            body = mock_client.last_json
            user_parts = body["messages"][-1]["content"]
            self.assertEqual(user_parts[0], {"type": "text", "text": "Screen 1 (Primary):"})
            self.assertEqual(user_parts[2], {"type": "text", "text": "Screen 2:"})

        # Vision False -> screenshots stripped
        with patch.object(provider, "supports_vision", return_value=False):
            mock_client = _MockAsyncClient(mock_resp)
            with patch("httpx.AsyncClient", return_value=mock_client):
                async def run_copilot_blind():
                    chunks = []
                    async for ch in provider.stream_response(
                        user_text="explain text only",
                        screenshots_b64=["s1", "s2"],
                        history=[],
                        system_prompt="sys",
                        model="o1-preview",
                    ):
                        chunks.append(ch)
                    return chunks

                asyncio.run(run_copilot_blind())

            body = mock_client.last_json
            user_msg = body["messages"][-1]
            self.assertEqual(user_msg["content"], "explain text only")

    # ── 5. History Sanitization Tests ────────────────────────────────────────

    def test_history_sanitization_removes_point_and_draw_tags(self):
        """Verify _sanitize_history_text removes [POINT:...], [RECT:...], etc. but keeps text."""
        raw = (
            "I found the button! [POINT:150,250:search:screen1] "
            "Let me highlight it with a box [RECT:100,200,300,400:blue]. "
            "Click on it to proceed."
        )
        cleaned = _sanitize_history_text(raw)
        expected = "I found the button! Let me highlight it with a box . Click on it to proceed."
        self.assertEqual(cleaned, expected)
        self.assertNotIn("POINT", cleaned)
        self.assertNotIn("RECT", cleaned)
        self.assertIn("I found the button!", cleaned)
        self.assertIn("Click on it to proceed.", cleaned)

    def test_history_sanitization_preserves_plain_text(self):
        """Verify responses without control tags remain identical."""
        raw = "Here is a clean response with no tags at all. It just explains Python syntax."
        cleaned = _sanitize_history_text(raw)
        self.assertEqual(cleaned, raw)

    # ── 6. Context Window Budgeting Tests ────────────────────────────────────

    def test_context_budgeting_large_window(self):
        """Verify large context window preserves full history and attached documents."""
        model_info = ModelInfo(id="gemini-2.5-pro", display_name="Gemini 2.5 Pro", context_window=1_000_000)
        system_base = "Base system prompt rules."
        user_text = "Summarize this document."
        docs = [("notes.txt", "This is a document about machine learning. " * 500)]
        history = [
            Message(role="user", content=f"question {i}")
            for i in range(10)
        ]

        doc_extra, budgeted_hist = _budget_context(
            model_info=model_info,
            system_base=system_base,
            user_text=user_text,
            attached_docs=docs,
            history=history,
            screenshots_b64=["fake_b64"],
        )

        # Everything fits in 1M window
        self.assertEqual(len(budgeted_hist), 10)
        self.assertIn("notes.txt", doc_extra)
        self.assertNotIn("truncated", doc_extra)

    def test_context_budgeting_small_window_truncates_doc(self):
        """Verify small context window truncates large document while preserving recent history."""
        # LM Studio 8k context window
        model_info = ModelInfo(id="local-model", display_name="Local Model", context_window=4_000)
        system_base = "Base system prompt rules." * 50
        user_text = "What is in my notes?"
        # 100,000 chars document
        docs = [("huge.txt", "Lorem ipsum dolor sit amet. " * 4000)]
        history = [
            Message(role="user", content="oldest question"),
            Message(role="assistant", content="oldest answer"),
            Message(role="user", content="recent question"),
            Message(role="assistant", content="recent answer"),
        ]

        doc_extra, budgeted_hist = _budget_context(
            model_info=model_info,
            system_base=system_base,
            user_text=user_text,
            attached_docs=docs,
            history=history,
            screenshots_b64=["fake_b64"],
        )

        # Recent history should be preserved
        self.assertGreater(len(budgeted_hist), 0)
        self.assertEqual(budgeted_hist[-1].content, "recent answer")

        # Document must be truncated to prevent context overflow
        self.assertIn("truncated to fit model context window", doc_extra)
        self.assertLess(len(doc_extra), 15_000)

    # ── 7. Async Screen Capture ──────────────────────────────────────────────

    def test_async_screen_capture_in_companion_manager(self):
        """Verify capture_all_screens is called via asyncio.to_thread in _end_capture_and_process."""
        manager = CompanionManager()
        # Mock listener PCM capture
        manager._listener = MagicMock()
        manager._listener.stop_recording = MagicMock(return_value=b"\x00" * 32000)

        # Mock STT transcribe
        mock_stt = AsyncMock()
        mock_stt.transcribe = AsyncMock(return_value="hello genie")
        manager._get_stt = MagicMock(return_value=mock_stt)

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.supports_vision = MagicMock(return_value=False)
        mock_llm.get_capabilities = MagicMock(
            return_value=ModelInfo(id="m", display_name="m", context_window=128_000)
        )

        async def fake_stream(**kwargs):
            yield "answer"

        mock_llm.stream_response = fake_stream
        manager._get_llm = MagicMock(return_value=mock_llm)
        manager._play_lesson = AsyncMock()

        with patch("companion_manager.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = []
            with patch("companion_manager.active_window_title", return_value="Test App"):
                asyncio.run(manager._end_capture_and_process())

                # capture_all_screens must have been offloaded to a thread
                thread_targets = [call_arg[0][0] for call_arg in mock_to_thread.call_args_list]
                from screen.capture import capture_all_screens
                self.assertIn(capture_all_screens, thread_targets)


if __name__ == "__main__":
    unittest.main()
