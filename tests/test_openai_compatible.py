"""
Unit tests for the reusable OpenAI-compatible provider layer.
Verifies message construction, multimodal vision formatting, SSE stream parsing,
HTTP error mapping, and provider initialization without making network calls.
"""

import asyncio
import json
import unittest

from ai.base_provider import Message
from ai.openai_compatible_provider import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NotFoundError,
    OpenAIProviderError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    build_openai_messages,
    format_http_error,
    parse_sse_chunk,
    parse_sse_stream,
)
from ai.openai_provider import OpenAIProvider
from ai.lmstudio_provider import LMStudioProvider


class TestOpenAICompatible(unittest.TestCase):

    def test_text_message_construction(self):
        """Verify standard text-only message construction with system prompt and history."""
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        messages = build_openai_messages(
            system_prompt="You are Genie.",
            history=history,
            user_text="Help me with Python.",
            screenshots_b64=None,
            supports_vision=True,
        )

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "system", "content": "You are Genie."})
        self.assertEqual(messages[1], {"role": "user", "content": "Hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Hi there!"})
        self.assertEqual(messages[3], {"role": "user", "content": "Help me with Python."})

    def test_multimodal_message_construction(self):
        """Verify image_url payload generation when screenshots are present and vision is supported."""
        history = [Message(role="user", content="Look at this")]
        screenshots = ["fake_base64_jpeg_data"]

        messages = build_openai_messages(
            system_prompt="System instructions",
            history=history,
            user_text="What is on my screen?",
            screenshots_b64=screenshots,
            supports_vision=True,
        )

        self.assertEqual(len(messages), 3)
        user_msg = messages[2]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 2)
        self.assertEqual(
            user_msg["content"][0],
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,fake_base64_jpeg_data",
                    "detail": "high",
                },
            },
        )
        self.assertEqual(
            user_msg["content"][1],
            {"type": "text", "text": "What is on my screen?"},
        )

    def test_multiple_screenshots(self):
        """Verify that multi-monitor setups (multiple screenshots) are formatted properly."""
        screenshots = ["monitor_1_data", "monitor_2_data", "monitor_3_data"]
        messages = build_openai_messages(
            system_prompt="",
            history=[],
            user_text="Which monitor has the browser?",
            screenshots_b64=screenshots,
            supports_vision=True,
        )

        self.assertEqual(len(messages), 1)
        user_parts = messages[0]["content"]
        self.assertEqual(len(user_parts), 4)  # 3 images + 1 text
        for i in range(3):
            self.assertEqual(user_parts[i]["type"], "image_url")
            self.assertEqual(
                user_parts[i]["image_url"]["url"],
                f"data:image/jpeg;base64,{screenshots[i]}",
            )
        self.assertEqual(user_parts[3]["text"], "Which monitor has the browser?")

    def test_vision_not_supported_fallback(self):
        """Verify screenshots are stripped if model does not support vision."""
        screenshots = ["screen_data"]
        messages = build_openai_messages(
            system_prompt="Test system",
            history=[],
            user_text="Analyze text",
            screenshots_b64=screenshots,
            supports_vision=False,
        )

        self.assertEqual(len(messages), 2)
        # Content must remain string, not multimodal list
        self.assertEqual(messages[1], {"role": "user", "content": "Analyze text"})

    def test_sse_chunk_parsing(self):
        """Verify SSE line parsing with standard delta and malformed lines."""
        # Standard delta
        line = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        self.assertEqual(parse_sse_chunk(line), "Hello")

        # Delta with 'text' fallback
        line_text = 'data: {"choices": [{"delta": {"text": " world"}}]}'
        self.assertEqual(parse_sse_chunk(line_text), " world")

        # Empty / metadata / comment lines
        self.assertIsNone(parse_sse_chunk(": keep-alive ping"))
        self.assertIsNone(parse_sse_chunk("data: "))
        self.assertIsNone(parse_sse_chunk('data: {"choices": [{"delta": {}}]}'))

        # Malformed JSON
        self.assertIsNone(parse_sse_chunk("data: {broken json"))

    def test_sse_done_handling(self):
        """Verify [DONE] sentinel handling in chunk parser and async stream."""
        self.assertIsNone(parse_sse_chunk("data: [DONE]"))

        async def _test_stream():
            lines = [
                'data: {"choices": [{"delta": {"content": "One "}}]}',
                'data: {"choices": [{"delta": {"content": "Two "}}]}',
                'data: [DONE]',
                'data: {"choices": [{"delta": {"content": "Three"}}]}',  # Should not be reached
            ]

            async def mock_aiter(items):
                for item in items:
                    yield item

            chunks = []
            async for chunk in parse_sse_stream(mock_aiter(lines)):
                chunks.append(chunk)

            self.assertEqual("".join(chunks), "One Two ")

        asyncio.run(_test_stream())

    def test_http_error_mapping(self):
        """Verify translation of common HTTP status codes into typed exceptions."""
        err_400 = format_http_error("TestProvider", 400, '{"error": {"message": "invalid model"}}')
        self.assertIsInstance(err_400, BadRequestError)
        self.assertIn("invalid model", str(err_400))

        err_401 = format_http_error("TestProvider", 401)
        self.assertIsInstance(err_401, AuthenticationError)

        err_403 = format_http_error("TestProvider", 403)
        self.assertIsInstance(err_403, PermissionDeniedError)

        err_404 = format_http_error("TestProvider", 404, host="http://localhost:1234")
        self.assertIsInstance(err_404, NotFoundError)
        self.assertIn("http://localhost:1234", str(err_404))

        err_429 = format_http_error("TestProvider", 429)
        self.assertIsInstance(err_429, RateLimitError)

        err_500 = format_http_error("TestProvider", 500)
        self.assertIsInstance(err_500, ServerError)

    def test_provider_initialization(self):
        """Verify that OpenAI and LM Studio providers instantiate cleanly with proper metadata."""
        import os
        old_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "sk-test-key-for-unit-testing"
            openai_p = OpenAIProvider()
            self.assertEqual(openai_p.provider_id, "openai")
            self.assertEqual(openai_p.display_name, "OpenAI")
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

        lm_p = LMStudioProvider()
        self.assertEqual(lm_p.provider_id, "lmstudio")
        self.assertEqual(lm_p.display_name, "LM Studio")
        self.assertEqual(lm_p.default_model, "local-model")


if __name__ == "__main__":
    unittest.main()
