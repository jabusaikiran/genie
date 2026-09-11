"""
Unit tests for the hardened Google Gemini provider.
Verifies payload construction, multimodal vision gating, SSE streaming,
error translation, safety block detection, and network timeouts with mocked responses.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from ai.base_provider import Message
from ai.gemini_provider import (
    DEFAULT_MODEL,
    GeminiProvider,
    GeminiSafetyError,
    _is_blocked_finish_reason,
    build_gemini_payload,
    format_gemini_error,
)
from ai.openai_compatible_provider import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NotFoundError,
    OpenAIProviderError,
    RateLimitError,
    ServerError,
)


class MockResponse:
    """Mock HTTP streaming and buffered response."""

    def __init__(
        self,
        status_code: int = 200,
        lines: list[str] | None = None,
        content: bytes = b"",
    ):
        self.status_code = status_code
        self._lines = lines or []
        self._content = content

    async def aread(self) -> bytes:
        return self._content

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockAsyncClient:
    """Mock httpx.AsyncClient capturing requests and returning MockResponse."""

    def __init__(self, response: MockResponse | None = None, side_effect: Exception | None = None):
        self.response = response or MockResponse()
        self.side_effect = side_effect
        self.last_method = None
        self.last_url = None
        self.last_json = None

    def stream(self, method: str, url: str, json: dict | None = None, **kwargs):
        if self.side_effect:
            raise self.side_effect
        self.last_method = method
        self.last_url = url
        self.last_json = json
        return self.response

    async def get(self, url: str, **kwargs):
        if self.side_effect:
            raise self.side_effect
        self.last_method = "GET"
        self.last_url = url
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestGeminiProvider(unittest.TestCase):

    def test_text_payload_construction(self):
        """Verify message history, user prompt, and generation config formatting."""
        history = [
            Message(role="user", content="Hello Gemini"),
            Message(role="assistant", content="Hello! How can I help?"),
        ]
        payload = build_gemini_payload(
            user_text="What is 2+2?",
            screenshots_b64=None,
            history=history,
            system_prompt="You are a helpful assistant.",
            supports_vision=True,
            max_tokens=2048,
            temperature=0.4,
        )

        self.assertIn("contents", payload)
        self.assertEqual(len(payload["contents"]), 3)
        # History turn 1
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Hello Gemini")
        # History turn 2
        self.assertEqual(payload["contents"][1]["role"], "model")
        self.assertEqual(payload["contents"][1]["parts"][0]["text"], "Hello! How can I help?")
        # Current user turn
        self.assertEqual(payload["contents"][2]["role"], "user")
        self.assertEqual(payload["contents"][2]["parts"][0]["text"], "What is 2+2?")

        # System instruction
        self.assertEqual(
            payload["systemInstruction"]["parts"][0]["text"],
            "You are a helpful assistant.",
        )
        # Generation config
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 2048)
        self.assertEqual(payload["generationConfig"]["temperature"], 0.4)

    def test_multimodal_payload_construction_vision_supported(self):
        """Verify screenshots are formatted as inline_data parts when model supports vision."""
        payload = build_gemini_payload(
            user_text="What is on this screen?",
            screenshots_b64=["fake_base64_data_1", "fake_base64_data_2"],
            history=[],
            system_prompt="",
            supports_vision=True,
        )

        contents = payload["contents"]
        self.assertEqual(len(contents), 1)
        parts = contents[0]["parts"]
        # 2 images + 1 text prompt
        self.assertEqual(len(parts), 3)
        self.assertEqual(
            parts[0],
            {"inline_data": {"mime_type": "image/jpeg", "data": "fake_base64_data_1"}},
        )
        self.assertEqual(
            parts[1],
            {"inline_data": {"mime_type": "image/jpeg", "data": "fake_base64_data_2"}},
        )
        self.assertEqual(parts[2], {"text": "What is on this screen?"})

    def test_multimodal_payload_construction_vision_unsupported(self):
        """Verify screenshots are cleanly omitted when model does NOT support vision."""
        payload = build_gemini_payload(
            user_text="Summarize the screen",
            screenshots_b64=["fake_base64_data"],
            history=[],
            system_prompt="",
            supports_vision=False,
        )

        contents = payload["contents"]
        self.assertEqual(len(contents), 1)
        parts = contents[0]["parts"]
        # Only text part should be present
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], {"text": "Summarize the screen"})

    def test_blocked_finish_reason_helper(self):
        """Verify detection of blocking candidate finish reasons."""
        self.assertTrue(_is_blocked_finish_reason("SAFETY"))
        self.assertTrue(_is_blocked_finish_reason("safety"))
        self.assertTrue(_is_blocked_finish_reason("FINISH_REASON_SAFETY"))
        self.assertTrue(_is_blocked_finish_reason("RECITATION"))
        self.assertTrue(_is_blocked_finish_reason("BLOCKLIST"))
        self.assertTrue(_is_blocked_finish_reason("PROHIBITED_CONTENT"))
        self.assertTrue(_is_blocked_finish_reason("SPII"))

        self.assertFalse(_is_blocked_finish_reason("STOP"))
        self.assertFalse(_is_blocked_finish_reason("MAX_TOKENS"))
        self.assertFalse(_is_blocked_finish_reason(""))
        self.assertFalse(_is_blocked_finish_reason(None))

    def test_http_error_mapping_400_bad_request(self):
        """Verify HTTP 400 without API key error is mapped to BadRequestError."""
        err_json = json.dumps({
            "error": {
                "code": 400,
                "message": "Invalid JSON payload: unknown field 'foo'",
                "status": "INVALID_ARGUMENT",
            }
        })
        err = format_gemini_error(400, err_json, model="gemini-2.5-flash")
        self.assertIsInstance(err, BadRequestError)
        self.assertEqual(err.status_code, 400)
        self.assertIn("unknown field", str(err))

    def test_http_error_mapping_400_invalid_api_key(self):
        """Verify HTTP 400 with API key failure message is mapped to AuthenticationError."""
        err_json = json.dumps({
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
            }
        })
        err = format_gemini_error(400, err_json, model="gemini-2.5-flash")
        self.assertIsInstance(err, AuthenticationError)
        self.assertIn("invalid", str(err).lower())

    def test_http_error_mapping_401_403(self):
        """Verify HTTP 401 and 403 are mapped to AuthenticationError."""
        err_401 = format_gemini_error(
            401,
            json.dumps({"error": {"message": "Invalid credentials", "status": "UNAUTHENTICATED"}}),
        )
        self.assertIsInstance(err_401, AuthenticationError)
        self.assertEqual(err_401.status_code, 401)

        err_403 = format_gemini_error(
            403,
            json.dumps({"error": {"message": "Permission denied", "status": "PERMISSION_DENIED"}}),
        )
        self.assertIsInstance(err_403, AuthenticationError)
        self.assertEqual(err_403.status_code, 403)

    def test_http_error_mapping_404(self):
        """Verify HTTP 404 is mapped to NotFoundError with model context."""
        err_json = json.dumps({
            "error": {
                "code": 404,
                "message": "models/gemini-invalid is not found",
                "status": "NOT_FOUND",
            }
        })
        err = format_gemini_error(404, err_json, model="gemini-invalid")
        self.assertIsInstance(err, NotFoundError)
        self.assertEqual(err.status_code, 404)
        self.assertIn("gemini-invalid", str(err))

    def test_http_error_mapping_429(self):
        """Verify HTTP 429 quota exhaustion is mapped to RateLimitError."""
        err_json = json.dumps({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
            }
        })
        err = format_gemini_error(429, err_json)
        self.assertIsInstance(err, RateLimitError)
        self.assertEqual(err.status_code, 429)

    def test_http_error_mapping_5xx(self):
        """Verify HTTP 500 and 503 are mapped to ServerError."""
        err_500 = format_gemini_error(500, "Internal Server Error")
        self.assertIsInstance(err_500, ServerError)
        self.assertEqual(err_500.status_code, 500)

        err_503 = format_gemini_error(
            503,
            json.dumps({"error": {"message": "Model is overloaded", "status": "UNAVAILABLE"}}),
        )
        self.assertIsInstance(err_503, ServerError)
        self.assertEqual(err_503.status_code, 503)

    def test_stream_response_success(self):
        """Verify successful SSE streaming yields text chunks incrementally."""
        sse_lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": " world!"}]}}]}',
            "data: [DONE]",
        ]
        mock_resp = MockResponse(status_code=200, lines=sse_lines)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key_123")

        async def run():
            chunks = []
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for chunk in provider.stream_response("Say hello", [], [], "System"):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        self.assertEqual(chunks, ["Hello", " world!"])
        self.assertIn("test_key_123", mock_client.last_url)

    def test_stream_response_missing_api_key(self):
        """Verify AuthenticationError is raised when API key is not configured."""
        provider = GeminiProvider(api_key="")

        async def run():
            async for _ in provider.stream_response("test", [], [], ""):
                pass

        with self.assertRaises(AuthenticationError):
            asyncio.run(run())

    def test_stream_response_http_error(self):
        """Verify HTTP 404 response in stream raises NotFoundError."""
        err_payload = json.dumps({"error": {"message": "Model not found", "status": "NOT_FOUND"}}).encode("utf-8")
        mock_resp = MockResponse(status_code=404, content=err_payload)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("test", [], [], ""):
                    pass

        with self.assertRaises(NotFoundError):
            asyncio.run(run())

    def test_stream_response_network_error(self):
        """Verify network connection failure raises ConnectionError."""
        connect_err = httpx.ConnectError("Failed to establish a new connection")
        mock_client = MockAsyncClient(side_effect=connect_err)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("test", [], [], ""):
                    pass

        with self.assertRaises(ConnectionError):
            asyncio.run(run())

    def test_stream_response_timeout(self):
        """Verify timeout during streaming raises ConnectionError."""
        timeout_err = httpx.ReadTimeout("The read operation timed out")
        mock_client = MockAsyncClient(side_effect=timeout_err)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("test", [], [], ""):
                    pass

        with self.assertRaises(ConnectionError):
            asyncio.run(run())

    def test_safety_blocking_prompt_feedback(self):
        """Verify GeminiSafetyError is raised when promptFeedback contains a blockReason."""
        sse_lines = [
            'data: {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []}}',
        ]
        mock_resp = MockResponse(status_code=200, lines=sse_lines)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("unsafe prompt", [], [], ""):
                    pass

        with self.assertRaises(GeminiSafetyError) as ctx:
            asyncio.run(run())

        self.assertEqual(ctx.exception.block_reason, "SAFETY")
        self.assertIn("prompt blocked", str(ctx.exception).lower())

    def test_safety_blocking_candidate_finish_reason(self):
        """Verify GeminiSafetyError is raised when candidate has finishReason SAFETY."""
        sse_lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Partial"}]}, "finishReason": "SAFETY"}]}',
        ]
        mock_resp = MockResponse(status_code=200, lines=sse_lines)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("test prompt", [], [], ""):
                    pass

        with self.assertRaises(GeminiSafetyError) as ctx:
            asyncio.run(run())

        self.assertEqual(ctx.exception.block_reason, "SAFETY")
        self.assertIn("safety policy", str(ctx.exception).lower())

    def test_safety_blocking_recitation(self):
        """Verify GeminiSafetyError is raised when finishReason is RECITATION."""
        sse_lines = [
            'data: {"candidates": [{"finishReason": "RECITATION"}]}',
        ]
        mock_resp = MockResponse(status_code=200, lines=sse_lines)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for _ in provider.stream_response("recite song", [], [], ""):
                    pass

        with self.assertRaises(GeminiSafetyError) as ctx:
            asyncio.run(run())

        self.assertEqual(ctx.exception.block_reason, "RECITATION")

    def test_malformed_and_empty_chunks_tolerated(self):
        """Verify SSE stream handles comments, empty lines, and malformed JSON gracefully."""
        sse_lines = [
            ": ping",
            "",
            "data: not valid json at all",
            'data: {"candidates": [{"content": {"parts": [{"text": "Valid text"}]}}]}',
            "data:   ",
            "data: [DONE]",
        ]
        mock_resp = MockResponse(status_code=200, lines=sse_lines)
        mock_client = MockAsyncClient(response=mock_resp)

        provider = GeminiProvider(api_key="test_key")

        async def run():
            chunks = []
            with patch("httpx.AsyncClient", return_value=mock_client):
                async for chunk in provider.stream_response("test", [], [], ""):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        self.assertEqual(chunks, ["Valid text"])

    def test_configurable_parameters(self):
        """Verify max_tokens and temperature can be set at init or per stream call."""
        provider = GeminiProvider(
            api_key="test_key",
            default_model="gemini-2.5-pro",
            max_tokens=4096,
            temperature=0.2,
        )

        mock_resp = MockResponse(status_code=200, lines=['data: {"candidates": []}'])
        mock_client = MockAsyncClient(response=mock_resp)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                # Override max_tokens and temperature per-call
                async for _ in provider.stream_response(
                    "test", [], [], "", max_tokens=512, temperature=0.9
                ):
                    pass

        asyncio.run(run())
        sent_body = mock_client.last_json
        self.assertEqual(sent_body["generationConfig"]["maxOutputTokens"], 512)
        self.assertEqual(sent_body["generationConfig"]["temperature"], 0.9)

    def test_vision_gating_in_stream_response(self):
        """Verify screenshots are omitted when provider.supports_vision returns False."""
        provider = GeminiProvider(api_key="test_key")
        mock_resp = MockResponse(status_code=200, lines=['data: {"candidates": []}'])
        mock_client = MockAsyncClient(response=mock_resp)

        async def run():
            with patch.object(provider, "supports_vision", return_value=False):
                with patch("httpx.AsyncClient", return_value=mock_client):
                    async for _ in provider.stream_response(
                        "Analyze this image",
                        ["b64_image_content"],
                        [],
                        "",
                    ):
                        pass

        asyncio.run(run())
        sent_body = mock_client.last_json
        user_parts = sent_body["contents"][0]["parts"]
        # Screenshots should NOT be in user_parts
        self.assertEqual(len(user_parts), 1)
        self.assertEqual(user_parts[0], {"text": "Analyze this image"})

    def test_health_check(self):
        """Verify health_check logic on success, failure, and missing key."""
        # 1. Healthy
        mock_client_ok = MockAsyncClient(response=MockResponse(status_code=200))
        provider_ok = GeminiProvider(api_key="valid_key")
        with patch("httpx.AsyncClient", return_value=mock_client_ok):
            self.assertTrue(asyncio.run(provider_ok.health_check()))

        # 2. Server returns error
        mock_client_err = MockAsyncClient(response=MockResponse(status_code=403))
        with patch("httpx.AsyncClient", return_value=mock_client_err):
            self.assertFalse(asyncio.run(provider_ok.health_check()))

        # 3. Missing API key
        provider_no_key = GeminiProvider(api_key="")
        self.assertFalse(asyncio.run(provider_no_key.health_check()))


if __name__ == "__main__":
    unittest.main()
