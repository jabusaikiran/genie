"""
Google Gemini provider — uses the REST SSE streaming API directly (no extra deps).
Default model: gemini-2.5-flash (fast, cheap, vision-capable).

Hardened in Phase 2, Step 3:
  • Explicit HTTP error translation (400, 401/403, 404, 429, 5xx).
  • Network timeout & connection error mapping.
  • Safety blocking detection at promptFeedback and candidate finishReason levels.
  • Configurable max_output_tokens and temperature (config/init/call).
  • Model registry vision capability verification before attaching screenshots.
  • Robust SSE chunk parsing and malformed-line resilience.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List, Optional

import httpx

from ai.base_provider import BaseLLMProvider, Message
from ai.openai_compatible_provider import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NotFoundError,
    OpenAIProviderError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
)
from config import cfg

log = logging.getLogger("genie.ai.gemini")

DEFAULT_MODEL = "gemini-2.5-flash"
STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
)

BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
}


class GeminiSafetyError(OpenAIProviderError):
    """Raised when Gemini content filters or safety policies block a prompt or response."""

    def __init__(
        self,
        message: str,
        *,
        block_reason: Optional[str] = None,
        raw_error: Optional[Any] = None,
    ):
        super().__init__(
            message,
            provider="gemini",
            status_code=400,
            error_code=block_reason or "SAFETY_BLOCKED",
            raw_error=raw_error,
        )
        self.block_reason = block_reason


def _is_blocked_finish_reason(reason: str) -> bool:
    """Check if candidate finishReason indicates content suppression or policy block."""
    if not reason:
        return False
    r = reason.upper()
    return (
        r in BLOCKED_FINISH_REASONS
        or any(k in r for k in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED", "SPII"))
    )


def format_gemini_error(
    status_code: int,
    response_text: str = "",
    model: str = "",
) -> OpenAIProviderError:
    """Translate a Gemini API HTTP error status and response payload into a typed error."""
    detail = ""
    error_status = ""
    raw = None

    if response_text:
        try:
            parsed = json.loads(response_text)
            raw = parsed
            if isinstance(parsed, dict):
                err_dict = parsed.get("error")
                if isinstance(err_dict, dict):
                    detail = err_dict.get("message", "")
                    error_status = err_dict.get("status", "")
                elif isinstance(err_dict, str):
                    detail = err_dict
            elif isinstance(parsed, str):
                detail = parsed
        except Exception:
            detail = response_text.strip()[:300]

    lowered_detail = detail.lower()

    # Detect invalid API key when Google returns HTTP 400 with API key messaging
    if (
        "api key not valid" in lowered_detail
        or "api_key_invalid" in lowered_detail
        or "api key expired" in lowered_detail
    ):
        return AuthenticationError(
            f"Gemini API key is invalid: {detail}",
            provider="gemini",
            status_code=status_code,
            error_code=error_status or "API_KEY_INVALID",
            raw_error=raw,
        )

    if status_code in (401, 403):
        return AuthenticationError(
            f"Gemini authentication failed (HTTP {status_code}): {detail or 'Invalid or unauthorized API key.'}",
            provider="gemini",
            status_code=status_code,
            error_code=error_status or "UNAUTHORIZED",
            raw_error=raw,
        )

    if status_code == 400:
        return BadRequestError(
            f"Gemini bad request ({model or 'default'}): {detail or 'Invalid request parameters or payload.'}",
            provider="gemini",
            status_code=400,
            error_code=error_status or "INVALID_ARGUMENT",
            raw_error=raw,
        )

    if status_code == 404:
        return NotFoundError(
            f"Gemini model or endpoint not found ({model}): {detail or 'Model not found or unsupported.'}",
            provider="gemini",
            status_code=404,
            error_code=error_status or "NOT_FOUND",
            raw_error=raw,
        )

    if status_code == 429:
        return RateLimitError(
            f"Gemini rate limit / quota exceeded: {detail or 'Too many requests or quota exhausted.'}",
            provider="gemini",
            status_code=429,
            error_code=error_status or "RESOURCE_EXHAUSTED",
            raw_error=raw,
        )

    if 500 <= status_code < 600:
        return ServerError(
            f"Gemini server error (HTTP {status_code}): {detail or 'Google Generative AI service error. Please try again later.'}",
            provider="gemini",
            status_code=status_code,
            error_code=error_status or "SERVER_ERROR",
            raw_error=raw,
        )

    return OpenAIProviderError(
        f"Gemini API error (HTTP {status_code}): {detail or 'Unexpected response.'}",
        provider="gemini",
        status_code=status_code,
        error_code=error_status,
        raw_error=raw,
    )


def build_gemini_payload(
    user_text: str,
    screenshots_b64: Optional[List[str]] = None,
    history: Optional[List[Message]] = None,
    system_prompt: str = "",
    supports_vision: bool = True,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Construct the Gemini REST JSON payload with history, system instruction, and parameters."""
    contents = []
    if history:
        for msg in history:
            if not msg.content:
                continue
            role = "user" if msg.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

    parts: list = []
    if screenshots_b64 and supports_vision:
        valid_shots = [s for s in screenshots_b64 if s]
        is_multi = len(valid_shots) > 1
        for idx, img_b64 in enumerate(valid_shots, start=1):
            if is_multi:
                label = f"Screen {idx} (Primary):" if idx == 1 else f"Screen {idx}:"
                parts.append({"text": label})
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": img_b64},
            })
    if user_text:
        parts.append({"text": user_text})
    elif not parts:
        parts.append({"text": " "})

    contents.append({"role": "user", "parts": parts})

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    return body


class GeminiProvider(BaseLLMProvider):
    provider_id = "gemini"
    display_name = "Google Gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key if api_key is not None else cfg.google_api_key
        self.default_model = default_model or getattr(cfg, "gemini_default_model", DEFAULT_MODEL)
        self.max_tokens = max_tokens if max_tokens is not None else getattr(cfg, "gemini_max_tokens", 1024)
        self.temperature = temperature if temperature is not None else getattr(cfg, "gemini_temperature", 0.7)
        self.timeout = timeout

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        if not self._api_key:
            raise AuthenticationError(
                "Gemini API key is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env.",
                provider="gemini",
                status_code=401,
            )

        model = model or self.default_model or DEFAULT_MODEL
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_temperature = temperature if temperature is not None else self.temperature

        # Model registry vision check
        vision_capable = self.supports_vision(model)
        if screenshots_b64 and not vision_capable:
            log.info(
                "Model '%s' does not support vision; omitting %d screenshot(s).",
                model,
                len(screenshots_b64),
            )

        body = build_gemini_payload(
            user_text=user_text,
            screenshots_b64=screenshots_b64,
            history=history,
            system_prompt=system_prompt,
            supports_vision=vision_capable,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
        )

        url = f"{STREAM_URL.format(model=model)}?alt=sse&key={self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code >= 400:
                        error_bytes = await resp.aread()
                        error_text = error_bytes.decode("utf-8", errors="replace")
                        raise format_gemini_error(resp.status_code, error_text, model=model)

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # Inspect promptFeedback for safety blocks
                        prompt_feedback = obj.get("promptFeedback")
                        if isinstance(prompt_feedback, dict):
                            block_reason = prompt_feedback.get("blockReason")
                            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                                raise GeminiSafetyError(
                                    f"Gemini prompt blocked: {block_reason}",
                                    block_reason=block_reason,
                                    raw_error=prompt_feedback,
                                )

                        # Inspect candidates for safety blocks and extract text
                        for cand in obj.get("candidates", []):
                            finish_reason = cand.get("finishReason")
                            if finish_reason and _is_blocked_finish_reason(finish_reason):
                                raise GeminiSafetyError(
                                    f"Gemini response blocked by safety policy: {finish_reason}",
                                    block_reason=finish_reason,
                                    raw_error=cand,
                                )

                            for part in cand.get("content", {}).get("parts", []):
                                text = part.get("text", "")
                                if text:
                                    yield text
        except (OpenAIProviderError, GeminiSafetyError):
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ConnectionError(
                f"Unable to connect to Google Gemini API: {e}",
                provider="gemini",
            ) from e
        except httpx.TimeoutException as e:
            raise ConnectionError(
                f"Google Gemini API request timed out: {e}",
                provider="gemini",
            ) from e
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Network error communicating with Google Gemini API: {e}",
                provider="gemini",
            ) from e

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self._api_key}"
                )
                return r.status_code == 200
        except Exception:
            return False
