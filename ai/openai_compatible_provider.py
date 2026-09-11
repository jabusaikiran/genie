"""
Reusable OpenAI-compatible provider layer.

Provides shared payload construction, vision/multimodal formatting,
SSE stream parsing, and robust error translation for any provider
implementing the OpenAI chat-completions specification (/chat/completions).

Used by:
  • LMStudioProvider (direct HTTP streaming)
  • OpenAIProvider (shared message construction & error translation with official SDK)
  • Future: Groq, OpenRouter, etc.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ai.base_provider import BaseLLMProvider, Message, ModelInfo
from ai.model_registry import get_model_info
from config import cfg

log = logging.getLogger("genie.ai.openai_compat")


# ─── Exceptions ───────────────────────────────────────────────────────────────

class OpenAIProviderError(RuntimeError):
    """Base exception for errors communicating with an OpenAI-compatible provider."""
    def __init__(
        self,
        message: str,
        *,
        provider: str = "OpenAI-Compatible",
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        raw_error: Optional[Any] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.raw_error = raw_error


class AuthenticationError(OpenAIProviderError):
    """Raised on HTTP 401: missing, invalid, or expired API key."""
    pass


class PermissionDeniedError(OpenAIProviderError):
    """Raised on HTTP 403: forbidden, seat/quota restriction."""
    pass


class NotFoundError(OpenAIProviderError):
    """Raised on HTTP 404: model not found or invalid endpoint."""
    pass


class RateLimitError(OpenAIProviderError):
    """Raised on HTTP 429: rate limits, token quotas, or credits exhausted."""
    pass


class BadRequestError(OpenAIProviderError):
    """Raised on HTTP 400: invalid parameters, unsupported options, or payload size."""
    pass


class ServerError(OpenAIProviderError):
    """Raised on HTTP 5xx: upstream provider server or gateway error."""
    pass


class ConnectionError(OpenAIProviderError):
    """Raised when the provider host is unreachable or connection timed out."""
    pass


# ─── Error Translation ────────────────────────────────────────────────────────

def format_http_error(
    provider: str,
    status_code: int,
    response_text: str = "",
    host: str = "",
) -> OpenAIProviderError:
    """Translate an HTTP status code and response payload into a typed error."""
    detail = ""
    error_code = None
    if response_text:
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                err_obj = parsed.get("error", {})
                if isinstance(err_obj, dict):
                    detail = err_obj.get("message", "")
                    error_code = err_obj.get("code") or err_obj.get("type")
                elif isinstance(err_obj, str):
                    detail = err_obj
                else:
                    detail = response_text[:300]
        except Exception:
            detail = response_text[:300]

    suffix = f" Details: {detail}" if detail else ""
    log.warning("[%s] HTTP %d: %s", provider, status_code, detail or response_text[:200])

    if status_code == 400:
        return BadRequestError(
            f"[{provider}] Bad request (400).{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    elif status_code == 401:
        return AuthenticationError(
            f"[{provider}] Authentication failed (401). Please verify your API key.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    elif status_code == 403:
        return PermissionDeniedError(
            f"[{provider}] Access forbidden (403). Your key or account may lack permission for this model.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    elif status_code == 404:
        host_hint = f" at {host}" if host else ""
        return NotFoundError(
            f"[{provider}] Model or endpoint not found (404){host_hint}.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    elif status_code == 429:
        return RateLimitError(
            f"[{provider}] Rate limit or quota exceeded (429). Check your usage limits or credit balance.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    elif status_code >= 500:
        return ServerError(
            f"[{provider}] Server error ({status_code}). The service may be experiencing downtime.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )
    else:
        return OpenAIProviderError(
            f"[{provider}] Request failed with HTTP {status_code}.{suffix}",
            provider=provider, status_code=status_code, error_code=error_code, raw_error=response_text,
        )


def translate_sdk_error(provider: str, err: Exception) -> Exception:
    """Translate official openai SDK exceptions into clean OpenAIProviderErrors."""
    msg = str(err)
    log.warning("[%s] SDK Error: %s (%s)", provider, msg, type(err).__name__)
    err_type = type(err).__name__

    if "AuthenticationError" in err_type:
        return AuthenticationError(f"[{provider}] Authentication failed. Please verify your API key: {msg}", provider=provider, raw_error=err)
    elif "RateLimitError" in err_type:
        return RateLimitError(f"[{provider}] Rate limit or quota exceeded: {msg}", provider=provider, raw_error=err)
    elif "BadRequestError" in err_type:
        return BadRequestError(f"[{provider}] Bad request: {msg}", provider=provider, raw_error=err)
    elif "PermissionDeniedError" in err_type:
        return PermissionDeniedError(f"[{provider}] Permission denied: {msg}", provider=provider, raw_error=err)
    elif "NotFoundError" in err_type:
        return NotFoundError(f"[{provider}] Model not found: {msg}", provider=provider, raw_error=err)
    elif "APIConnectionError" in err_type:
        return ConnectionError(f"[{provider}] Could not connect to API server: {msg}", provider=provider, raw_error=err)
    elif "InternalServerError" in err_type:
        return ServerError(f"[{provider}] Provider internal server error: {msg}", provider=provider, raw_error=err)
    return OpenAIProviderError(f"[{provider}] {msg}", provider=provider, raw_error=err)


# ─── Message & Multimodal Construction ────────────────────────────────────────

def build_openai_messages(
    system_prompt: str,
    history: List[Message],
    user_text: str,
    screenshots_b64: Optional[List[str]] = None,
    supports_vision: bool = True,
) -> List[Dict[str, Any]]:
    """
    Construct standard OpenAI-compatible messages payload.
    Supports system prompt, history, text, and multimodal image_url content blocks.
    """
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    attach_images = bool(screenshots_b64) and supports_vision

    if attach_images:
        valid_shots = [s for s in (screenshots_b64 or []) if s]
        content_parts: List[Dict[str, Any]] = []
        is_multi = len(valid_shots) > 1
        for idx, img_b64 in enumerate(valid_shots, start=1):
            if is_multi:
                label = f"Screen {idx} (Primary):" if idx == 1 else f"Screen {idx}:"
                content_parts.append({"type": "text", "text": label})
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high",
                },
            })
        content_parts.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


# ─── SSE Stream Parser ────────────────────────────────────────────────────────

def parse_sse_chunk(line: str) -> Optional[str]:
    """
    Parse a single SSE data line from an OpenAI-compatible /chat/completions stream.
    Returns:
      - extracted string chunk if present
      - None if empty, comment, non-content, or [DONE]
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None

    data_str = stripped[5:].strip()
    if not data_str or data_str == "[DONE]":
        return None

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        return None

    first_choice = choices[0]
    delta = first_choice.get("delta") or {}
    content = delta.get("content")
    if content:
        return content

    # Some OpenAI-compatible servers put content in 'text'
    text = delta.get("text")
    if text:
        return text

    return None


async def parse_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async iterator that extracts text deltas from an SSE line stream until [DONE]."""
    async for line in lines:
        stripped = line.strip()
        if stripped == "data: [DONE]":
            break
        chunk = parse_sse_chunk(line)
        if chunk:
            yield chunk


# ─── OpenAICompatibleProvider Base Class ──────────────────────────────────────

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Base class for providers that communicate directly over HTTP with an
    OpenAI-compatible /chat/completions endpoint (e.g. LM Studio, Groq, OpenRouter).
    """

    provider_id: str = "openai_compatible"
    display_name: str = "OpenAI-Compatible"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        default_model: str = "",
        timeout: float = 120.0,
        custom_headers: Optional[Dict[str, str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._custom_headers = dict(custom_headers or {})
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def default_model(self) -> str:
        return self._default_model

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._custom_headers)
        return headers

    def build_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = True,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": stream,
        }

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        chosen_model = model or self._default_model or "default"
        supports_vis = self.supports_vision(chosen_model)

        messages = build_openai_messages(
            system_prompt=system_prompt,
            history=history,
            user_text=user_text,
            screenshots_b64=screenshots_b64,
            supports_vision=supports_vis,
        )

        payload = self.build_payload(chosen_model, messages, stream=True)
        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        err_body = await response.aread()
                        err_text = err_body.decode("utf-8", errors="replace")
                        raise format_http_error(
                            provider=self.display_name,
                            status_code=response.status_code,
                            response_text=err_text,
                            host=self._base_url,
                        )

                    async for chunk in parse_sse_stream(response.aiter_lines()):
                        yield chunk

            except (OpenAIProviderError, httpx.HTTPStatusError):
                raise
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"[{self.display_name}] Could not connect to server at {self._base_url}. Is the service running?",
                    provider=self.display_name,
                    raw_error=e,
                ) from e
            except httpx.TimeoutException as e:
                raise ConnectionError(
                    f"[{self.display_name}] Request timed out after {self._timeout}s to {self._base_url}.",
                    provider=self.display_name,
                    raw_error=e,
                ) from e
            except Exception as e:
                log.exception("[%s] Unexpected stream error: %s", self.display_name, e)
                raise OpenAIProviderError(
                    f"[{self.display_name}] Streaming request failed: {e}",
                    provider=self.display_name,
                    raw_error=e,
                ) from e

    async def health_check(self) -> bool:
        """Check if provider is reachable via GET /models or base URL."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{self._base_url}/models",
                    headers=self._get_headers(),
                )
                return r.status_code == 200
        except Exception:
            return False
