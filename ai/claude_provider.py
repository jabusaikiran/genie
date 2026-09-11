from typing import AsyncIterator, List

import anthropic

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

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def translate_anthropic_error(err: Exception) -> OpenAIProviderError:
    """Translate Anthropic SDK exceptions into typed provider errors."""
    msg = str(err)
    err_type = type(err).__name__
    status_code = getattr(err, "status_code", None)

    if "AuthenticationError" in err_type or "AuthError" in err_type or status_code == 401:
        return AuthenticationError(
            f"[Anthropic Claude] Authentication failed. Check your ANTHROPIC_API_KEY: {msg}",
            provider="claude",
            status_code=401,
            raw_error=err,
        )
    elif "PermissionDeniedError" in err_type or status_code == 403:
        return PermissionDeniedError(
            f"[Anthropic Claude] Permission denied: {msg}",
            provider="claude",
            status_code=403,
            raw_error=err,
        )
    elif "RateLimitError" in err_type or status_code == 429:
        return RateLimitError(
            f"[Anthropic Claude] Rate limit or quota exceeded: {msg}",
            provider="claude",
            status_code=429,
            raw_error=err,
        )
    elif "BadRequestError" in err_type or status_code == 400:
        return BadRequestError(
            f"[Anthropic Claude] Bad request: {msg}",
            provider="claude",
            status_code=400,
            raw_error=err,
        )
    elif "NotFoundError" in err_type or status_code == 404:
        return NotFoundError(
            f"[Anthropic Claude] Model not found: {msg}",
            provider="claude",
            status_code=404,
            raw_error=err,
        )
    elif (
        "APIConnectionError" in err_type
        or "APITimeoutError" in err_type
        or "ConnectionError" in err_type
        or "ConnError" in err_type
        or "connection" in msg.lower()
        or "timeout" in msg.lower()
    ):
        return ConnectionError(
            f"[Anthropic Claude] Could not connect to Anthropic API server: {msg}",
            provider="claude",
            raw_error=err,
        )
    elif (
        "InternalServerError" in err_type
        or "ServerError" in err_type
        or (status_code and status_code >= 500)
    ):
        return ServerError(
            f"[Anthropic Claude] Anthropic server error: {msg}",
            provider="claude",
            status_code=status_code or 500,
            raw_error=err,
        )
    return OpenAIProviderError(
        f"[Anthropic Claude] Request failed: {msg}",
        provider="claude",
        raw_error=err,
    )


class ClaudeProvider(BaseLLMProvider):
    provider_id = "claude"
    display_name = "Anthropic Claude"

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        model = model or DEFAULT_MODEL
        supports_vis = self.supports_vision(model)

        messages = []

        # Inject conversation history
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Build current user message with optional screenshots
        content: list = []
        if screenshots_b64 and supports_vis:
            valid_shots = [s for s in screenshots_b64 if s]
            is_multi = len(valid_shots) > 1
            for i, img_b64 in enumerate(valid_shots, start=1):
                if is_multi:
                    label = f"Screen {i} (Primary):" if i == 1 else f"Screen {i}:"
                    content.append({"type": "text", "text": label})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                })

        content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except OpenAIProviderError:
            raise
        except Exception as e:
            raise translate_anthropic_error(e) from e

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
