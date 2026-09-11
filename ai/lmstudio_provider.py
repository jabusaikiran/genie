"""
LM Studio provider — local OpenAI-compatible endpoint.

Talks directly to LM Studio's local OpenAI-compatible server (default: http://localhost:1234/v1).
No API key required.
"""
from typing import AsyncIterator, List

import httpx

from ai.base_provider import Message
from ai.openai_compatible_provider import (
    OpenAICompatibleProvider,
    ConnectionError,
    NotFoundError,
)
from config import cfg


class LMStudioProvider(OpenAICompatibleProvider):
    provider_id = "lmstudio"
    display_name = "LM Studio"

    def __init__(self):
        super().__init__(
            base_url=cfg.lmstudio_host,
            api_key=None,
            default_model=cfg.lmstudio_model or "local-model",
            timeout=120.0,
            max_tokens=1024,
        )
        # Retain self._base for backward compatibility with any direct attribute readers
        self._base = self._base_url
        self._model = self._default_model

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        chosen = model or self._default_model or "local-model"
        try:
            async for chunk in super().stream_response(
                user_text=user_text,
                screenshots_b64=screenshots_b64,
                history=history,
                system_prompt=system_prompt,
                model=chosen,
            ):
                yield chunk
        except ConnectionError as e:
            raise RuntimeError(
                f"Can't reach LM Studio at {self._base_url}. Is the local server running? "
                "(LM Studio → Developer tab → Start Server)"
            ) from e
        except NotFoundError as e:
            raise RuntimeError(
                f"LM Studio server not reachable at {self._base_url}. "
                "Open LM Studio → Developer tab → Start Server, and make sure a model is loaded."
            ) from e

    async def list_models(self) -> List[str]:
        """Return model ids LM Studio currently reports via /v1/models."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._base_url}/models")
                data = r.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
