import json
from typing import AsyncIterator, List

import httpx

from ai.base_provider import BaseLLMProvider, Message, ModelInfo
from ai.ollama_models_registry import is_vision_capable
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


def format_ollama_error(
    status_code: int,
    response_text: str = "",
    model: str = "",
) -> OpenAIProviderError:
    """Translate Ollama HTTP status code and response payload into a typed provider error."""
    detail = ""
    if response_text:
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                detail = parsed.get("error", "")
            elif isinstance(parsed, str):
                detail = parsed
        except Exception:
            detail = response_text.strip()[:300]

    suffix = f" Details: {detail}" if detail else ""

    if status_code == 404:
        return NotFoundError(
            f"[Ollama] Model '{model}' is not installed locally. "
            f"Run `ollama pull {model}` or pick another model from Tray → Ollama.{suffix}",
            provider="ollama",
            status_code=404,
            raw_error=response_text,
        )
    elif status_code == 429:
        return RateLimitError(
            f"[Ollama] Request rate limit exceeded.{suffix}",
            provider="ollama",
            status_code=429,
            raw_error=response_text,
        )
    elif status_code in (401, 403):
        return AuthenticationError(
            f"[Ollama] Authentication failed (HTTP {status_code}).{suffix}",
            provider="ollama",
            status_code=status_code,
            raw_error=response_text,
        )
    elif status_code == 400:
        return BadRequestError(
            f"[Ollama] Bad request for model '{model}'.{suffix}",
            provider="ollama",
            status_code=400,
            raw_error=response_text,
        )
    elif status_code >= 500:
        return ServerError(
            f"[Ollama] Server error ({status_code}) from local daemon.{suffix}",
            provider="ollama",
            status_code=status_code,
            raw_error=response_text,
        )
    return OpenAIProviderError(
        f"[Ollama] Request failed with HTTP {status_code}.{suffix}",
        provider="ollama",
        status_code=status_code,
        raw_error=response_text,
    )


class OllamaProvider(BaseLLMProvider):
    provider_id = "ollama"
    display_name = "Ollama"

    def __init__(self):
        self._base = cfg.ollama_host.rstrip("/")
        # Kept for backward compat with old code paths reading self._model
        self._model = cfg.ollama_model

    def _pick_model(self, has_screenshots: bool) -> str:
        return cfg.get_ollama_model("vision" if has_screenshots else "text")

    def get_capabilities(self, model: str | None = None) -> ModelInfo:
        effective = None if model in ("auto", "default", "") else model
        chosen = effective or self._pick_model(has_screenshots=True)
        is_vis = is_vision_capable(chosen)
        return ModelInfo(
            id=chosen,
            display_name=chosen,
            vision=is_vis,
            context_window=128_000 if "llama" in chosen.lower() else 32_768,
            cost_tier="free",
        )

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        # Resolution order:
        #   1. explicit `model=` arg (panel override)
        #   2. cfg vision/text slot based on attachment kind
        if model and model not in ("auto", "default"):
            chosen = model
        else:
            chosen = self._pick_model(bool(screenshots_b64))

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Ollama passes images as base64 strings inside the message
        user_msg: dict = {"role": "user", "content": user_text}
        if screenshots_b64 and self.supports_vision(chosen):
            user_msg["images"] = screenshots_b64
        messages.append(user_msg)

        payload = {
            "model": chosen,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": 1024},
        }

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._base}/api/chat",
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        err_body = await response.aread()
                        err_text = err_body.decode("utf-8", errors="replace")
                        raise format_ollama_error(response.status_code, err_text, model=chosen)

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            except OpenAIProviderError:
                raise
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"[Ollama] Could not connect to Ollama daemon at {self._base}. Is the service running? "
                    "Start Ollama or run `ollama serve`.",
                    provider="ollama",
                    raw_error=e,
                ) from e
            except httpx.TimeoutException as e:
                raise ConnectionError(
                    f"[Ollama] Request timed out after 120s to {self._base}.",
                    provider="ollama",
                    raw_error=e,
                ) from e
            except Exception as e:
                raise OpenAIProviderError(
                    f"[Ollama] Streaming request failed: {e}",
                    provider="ollama",
                    raw_error=e,
                ) from e

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._base}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """Return all installed model names (flat list)."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._base}/api/tags")
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def list_models_classified(self) -> dict[str, list[str]]:
        """Installed models split into {'vision': [...], 'text': [...]}.

        Heuristic-based — see ollama_models_registry.is_vision_capable().
        """
        names = await self.list_models()
        out: dict[str, list[str]] = {"vision": [], "text": []}
        for n in names:
            if is_vision_capable(n):
                out["vision"].append(n)
            else:
                out["text"].append(n)
        out["vision"].sort()
        out["text"].sort()
        return out
