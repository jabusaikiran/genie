from typing import AsyncIterator, List

from openai import AsyncOpenAI

from ai.base_provider import BaseLLMProvider, Message
from ai.openai_compatible_provider import (
    build_openai_messages,
    translate_sdk_error,
)
from config import cfg

DEFAULT_MODEL = "gpt-4o"
MAX_TOKENS = 1024


class OpenAIProvider(BaseLLMProvider):
    provider_id = "openai"
    display_name = "OpenAI"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            key = self._api_key or cfg.openai_api_key
            base_url = self._base_url or cfg.openai_base_url
            if not key and not base_url:
                from ai.openai_compatible_provider import AuthenticationError
                raise AuthenticationError(
                    "[OpenAI] API key is not configured. Set OPENAI_API_KEY in your environment or Tray -> Settings -> API Keys.",
                    provider="openai",
                    status_code=401,
                )
            kwargs = {}
            if key:
                kwargs["api_key"] = key
            elif base_url:
                kwargs["api_key"] = "dummy-key"
            if base_url:
                kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        model = model or cfg.openai_default_model or DEFAULT_MODEL
        supports_vis = self.supports_vision(model)

        messages = build_openai_messages(
            system_prompt=system_prompt,
            history=history,
            user_text=user_text,
            screenshots_b64=screenshots_b64,
            supports_vision=supports_vis,
        )

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except OpenAIProviderError:
            raise
        except Exception as e:
            raise translate_sdk_error(self.display_name, e) from e

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False
