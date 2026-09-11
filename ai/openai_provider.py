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

    def __init__(self):
        # OPENAI_BASE_URL turns this into a generic OpenAI-compatible client:
        # DeepSeek (https://api.deepseek.com), Alibaba DashScope/Qwen,
        # SiliconFlow, OpenRouter, etc. Set OPENAI_DEFAULT_MODEL to match.
        kwargs = {"api_key": cfg.openai_api_key}
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        self._client = AsyncOpenAI(**kwargs)

    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
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
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            raise translate_sdk_error(self.display_name, e) from e

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
