from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional
from dataclasses import dataclass

from ai.model_registry import ModelInfo, get_model_info


@dataclass
class Message:
    role: str   # "user" or "assistant"
    content: str


class BaseLLMProvider(ABC):
    """All LLM providers implement this interface."""

    provider_id: str = "base"
    display_name: str = "Base Provider"

    @abstractmethod
    async def stream_response(
        self,
        user_text: str,
        screenshots_b64: List[str],
        history: List[Message],
        system_prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Yields text chunks as they stream in."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the provider is reachable."""
        ...

    def get_capabilities(self, model: str | None = None) -> ModelInfo:
        """Return model metadata and capabilities (vision support, context window, cost tier)
        for the given model, or the provider's default model if model is None."""
        return get_model_info(self.provider_id, model)

    def supports_vision(self, model: str | None = None) -> bool:
        """Convenience method: returns True if the specified (or default) model accepts image/screenshot inputs."""
        return self.get_capabilities(model).vision
