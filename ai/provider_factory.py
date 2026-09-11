"""
Centralized AI Provider Factory for Genie.

Provides a unified factory function `get_provider(provider_id: str | None = None)`
to instantiate supported LLM providers lazily without circular dependencies.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ai.base_provider import BaseLLMProvider
from config import cfg

log = logging.getLogger("genie.ai.factory")

SUPPORTED_PROVIDERS = ("claude", "openai", "gemini", "copilot", "lmstudio", "ollama")


def _ensure_ollama_running():
    """Start Ollama if it isn't already running. Waits up to 8 s for it to be ready."""
    import subprocess
    import urllib.request

    url = "http://localhost:11434/api/tags"
    for _ in range(2):
        try:
            urllib.request.urlopen(url, timeout=2)
            return  # already up
        except Exception:
            pass

    already_running = False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        already_running = "ollama.exe" in out.lower()
    except Exception:
        pass

    if not already_running:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except FileNotFoundError:
            return  # ollama not installed, provider will fail gracefully

    # Wait up to 8 s for the server to come up
    for _ in range(16):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            pass


def get_provider(provider_id: Optional[str] = None, **kwargs) -> BaseLLMProvider:
    """Instantiate and return the requested LLM provider.

    Args:
        provider_id: Identifier of the provider ("claude", "openai", "gemini",
                     "copilot", "lmstudio", "ollama"). Defaults to cfg.llm_provider().
        **kwargs: Optional provider-specific constructor parameters.

    Returns:
        An instance of BaseLLMProvider.

    Raises:
        ValueError: If provider_id is not in SUPPORTED_PROVIDERS.
    """
    pid = (provider_id or cfg.llm_provider()).strip().lower()

    if pid == "claude":
        from ai.claude_provider import ClaudeProvider
        return ClaudeProvider()

    if pid == "openai":
        from ai.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if pid == "gemini":
        from ai.gemini_provider import GeminiProvider
        return GeminiProvider(**kwargs) if kwargs else GeminiProvider()

    if pid == "copilot":
        from ai.github_copilot_provider import GitHubCopilotProvider
        return GitHubCopilotProvider()

    if pid == "lmstudio":
        from ai.lmstudio_provider import LMStudioProvider
        return LMStudioProvider(**kwargs) if kwargs else LMStudioProvider()

    if pid == "ollama":
        _ensure_ollama_running()
        from ai.ollama_provider import OllamaProvider
        return OllamaProvider()

    raise ValueError(
        f"Unknown or unsupported AI provider: '{provider_id}'. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )
