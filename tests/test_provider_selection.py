"""
Unit tests for Provider Selection, Provider Factory, and Model Routing in Genie.
Verifies factory instantiation, LM Studio support, model propagation through
CompanionManager, and Ollama automatic vision/text routing preservation.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from ai.base_provider import BaseLLMProvider, Message
from ai.claude_provider import ClaudeProvider
from ai.gemini_provider import GeminiProvider
from ai.github_copilot_provider import GitHubCopilotProvider
from ai.lmstudio_provider import LMStudioProvider
from ai.ollama_provider import OllamaProvider
from ai.openai_provider import OpenAIProvider
from ai.provider_factory import SUPPORTED_PROVIDERS, get_provider
from companion_manager import CompanionManager
from config import cfg
from ui.panel import CompanionPanel, PROVIDER_LABELS


class TestProviderSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure QApplication instance exists for Qt widgets
        cls.app = QApplication.instance() or QApplication(["test"])

    def test_factory_supported_providers_list(self):
        """Verify all six core providers are in SUPPORTED_PROVIDERS."""
        expected = {"claude", "openai", "gemini", "copilot", "lmstudio", "ollama"}
        self.assertEqual(set(SUPPORTED_PROVIDERS), expected)

    @patch("config.cfg.anthropic_api_key", "fake_claude_key")
    def test_factory_creates_claude_provider(self):
        """Verify factory returns ClaudeProvider for 'claude'."""
        provider = get_provider("claude")
        self.assertIsInstance(provider, ClaudeProvider)
        self.assertEqual(provider.provider_id, "claude")

    @patch("config.cfg.openai_api_key", "fake_openai_key")
    def test_factory_creates_openai_provider(self):
        """Verify factory returns OpenAIProvider for 'openai'."""
        provider = get_provider("openai")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.provider_id, "openai")

    def test_factory_creates_gemini_provider(self):
        """Verify factory returns GeminiProvider for 'gemini'."""
        provider = get_provider("gemini")
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.provider_id, "gemini")

    def test_factory_creates_lmstudio_provider(self):
        """Verify factory returns LMStudioProvider for 'lmstudio'."""
        provider = get_provider("lmstudio")
        self.assertIsInstance(provider, LMStudioProvider)
        self.assertEqual(provider.provider_id, "lmstudio")

    @patch("ai.provider_factory._ensure_ollama_running")
    def test_factory_creates_ollama_provider(self, mock_ensure):
        """Verify factory triggers daemon check and returns OllamaProvider for 'ollama'."""
        provider = get_provider("ollama")
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.provider_id, "ollama")
        mock_ensure.assert_called_once()

    @patch("ai.github_copilot_provider.load_github_token", return_value="fake_token_123")
    def test_factory_creates_copilot_provider(self, mock_load):
        """Verify factory returns GitHubCopilotProvider when token is available."""
        provider = get_provider("copilot")
        self.assertIsInstance(provider, GitHubCopilotProvider)
        self.assertEqual(provider.provider_id, "copilot")

    def test_factory_unknown_provider(self):
        """Verify factory raises ValueError for invalid/unknown provider IDs."""
        with self.assertRaises(ValueError) as ctx:
            get_provider("invalid_ai_service")
        self.assertIn("invalid_ai_service", str(ctx.exception))

    def test_factory_default_provider_from_config(self):
        """Verify get_provider() with no arguments resolves to cfg.llm_provider()."""
        with patch.object(cfg, "llm_provider", return_value="gemini"):
            provider = get_provider()
            self.assertIsInstance(provider, GeminiProvider)

    def test_lmstudio_availability_and_labels(self):
        """Verify LM Studio is registered in available_llm_providers and panel labels."""
        self.assertIn("lmstudio", cfg.available_llm_providers())
        self.assertIn("lmstudio", PROVIDER_LABELS)
        self.assertEqual(PROVIDER_LABELS["lmstudio"], "LM Studio")

    @patch("companion_manager.AmbientListener")
    @patch("threading.Thread")
    def test_model_propagation_through_companion_manager(self, mock_thread, mock_listener):
        """Verify model setting and auto/default/empty normalization in CompanionManager."""
        manager = CompanionManager()

        # Explicit model name propagates
        manager.set_model("gpt-4o-mini")
        self.assertEqual(manager._current_model, "gpt-4o-mini")

        # "auto" resets to None to enable automatic model routing
        manager.set_model("auto")
        self.assertIsNone(manager._current_model)

        # "default" resets to None
        manager.set_model("explicit-model")
        manager.set_model("default")
        self.assertIsNone(manager._current_model)

        # Empty string resets to None
        manager.set_model("explicit-model")
        manager.set_model("")
        self.assertIsNone(manager._current_model)

    @patch("companion_manager.AmbientListener")
    @patch("threading.Thread")
    def test_companion_manager_lazy_provider_init(self, mock_thread, mock_listener):
        """Verify CompanionManager lazily uses get_provider and resets on switch."""
        manager = CompanionManager()
        manager._submit = lambda coro: coro.close()
        self.assertIsNone(manager._llm)

        with patch("companion_manager.get_provider") as mock_get:
            fake_provider = MagicMock(spec=BaseLLMProvider)
            mock_get.return_value = fake_provider

            # First access initializes provider
            llm = manager._get_llm()
            self.assertEqual(llm, fake_provider)
            mock_get.assert_called_once()

            # Subsequent access returns cached instance
            llm2 = manager._get_llm()
            self.assertEqual(llm2, fake_provider)
            self.assertEqual(mock_get.call_count, 1)

            # Switching provider resets cache
            manager.set_active_provider("openai")
            self.assertIsNone(manager._llm)
            self.assertIsNone(manager._current_model)

    def test_ollama_automatic_routing_preserved(self):
        """Verify Ollama automatically selects vision vs text model based on screenshots."""
        provider = OllamaProvider()

        with patch.object(cfg, "get_ollama_model", side_effect=lambda kind: "llama3.2-vision" if kind == "vision" else "llama3.2:3b"):
            # Direct _pick_model checks
            self.assertEqual(provider._pick_model(has_screenshots=True), "llama3.2-vision")
            self.assertEqual(provider._pick_model(has_screenshots=False), "llama3.2:3b")

            # Capabilities check with "auto" should evaluate the vision model
            with patch("ai.ollama_provider.is_vision_capable", return_value=True):
                cap = provider.get_capabilities(model="auto")
                self.assertEqual(cap.id, "llama3.2-vision")
                self.assertTrue(cap.vision)

                cap_none = provider.get_capabilities(model=None)
                self.assertEqual(cap_none.id, "llama3.2-vision")
                self.assertTrue(cap_none.vision)

    def test_panel_model_dropdown_for_lmstudio(self):
        """Verify panel populates LM Studio models properly without errors."""
        panel = CompanionPanel()
        panel._set_models_for("lmstudio")

        combo = panel._model_combo
        self.assertGreater(combo.count(), 0)
        # Should contain at least one model entry
        first_data = combo.itemData(0)
        self.assertIsNotNone(first_data)

    def test_panel_model_dropdown_for_ollama(self):
        """Verify panel populates 'Auto (vision/text)' as first option for Ollama."""
        panel = CompanionPanel()

        with patch.object(cfg, "get_ollama_model", side_effect=lambda kind: "llama3.2-vision" if kind == "vision" else "llama3.2:3b"):
            panel._set_models_for("ollama")

            combo = panel._model_combo
            self.assertGreater(combo.count(), 0)
            # First option must be Auto with userData="auto"
            self.assertIn("Auto", combo.itemText(0))
            self.assertEqual(combo.itemData(0), "auto")


if __name__ == "__main__":
    unittest.main()
