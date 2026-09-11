"""
Unit tests for Genie Model Registry and Model Discovery (Phase 2, Step 5).

Verifies:
- refresh("lmstudio") uses the LM Studio live fetcher
- LM Studio model IDs are converted into valid ModelInfo objects
- LM Studio discovery failure preserves fallback behavior
- Gemini models/foo and foo resolve to the same model
- Existing Claude, OpenAI, Gemini registry behavior remains functional
- Existing Ollama behavior is unchanged
- ModelInfo attribute and dictionary-style access compatibility
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ai.gemini_provider import GeminiProvider
from ai.model_registry import (
    ModelInfo,
    _FALLBACKS,
    _FETCHERS,
    _cache_path,
    best_default,
    cached_models,
    get_model_info,
    get_models,
    model_ids,
    refresh,
)
from ai.ollama_models_registry import is_vision_capable
from ai.ollama_provider import OllamaProvider
from config import cfg


class TestModelRegistry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _patch_models_dir(self):
        return patch("ai.model_registry._models_dir", return_value=self.models_dir)

    # ── 1. ModelInfo Compatibility Tests ─────────────────────────────────────

    def test_model_info_attribute_and_dict_access_compatibility(self):
        """Verify ModelInfo supports both attribute access and dictionary-style indexing."""
        info = ModelInfo(
            id="test-model-1",
            display_name="Test Model 1",
            vision=True,
            context_window=64_000,
            cost_tier="lightweight",
            extra={"vendor": "meta", "special_flag": 42},
        )

        # Attribute access
        self.assertEqual(info.id, "test-model-1")
        self.assertEqual(info.display_name, "Test Model 1")
        self.assertTrue(info.vision)
        self.assertEqual(info.context_window, 64_000)
        self.assertEqual(info.cost_tier, "lightweight")

        # Dictionary-style access
        self.assertEqual(info["id"], "test-model-1")
        self.assertEqual(info["display_name"], "Test Model 1")
        self.assertEqual(info["label"], "Test Model 1")
        self.assertTrue(info["vision"])
        self.assertEqual(info["context_window"], 64_000)
        self.assertEqual(info["cost_tier"], "lightweight")
        self.assertEqual(info["vendor"], "meta")
        self.assertEqual(info["special_flag"], 42)

        # get() with and without default
        self.assertTrue(info.get("vision"))
        self.assertEqual(info.get("vendor"), "meta")
        self.assertEqual(info.get("nonexistent", "fallback_val"), "fallback_val")
        self.assertIsNone(info.get("nonexistent"))

        # __contains__
        self.assertIn("id", info)
        self.assertIn("label", info)
        self.assertIn("vision", info)
        self.assertIn("vendor", info)
        self.assertNotIn("nonexistent", info)

        # KeyError on missing index
        with self.assertRaises(KeyError):
            _ = info["missing_key"]

        # Serialization / Deserialization round-trip
        d = info.to_dict()
        self.assertEqual(d["id"], "test-model-1")
        self.assertEqual(d["vendor"], "meta")

        reconstructed = ModelInfo.from_dict(d)
        self.assertEqual(reconstructed.id, info.id)
        self.assertEqual(reconstructed.display_name, info.display_name)
        self.assertEqual(reconstructed.vision, info.vision)
        self.assertEqual(reconstructed.context_window, info.context_window)
        self.assertEqual(reconstructed.cost_tier, info.cost_tier)
        self.assertEqual(reconstructed.extra.get("vendor"), "meta")

    # ── 2. Gemini Model ID Normalization Tests ───────────────────────────────

    def test_gemini_model_id_normalization_with_and_without_prefix(self):
        """Verify models/gemini-2.5-flash and gemini-2.5-flash resolve to the same model."""
        with self._patch_models_dir():
            # Test with fallback models in registry
            flash_unprefixed = get_model_info("gemini", "gemini-2.5-flash")
            flash_prefixed = get_model_info("gemini", "models/gemini-2.5-flash")

            self.assertEqual(flash_unprefixed.id, "gemini-2.5-flash")
            self.assertEqual(flash_prefixed.id, "gemini-2.5-flash")
            self.assertEqual(flash_unprefixed, flash_prefixed)

            pro_unprefixed = get_model_info("gemini", "gemini-2.5-pro")
            pro_prefixed = get_model_info("gemini", "models/gemini-2.5-pro")

            self.assertEqual(pro_unprefixed.id, "gemini-2.5-pro")
            self.assertEqual(pro_prefixed.id, "gemini-2.5-pro")
            self.assertEqual(pro_unprefixed, pro_prefixed)

    def test_gemini_model_id_normalization_custom_cached_model(self):
        """Verify tolerant lookup for custom Gemini models foo vs models/foo."""
        with self._patch_models_dir():
            cache_file = self.models_dir / "models_gemini.json"
            blob = {
                "fetched_at": 9999999999.0,
                "models": [
                    {
                        "id": "custom-gemini-model",
                        "label": "Custom Gemini",
                        "display_name": "Custom Gemini",
                        "vision": False,
                        "context_window": 500_000,
                        "cost_tier": "standard",
                    }
                ],
            }
            cache_file.write_text(json.dumps(blob), encoding="utf-8")

            res_unprefixed = get_model_info("gemini", "custom-gemini-model")
            res_prefixed = get_model_info("gemini", "models/custom-gemini-model")

            self.assertEqual(res_unprefixed.id, "custom-gemini-model")
            self.assertEqual(res_prefixed.id, "custom-gemini-model")
            self.assertFalse(res_prefixed.vision)

    def test_gemini_provider_supports_vision_with_prefixed_model(self):
        """Verify GeminiProvider.supports_vision handles models/ prefix transparently."""
        provider = GeminiProvider()
        self.assertTrue(provider.supports_vision("gemini-2.5-flash"))
        self.assertTrue(provider.supports_vision("models/gemini-2.5-flash"))

    # ── 3. LM Studio Live Discovery Tests ────────────────────────────────────

    def test_lmstudio_fetcher_registered(self):
        """Verify 'lmstudio' is in _FETCHERS."""
        self.assertIn("lmstudio", _FETCHERS)
        self.assertTrue(callable(_FETCHERS["lmstudio"]))

    def test_refresh_lmstudio_uses_live_fetcher(self):
        """Verify refresh('lmstudio') queries LMStudioProvider.list_models."""
        with self._patch_models_dir():
            mock_list = AsyncMock(
                return_value=["qwen2.5-coder-7b-instruct", "qwen2-vl-7b-instruct", "local-model"]
            )
            with patch("ai.lmstudio_provider.LMStudioProvider.list_models", mock_list):
                results = asyncio.run(refresh("lmstudio"))

                mock_list.assert_called_once()
                self.assertEqual(len(results), 3)

    def test_lmstudio_model_ids_converted_to_valid_model_info(self):
        """Verify returned LM Studio model IDs are properly converted to ModelInfo objects."""
        with self._patch_models_dir():
            mock_list = AsyncMock(
                return_value=[
                    "qwen2.5-coder-7b-instruct",
                    "qwen2-vl-7b-instruct",
                    "local-model",
                ]
            )
            with patch("ai.lmstudio_provider.LMStudioProvider.list_models", mock_list):
                models = asyncio.run(refresh("lmstudio"))

                # Check ModelInfo types and attributes
                self.assertTrue(all(isinstance(m, ModelInfo) for m in models))

                by_id = {m.id: m for m in models}

                # Text-only model -> vision False
                coder = by_id["qwen2.5-coder-7b-instruct"]
                self.assertEqual(coder.id, "qwen2.5-coder-7b-instruct")
                self.assertEqual(coder.display_name, "qwen2.5-coder-7b-instruct")
                self.assertFalse(coder.vision)
                self.assertEqual(coder.context_window, 8_192)
                self.assertEqual(coder.cost_tier, "free")

                # VL model -> vision True
                vl = by_id["qwen2-vl-7b-instruct"]
                self.assertEqual(vl.id, "qwen2-vl-7b-instruct")
                self.assertTrue(vl.vision)

                # Fallback "local-model" -> vision True
                local = by_id["local-model"]
                self.assertEqual(local.id, "local-model")
                self.assertTrue(local.vision)

                # Check on-disk cache persistence
                cached = cached_models("lmstudio")
                self.assertEqual(len(cached), 3)
                self.assertEqual(cached[0].id, "qwen2.5-coder-7b-instruct")

    def test_lmstudio_discovery_failure_preserves_fallback_behavior(self):
        """Verify discovery failure (empty list or exception) preserves fallback models."""
        with self._patch_models_dir():
            # 1. When list_models returns empty list
            mock_empty = AsyncMock(return_value=[])
            with patch("ai.lmstudio_provider.LMStudioProvider.list_models", mock_empty):
                models = asyncio.run(refresh("lmstudio"))
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].id, "local-model")
                self.assertEqual(models[0].display_name, "Loaded Model")
                self.assertTrue(models[0].vision)

            # 2. When list_models raises exception
            mock_error = AsyncMock(side_effect=RuntimeError("Connection refused"))
            with patch("ai.lmstudio_provider.LMStudioProvider.list_models", mock_error):
                models = asyncio.run(refresh("lmstudio"))
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].id, "local-model")

            # 3. get_model_info returns fallback model
            info = get_model_info("lmstudio")
            self.assertEqual(info.id, "local-model")
            self.assertTrue(info.vision)

    # ── 4. Existing Provider Registry & Fallback Tests ────────────────────────

    def test_existing_claude_openai_gemini_fallbacks(self):
        """Verify Claude, OpenAI, and Gemini return their curated fallbacks when cache empty."""
        with self._patch_models_dir():
            claude_models = cached_models("claude")
            self.assertGreater(len(claude_models), 0)
            self.assertTrue(any("sonnet" in m.id for m in claude_models))

            openai_models = cached_models("openai")
            self.assertGreater(len(openai_models), 0)
            self.assertTrue(any("gpt-4o" in m.id for m in openai_models))

            gemini_models = cached_models("gemini")
            self.assertGreater(len(gemini_models), 0)
            self.assertTrue(any("gemini-2.5-flash" in m.id for m in gemini_models))

    def test_existing_ollama_behavior_unchanged(self):
        """Verify Ollama behavior and vision heuristics remain unchanged."""
        with self._patch_models_dir():
            ollama_models = cached_models("ollama")
            self.assertGreater(len(ollama_models), 0)
            self.assertEqual(ollama_models[0].id, "qwen2-vl:7b")

            # Vision heuristic keywords
            self.assertTrue(is_vision_capable("qwen2-vl:7b"))
            self.assertTrue(is_vision_capable("llama3.2-vision:11b"))
            self.assertTrue(is_vision_capable("llava:7b"))
            self.assertFalse(is_vision_capable("llama3.2:3b"))
            self.assertFalse(is_vision_capable("mistral:7b"))
            self.assertFalse(is_vision_capable("qwen2.5-coder:7b"))

            # OllamaProvider model picking
            provider = OllamaProvider()
            with patch.object(
                cfg,
                "get_ollama_model",
                side_effect=lambda kind: "vis-model" if kind == "vision" else "txt-model",
            ):
                self.assertEqual(provider._pick_model(has_screenshots=True), "vis-model")
                self.assertEqual(provider._pick_model(has_screenshots=False), "txt-model")


if __name__ == "__main__":
    unittest.main()
