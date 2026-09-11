"""
Live model discovery + caching for Claude, OpenAI, Gemini, Ollama, LM Studio, and Copilot.

Each cloud provider exposes a "list models" endpoint we hit on demand:
  • Anthropic:  GET /v1/models                    (key in x-api-key)
  • OpenAI:     GET /v1/models                    (key in Authorization)
  • Gemini:     GET /v1beta/models?key=...        (key in query)

Cached per-provider to %LOCALAPPDATA%\\Genie\\models\\models_<provider>.json with a
30-day TTL — long enough that you don't refetch constantly, short enough
that new model releases land within a month without manual refresh.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import httpx

from config import cfg


CACHE_TTL_SECONDS = 30 * 24 * 60 * 60   # 30 days


@dataclass
class ModelInfo:
    """Represents an AI model's metadata and capabilities."""
    id: str
    display_name: str
    vision: bool = True
    context_window: int = 128_000
    cost_tier: str = "standard"  # "free" | "lightweight" | "standard" | "premium"
    extra: dict = field(default_factory=dict)

    def __getitem__(self, key: str):
        if key == "id":
            return self.id
        elif key in ("display_name", "label"):
            return self.display_name
        elif key == "vision":
            return self.vision
        elif key == "context_window":
            return self.context_window
        elif key == "cost_tier":
            return self.cost_tier
        elif key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in ("id", "display_name", "label", "vision", "context_window", "cost_tier") or key in self.extra

    def keys(self):
        return ["id", "display_name", "label", "vision", "context_window", "cost_tier"] + list(self.extra.keys())

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "label": self.display_name,
            "display_name": self.display_name,
            "vision": self.vision,
            "context_window": self.context_window,
            "cost_tier": self.cost_tier,
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Union[dict, ModelInfo]) -> ModelInfo:
        if isinstance(d, ModelInfo):
            return d
        mid = d.get("id") or d.get("name") or "unknown"
        display = d.get("display_name") or d.get("label") or d.get("displayName") or mid
        vision = bool(d.get("vision", True))
        context_window = int(d.get("context_window", 128_000))
        cost_tier = str(d.get("cost_tier", "standard"))
        known = {"id", "name", "display_name", "label", "displayName", "vision", "context_window", "cost_tier"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=mid,
            display_name=display,
            vision=vision,
            context_window=context_window,
            cost_tier=cost_tier,
            extra=extra,
        )


# Curated fallback lists — used when the live endpoint is unreachable AND
# the on-disk cache is empty. Reasonable defaults so Genie still works
# offline / on first run before refresh completes.
_FALLBACKS: dict[str, list[dict]] = {
    "claude": [
        {"id": "claude-sonnet-4-6",          "label": "Claude Sonnet 4.6", "display_name": "Claude Sonnet 4.6", "vision": True, "context_window": 200_000, "cost_tier": "standard"},
        {"id": "claude-opus-4-7",            "label": "Claude Opus 4.7",   "display_name": "Claude Opus 4.7",   "vision": True, "context_window": 200_000, "cost_tier": "premium"},
        {"id": "claude-haiku-4-5-20251001",  "label": "Claude Haiku 4.5",  "display_name": "Claude Haiku 4.5",  "vision": True, "context_window": 200_000, "cost_tier": "lightweight"},
    ],
    "openai": [
        {"id": "gpt-4o",        "label": "GPT-4o",       "display_name": "GPT-4o",       "vision": True, "context_window": 128_000, "cost_tier": "standard"},
        {"id": "gpt-4o-mini",   "label": "GPT-4o mini",  "display_name": "GPT-4o mini",  "vision": True, "context_window": 128_000, "cost_tier": "lightweight"},
        {"id": "gpt-4-turbo",   "label": "GPT-4 Turbo",  "display_name": "GPT-4 Turbo",  "vision": True, "context_window": 128_000, "cost_tier": "premium"},
    ],
    "gemini": [
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "display_name": "Gemini 2.5 Flash", "vision": True, "context_window": 1_048_576, "cost_tier": "free"},
        {"id": "gemini-2.5-pro",   "label": "Gemini 2.5 Pro",   "display_name": "Gemini 2.5 Pro",   "vision": True, "context_window": 2_097_152, "cost_tier": "standard"},
        {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash", "display_name": "Gemini 2.0 Flash", "vision": True, "context_window": 1_048_576, "cost_tier": "free"},
    ],
    "ollama": [
        {"id": "qwen2-vl:7b",          "label": "Qwen2-VL 7B",          "display_name": "Qwen2-VL 7B",          "vision": True,  "context_window": 32_768,  "cost_tier": "free"},
        {"id": "llama3.2-vision:11b",  "label": "Llama 3.2 Vision 11B", "display_name": "Llama 3.2 Vision 11B", "vision": True,  "context_window": 128_000, "cost_tier": "free"},
        {"id": "llama3.2:3b",          "label": "Llama 3.2 3B",         "display_name": "Llama 3.2 3B",         "vision": False, "context_window": 128_000, "cost_tier": "free"},
    ],
    "lmstudio": [
        {"id": "local-model", "label": "Loaded Model", "display_name": "Loaded Model", "vision": True, "context_window": 8_192, "cost_tier": "free"},
    ],
    "copilot": [
        {"id": "gpt-4o-mini", "label": "GPT-4o mini", "display_name": "GPT-4o mini", "vision": True, "context_window": 128_000, "cost_tier": "free"},
        {"id": "gpt-4o",      "label": "GPT-4o",      "display_name": "GPT-4o",      "vision": True, "context_window": 128_000, "cost_tier": "standard"},
    ],
}


def _models_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / "Genie" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(provider: str) -> Path:
    new_path = _models_dir() / f"models_{provider}.json"
    if not new_path.exists():
        # Safe migration from legacy %LOCALAPPDATA%\Clicky\models_<provider>.json
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        old_path = Path(base) / "Clicky" / f"models_{provider}.json"
        if old_path.exists():
            try:
                shutil.copy2(old_path, new_path)
            except Exception:
                pass
    return new_path


# ─── Per-provider live fetchers ───────────────────────────────────────────────

async def _fetch_claude() -> list[dict]:
    if not cfg.anthropic_api_key:
        return []
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": cfg.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
        )
    r.raise_for_status()
    data = r.json().get("data", [])
    out = []
    for m in data:
        mid = m.get("id") or m.get("name")
        if not mid:
            continue
        # All current Claude models are vision-capable; future ones likely too.
        cost_tier = "lightweight" if "haiku" in mid else ("premium" if "opus" in mid else "standard")
        out.append({
            "id": mid,
            "label": m.get("display_name") or mid,
            "display_name": m.get("display_name") or mid,
            "vision": True,
            "context_window": 200_000,
            "cost_tier": cost_tier,
        })
    # Newest first (Anthropic returns newest first already, but be defensive)
    out.sort(key=lambda m: m["id"], reverse=True)
    return out


async def _fetch_openai() -> list[dict]:
    if not cfg.openai_api_key:
        return []
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {cfg.openai_api_key}"},
        )
    r.raise_for_status()
    data = r.json().get("data", [])
    out = []
    # Filter to chat-completion-capable models. OpenAI's /v1/models returns
    # everything (embeddings, TTS, image-gen, audio, etc.) so we whitelist by
    # known prefixes. Vision flag is true for the gpt-4o family + o3-vision.
    chat_prefixes = ("gpt-4", "gpt-5", "o1", "o3", "o4", "chatgpt-")
    # Models that match a chat prefix but are NOT chat-completion models —
    # picking one of these makes Genie silently stop responding (e.g.
    # "chatgpt-image-latest" generates images, it can't hold a conversation).
    non_chat_markers = ("image", "audio", "realtime", "tts", "transcribe",
                        "embed", "moderation", "dall", "instruct", "codex")
    vision_prefixes = ("gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5",
                       "o1-", "o3-", "o4-")
    seen = set()
    for m in data:
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        if not mid.startswith(chat_prefixes):
            continue
        if any(marker in mid for marker in non_chat_markers):
            continue
        # Drop fine-tune / preview-snapshot variants like ".../2024-08-06"
        if mid.count("-") >= 4 and any(seg.isdigit() for seg in mid.split("-")):
            continue
        seen.add(mid)
        cost_tier = "lightweight" if "mini" in mid else ("premium" if "turbo" in mid or "o1" in mid or "o3" in mid else "standard")
        out.append({
            "id": mid,
            "label": mid,
            "display_name": mid,
            "vision": mid.startswith(vision_prefixes),
            "context_window": 128_000,
            "cost_tier": cost_tier,
        })
    out.sort(key=lambda m: m["id"])
    return out


async def _fetch_gemini() -> list[dict]:
    if not cfg.google_api_key:
        return []
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": cfg.google_api_key},
        )
    r.raise_for_status()
    data = r.json().get("models", [])
    out = []
    for m in data:
        # Names look like "models/gemini-2.5-flash" — strip the prefix
        full = m.get("name", "")
        mid = full.replace("models/", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue   # skip embedding-only / TTS-only models
        if not mid:
            continue
        is_vision = "vision" in mid or "gemini-1.5" in mid or "gemini-2" in mid or "gemini-3" in mid
        cost_tier = "free" if "flash" in mid else "standard"
        ctx = 2_097_152 if "pro" in mid else 1_048_576
        out.append({
            "id": mid,
            "label": m.get("displayName") or mid,
            "display_name": m.get("displayName") or mid,
            # All Gemini 1.5+ models accept images as input
            "vision": is_vision,
            "context_window": ctx,
            "cost_tier": cost_tier,
        })
    # Sort: newer first (rough heuristic — versions in name)
    out.sort(key=lambda m: m["id"], reverse=True)
    return out


_FETCHERS = {
    "claude":  _fetch_claude,
    "openai":  _fetch_openai,
    "gemini":  _fetch_gemini,
}


# ─── Public API ───────────────────────────────────────────────────────────────

def cached_models(provider: str) -> list[ModelInfo]:
    """Read on-disk cache, falling back to a curated list if missing.

    Returns ModelInfo instances which support both attribute access
    (m.vision, m.cost_tier) and backward-compatible dict access (m["id"], m.get("vision")).
    """
    p = _cache_path(provider)
    raw: list = []
    if p.exists():
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            ms = blob.get("models", [])
            if ms:
                raw = ms
        except Exception:
            pass
    if not raw:
        raw = list(_FALLBACKS.get(provider, []))
    return [ModelInfo.from_dict(m) for m in raw]


def get_models(provider: str) -> list[ModelInfo]:
    """Modern typed lookup for all cached models under a provider."""
    return cached_models(provider)


def get_model_info(provider: str, model_id: str | None = None) -> ModelInfo:
    """Retrieve ModelInfo for a specific model under a provider,
    or the provider's default model if model_id is omitted."""
    models = cached_models(provider)
    if model_id:
        for m in models:
            if m.id == model_id:
                return m
    # Fallback to best default or first available model
    def_id = best_default(provider)
    if def_id:
        for m in models:
            if m.id == def_id:
                return m
    if models:
        return models[0]
    # Ultimate synthetic fallback if provider is entirely unknown
    mid = model_id or "default"
    return ModelInfo(id=mid, display_name=mid, vision=True, context_window=128_000, cost_tier="standard")


def cache_is_stale(provider: str, ttl: int = CACHE_TTL_SECONDS) -> bool:
    p = _cache_path(provider)
    if not p.exists():
        return True
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        return (time.time() - float(blob.get("fetched_at", 0))) > ttl
    except Exception:
        return True


async def refresh(provider: str) -> list[ModelInfo]:
    """Fetch live + write to cache. Returns the new model list (raises on error)."""
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        raise ValueError(f"No live model fetcher for provider '{provider}'")
    models = await fetcher()
    if not models:
        # No key → no models. Don't overwrite cache with empty list.
        return cached_models(provider)
    serialized = [m.to_dict() if isinstance(m, ModelInfo) else m for m in models]
    blob = {"fetched_at": time.time(), "models": serialized}
    _cache_path(provider).write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return [ModelInfo.from_dict(m) for m in models]


async def refresh_all_stale() -> dict[str, int]:
    """Refresh every provider whose cache is stale. Returns counts per provider."""
    results = {}
    for provider in _FETCHERS:
        if cache_is_stale(provider):
            try:
                ms = await refresh(provider)
                results[provider] = len(ms)
            except Exception as e:
                results[provider] = -1   # signals failure
    return results


def model_ids(provider: str) -> list[str]:
    return [m.id for m in cached_models(provider)]


def best_default(provider: str) -> Optional[str]:
    """Pick a sensible default model from the cache — vision-capable first."""
    models = cached_models(provider)
    for m in models:
        if m.vision:
            return m.id
    return models[0].id if models else None


# ─── CLI: `python -m ai.model_registry [show|refresh] [provider]` ─────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) >= 2 else "show"
    target = sys.argv[2] if len(sys.argv) >= 3 else None

    if cmd == "show":
        for prov in (target,) if target else _FETCHERS:
            stale = "stale" if cache_is_stale(prov) else "fresh"
            print(f"\n[{prov}] {stale}")
            for m in cached_models(prov):
                v = "👁" if m.get("vision") else "  "
                print(f"  {v} {m['id']}")
    elif cmd == "refresh":
        async def _run():
            for prov in (target,) if target else _FETCHERS:
                try:
                    ms = await refresh(prov)
                    print(f"[{prov}] refreshed {len(ms)} models")
                except Exception as e:
                    print(f"[{prov}] FAILED: {e}")
        asyncio.run(_run())
    else:
        print("Usage: python -m ai.model_registry [show|refresh] [claude|openai|gemini]")
