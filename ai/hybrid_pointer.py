"""
Hybrid pointer — Clicky pointing accuracy upgrade.

The original grid-locator was inaccurate because it asked the LLM to guess
pixel coordinates from a numbered grid overlay. Vision models trained on
natural images don't have pixel-precise spatial reasoning, so they'd often
pick a neighboring cell or off-by-one row, sending Clicky's blue cursor to
the wrong button.

This module uses a three-tier resolver, each tier much more accurate than the next:

  Tier 1  Windows UI Automation (UIA)     ~5 ms     pixel-perfect
          Walks the OS accessibility tree, finds elements by name/role.
          Works in every standard UI: Chrome/Edge/Firefox, VS Code, IDEs,
          Office, Electron apps, native Win32, WinUI/UWP, Settings, etc.

  Tier 2  Offline OCR (RapidOCR/ONNX)     ~300 ms   text-perfect
          For canvas apps where UIA gives no tree: Figma, Photoshop, games,
          older Java/Swing UIs. Runs entirely on CPU, no API calls.

  Tier 3  Vision LLM grid fallback        ~1-3 s    best effort
          Only used when UIA + OCR both whiff. Delegates to the existing
          element_locator.py — kept as a safety net.

Public API:
    find_target(query: str) -> Optional[Target]

The caller (companion_manager) treats Target as opaque — it has .center_xy
in LOGICAL screen pixels, ready to feed directly into the overlay pointer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

log = logging.getLogger("clicky.pointer")


@dataclass
class Target:
    """A pointing target with exact pixel coordinates in LOGICAL screen space."""
    x: int                # center X
    y: int                # center Y
    bbox: Tuple[int, int, int, int]   # (left, top, right, bottom)
    label: str            # what we matched (button text, control type, etc.)
    source: str           # "uia" | "ocr" | "vision"
    confidence: float     # 0.0–1.0

    @property
    def center_xy(self) -> Tuple[int, int]:
        return (self.x, self.y)


# ──────────────────────────────────────────────────────────────────────────────
#  TIER 1 — Windows UI Automation
# ──────────────────────────────────────────────────────────────────────────────

_INTERACTIVE_TYPES = {
    # Most reliable click targets in UIA
    "Button", "Hyperlink", "MenuItem", "TabItem", "TreeItem", "ListItem",
    "RadioButton", "CheckBox", "ComboBox", "Edit", "Text",
    "Custom",  # often used by Electron / web apps
}


_CONVERSATIONAL_PREFIXES = (
    "where do i click to ",
    "where do i click on the ",
    "where do i click on ",
    "where do i click the ",
    "where do i click ",
    "where can i find the ",
    "where can i find ",
    "where should i click on ",
    "where should i click ",
    "where is the ",
    "where's the ",
    "where is ",
    "how do i click on the ",
    "how do i click on ",
    "how do i click the ",
    "how do i click ",
    "how do i find the ",
    "how do i find ",
    "how do i open the ",
    "how do i open ",
    "how do i access the ",
    "how do i access ",
    "how do i use the ",
    "how do i use ",
    "how do i get to the ",
    "how do i get to ",
    "show me the ",
    "show me where is the ",
    "show me where the ",
    "show me where is ",
    "show me where ",
    "show me ",
    "point at the ",
    "point at ",
    "point to the ",
    "point to ",
    "find the ",
    "find ",
    "locate the ",
    "locate ",
    "click the ",
    "click on the ",
    "click on ",
    "click ",
    "press the ",
    "press ",
    "highlight the ",
    "highlight ",
)

_FILLER_WORDS = {
    "button", "icon", "tab", "menu", "item", "option", "link",
    "field", "this", "that", "here", "please",
}


def _extract_target(query: str) -> str:
    """Extract candidate target entity from a natural-language conversational query."""
    q_norm = (query or "").lower().strip().rstrip("?.!")
    for prefix in _CONVERSATIONAL_PREFIXES:
        if q_norm.startswith(prefix):
            q_norm = q_norm[len(prefix):].strip()
            break

    words = q_norm.split()
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    while words and words[-1] in _FILLER_WORDS:
        words = words[:-1]

    return " ".join(words).strip()


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _score_match(query: str, element_name: str, element_type: str = "") -> float:
    """Fuzzy match score 0..1. Boosts:
      - exact match (raw or extracted entity)
      - substring match (element name containing extracted entity)
      - interactive control types
      - whole-word match
    """
    q = _normalize(query)
    name = _normalize(element_name)
    if not q or not name:
        return 0.0

    target = _normalize(_extract_target(query))

    score = 0.0
    if q == name or (target and target == name):
        score = 1.0
    elif target and (target in name or name in target):
        ratio = len(target) / max(len(name), len(target), 1)
        score = max(0.85, 0.7 + 0.25 * ratio)
    elif q in name:
        score = 0.85
    else:
        # Word overlap - evaluate against extracted target first, then raw query
        q_words = set(q.split())
        target_words = set(target.split()) if target else set()
        n_words = set(name.split())
        if target_words and n_words:
            t_overlap = len(target_words & n_words) / max(len(target_words), 1)
            score = t_overlap * 0.75
        elif q_words and n_words:
            overlap = len(q_words & n_words) / max(len(q_words), 1)
            score = overlap * 0.7

    if element_type in _INTERACTIVE_TYPES and score > 0.0:
        score = min(1.0, score + 0.1)
    return score


def _find_via_uia(query: str, min_score: float = 0.5) -> Optional[Target]:
    """Walk the UIA tree for the foreground window + descendants, find best match."""
    try:
        import uiautomation as auto
    except ImportError:
        log.warning("uiautomation not installed — Tier 1 (UIA) disabled")
        return None

    try:
        # Get the focused (foreground) window — pointing is almost always
        # for the active app, and walking the whole desktop is slow.
        root = auto.GetForegroundControl()
        if root is None:
            root = auto.GetRootControl()
    except Exception as e:
        log.debug("UIA root lookup failed: %s", e)
        return None

    best: Optional[Tuple[float, "auto.Control"]] = None
    # Bounded walk — UIA trees can be huge in Chrome/Electron
    queue: List[Tuple["auto.Control", int]] = [(root, 0)]
    visited = 0
    MAX_NODES = 3500
    MAX_DEPTH = 40

    while queue and visited < MAX_NODES:
        node, depth = queue.pop(0)
        visited += 1
        try:
            name = node.Name or ""
            ctrl_type = node.ControlTypeName or ""
            rect = node.BoundingRectangle  # mss/uia returns Rect
        except Exception:
            continue
        # Skip off-screen / zero-size
        if not rect or rect.width() <= 0 or rect.height() <= 0:
            pass
        else:
            score = _score_match(query, name, ctrl_type)
            # Also try AutomationId and HelpText as backup match sources
            if score < 0.85:
                try:
                    aid = getattr(node, "AutomationId", "") or ""
                    if aid:
                        score = max(score, _score_match(query, aid, ctrl_type) * 0.8)
                except Exception:
                    pass
            if score >= min_score and (best is None or score > best[0]):
                best = (score, node)

        if depth < MAX_DEPTH:
            try:
                for child in node.GetChildren():
                    queue.append((child, depth + 1))
            except Exception:
                continue

    if not best:
        log.debug("UIA: no match for %r (scanned %d nodes)", query, visited)
        return None

    score, node = best
    r = node.BoundingRectangle
    cx = int((r.left + r.right) // 2)
    cy = int((r.top + r.bottom) // 2)
    log.info("UIA hit: %r -> %s [%s] @ (%d,%d) score=%.2f",
             query, node.Name, node.ControlTypeName, cx, cy, score)
    return Target(
        x=cx, y=cy,
        bbox=(int(r.left), int(r.top), int(r.right), int(r.bottom)),
        label=node.Name or node.ControlTypeName,
        source="uia",
        confidence=score,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  TIER 2 — Offline OCR (RapidOCR, ONNX)
# ──────────────────────────────────────────────────────────────────────────────

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except ImportError:
            log.warning("rapidocr-onnxruntime not installed — Tier 2 (OCR) disabled")
            return None
        except Exception as e:
            log.warning("OCR engine init failed: %s", e)
            return None
    return _ocr_engine


def _find_via_ocr(query: str, screenshot_path: Optional[str] = None,
                  pil_image=None, screenshot=None, min_score: float = 0.5) -> Optional[Target]:
    """Run OCR on the specified screen, fuzzy-match the query against detected text."""
    ocr = _get_ocr()
    if ocr is None:
        return None

    try:
        if pil_image is None and screenshot_path is None:
            # Capture target monitor (or primary) at full resolution
            import mss
            with mss.mss() as sct:
                mon_idx = getattr(screenshot, "index", 1) if screenshot else 1
                if 1 <= mon_idx < len(sct.monitors):
                    mon = sct.monitors[mon_idx]
                else:
                    mon = sct.monitors[1]
                raw = sct.grab(mon)
                from PIL import Image
                pil_image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        import numpy as np
        if pil_image is not None:
            img_arr = np.array(pil_image)
        else:
            img_arr = screenshot_path  # RapidOCR accepts paths directly

        result, _ = ocr(img_arr)
        if not result:
            log.debug("OCR returned no detections")
            return None

        best: Optional[Tuple[float, dict]] = None
        for det in result:
            # RapidOCR format: [bbox(4 corners), text, score]
            try:
                box, text, conf = det
            except Exception:
                continue
            if conf < 0.4:
                continue
            score = _score_match(query, text, "")
            if score >= min_score and (best is None or score > best[0]):
                # Compute bbox from 4 corners
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                best = (score, {
                    "text": text,
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "conf": conf,
                })

        if not best:
            log.debug("OCR: no text matched %r", query)
            return None

        score, hit = best
        l, t, r, b = hit["bbox"]
        cx, cy = (l + r) // 2, (t + b) // 2
        log.info("OCR hit: %r -> %r @ (%d,%d) score=%.2f",
                 query, hit["text"], cx, cy, score)
        return Target(
            x=cx, y=cy, bbox=hit["bbox"],
            label=hit["text"], source="ocr",
            confidence=score * hit["conf"],
        )
    except Exception as e:
        log.warning("OCR tier failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  TIER 3 — Vision LLM grid fallback (delegated to existing element_locator)
# ──────────────────────────────────────────────────────────────────────────────

def _find_via_vision(query: str, screenshot, llm_provider) -> Optional[Target]:
    """Last-resort: use the grid-locator path (element_locator.py)."""
    try:
        from ai.element_locator import locate_element  # v1 module
    except ImportError:
        log.warning("element_locator not available — Tier 3 (vision) disabled")
        return None

    try:
        coord = locate_element(query, screenshot, llm_provider)
        if not coord:
            return None
        x, y = coord
        return Target(
            x=x, y=y, bbox=(x - 20, y - 20, x + 20, y + 20),
            label=query, source="vision",
            confidence=0.5,   # vision is least trustworthy
        )
    except Exception as e:
        log.warning("Vision tier failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def find_target(
    query: str,
    *,
    screenshot=None,
    pil_image=None,
    llm_provider=None,
    skip_uia: bool = False,
    skip_ocr: bool = False,
    skip_vision: bool = False,
) -> Optional[Target]:
    """Resolve a natural-language pointing query into pixel coordinates.

    Tries UIA → OCR → Vision in order. Returns the first hit with confidence
    >= the tier's threshold. Returns None if all three tiers whiff.

    Args:
        query:        what the user asked Clicky to point at (e.g. "save button",
                      "send icon", "login text field").
        screenshot:   ScreenShot object (from screen.capture) — used by vision tier.
        pil_image:    PIL.Image of the full screen — used by OCR tier. Optional;
                      OCR will capture its own if not provided.
        llm_provider: BaseLLMProvider instance — needed only for vision fallback.
        skip_*:       Force-skip a tier (for testing or perf-sensitive paths).

    Returns:
        Target with .center_xy in logical screen pixels, or None.
    """
    if not query or not query.strip():
        return None

    # Tier 1: UIA — fast, free, often perfect
    if not skip_uia:
        t = _find_via_uia(query)
        if t is not None and t.confidence >= 0.5:
            return t

    # Tier 2: OCR — text-based fallback for canvas apps
    if not skip_ocr:
        t = _find_via_ocr(query, pil_image=pil_image, screenshot=screenshot)
        if t is not None and t.confidence >= 0.5:
            return t

    # Tier 3: vision LLM grid — last resort
    if not skip_vision and screenshot is not None and llm_provider is not None:
        t = _find_via_vision(query, screenshot, llm_provider)
        if t is not None:
            return t

    log.info("All pointer tiers failed for query: %r", query)
    return None


__all__ = ["Target", "find_target"]
