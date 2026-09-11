"""
Tutor-layer helpers: active window detection, privacy masking, "next" detection,
and locate-query classification. Kept out of companion_manager.py so the
orchestrator stays readable.
"""

from __future__ import annotations

import ctypes
import re
from ctypes import wintypes


# ── Active-window title (for per-app context memory) ─────────────────────────

def active_window_title() -> str:
    info = active_window_info()
    return info.get("title", "")


def active_window_info() -> dict:
    """Return title, physical rect (l, t, r, b), and physical center (cx, cy) of active window."""
    try:
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return {"title": "", "rect": None, "center": None}
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value or ""

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]
        rect = RECT()
        u.GetWindowRect(hwnd, ctypes.byref(rect))
        cx = int((rect.left + rect.right) // 2)
        cy = int((rect.top + rect.bottom) // 2)
        return {
            "title": title,
            "rect": (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)),
            "center": (cx, cy),
        }
    except Exception:
        return {"title": "", "rect": None, "center": None}


def cursor_position() -> tuple[int, int]:
    """Return the physical screen coordinates of the cursor."""
    try:
        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    except Exception:
        return 0, 0


def find_monitor_for_point(point: tuple[int, int] | None, screens: list) -> int:
    """Find the ScreenShot index (1-based) containing the physical point (x, y)."""
    if not screens:
        return 1
    if not point:
        return screens[0].index
    px, py = point
    for s in screens:
        if (s.physical_left <= px < s.physical_left + s.physical_width and
            s.physical_top <= py < s.physical_top + s.physical_height):
            return s.index
    return screens[0].index


def get_element_at_point(x: int, y: int) -> str:
    """Safely query UIA for the control at physical coordinates (x, y)."""
    try:
        import uiautomation as auto
        ctrl = auto.ControlFromPoint(x, y)
        if ctrl is None:
            return ""
        name = (ctrl.Name or "").strip()
        ctrl_type = (ctrl.ControlTypeName or "").replace("Control", "").strip()
        help_text = (getattr(ctrl, "HelpText", "") or "").strip()

        if not name and not help_text:
            return ""

        parts = []
        if ctrl_type:
            parts.append(ctrl_type)
        if name:
            parts.append(f'"{name}"')
        if help_text and help_text.lower() != name.lower():
            parts.append(f"({help_text})")
        return " ".join(parts).strip()
    except Exception:
        return ""


def app_key(title: str) -> str:
    """Reduce a noisy window title to a stable per-app key.

    Examples:
      "Premiere Pro - project.prproj"  -> "Premiere Pro"
      "YouTube — Google Chrome"        -> "Google Chrome"
      "VS Code — main.py"              -> "VS Code"
    """
    if not title:
        return "desktop"
    # Take the right-most app name chunk (usually after last "-" or "—")
    parts = re.split(r"\s[-—–|]\s", title)
    return (parts[-1] if parts else title).strip()[:48] or "desktop"


# ── Locate-query classification ──────────────────────────────────────────────

LOCATE_RE = re.compile(
    r"\b(where\s+(is|do|can|should)|how\s+do\s+i\s+(click|find|open|access|use|get\s+to)|"
    r"point\s+(at|to)|show\s+me\s+(the|where)|click\s+(the|on)|find\s+the|"
    r"locate\s+the|highlight\s+the)\b",
    re.IGNORECASE,
)

MULTISTEP_RE = re.compile(
    r"\b(how\s+(do\s+i|to)\s+(export|install|configure|set\s*up|setup|publish|"
    r"deploy|enable|disable|build|launch|download|upload|record))\b",
    re.IGNORECASE,
)

NEXT_RE = re.compile(r"^\s*(next|continue|go\s*on|keep\s*going|what'?s?\s*next)[\s.!?]*$",
                     re.IGNORECASE)

STOP_RE = re.compile(r"^\s*(stop|quit|cancel|never\s*mind|nevermind)[\s.!?]*$",
                     re.IGNORECASE)

DEICTIC_RE = re.compile(
    r"\b(what\s+(is|does|are)\s+(this|that|it|here)|"
    r"what('s|\s+is)\s+(this|that|the)\s+(button|icon|symbol|thing|tool|item|option|setting|switch)|"
    r"what\s+does\s+this\s+(button|icon|symbol|tool|switch|option)?\s*(do|mean)?|"
    r"explain\s+(this|that)\s+(button|icon|element|symbol|error|message|window|dialog|text|panel)|"
    r"what\s+am\s+i\s+looking\s+at)\b",
    re.IGNORECASE,
)


def is_deictic(transcript: str) -> bool:
    """Return True if the question is a deictic reference ('What is this?', 'What does this button do?')."""
    return bool(DEICTIC_RE.search(transcript or ""))


def is_locate(q: str) -> bool:
    return LOCATE_RE.search(q) is not None


def is_multistep(q: str) -> bool:
    return MULTISTEP_RE.search(q) is not None


def is_next(q: str) -> bool:
    return NEXT_RE.match(q or "") is not None


def is_stop(q: str) -> bool:
    return STOP_RE.match(q or "") is not None


# ── New voice classifiers ────────────────────────────────────────────────────

REPEAT_RE = re.compile(
    r"^\s*(repeat|say\s+(it|that)\s*again|say\s+again|once\s+more|"
    r"what\s+(did\s+you|d['’]you)\s+say)[\s.!?]*$",
    re.IGNORECASE,
)

JOURNAL_TODAY_RE = re.compile(
    r"\bwhat\s+(did|have)\s+i\s+(learn|learned|asked)\s+(today|so\s+far)\b",
    re.IGNORECASE,
)

JOURNAL_WEEK_RE = re.compile(
    r"\bwhat\s+(did|have)\s+i\s+(learn|learned)\s+(this\s+week|recently|"
    r"in\s+the\s+past\s+week)\b",
    re.IGNORECASE,
)

QUIZ_REVIEW_RE = re.compile(
    r"\b(quiz\s+me|review\s+me|test\s+me)(\s+on\s+(what\s+i\s+learned|my\s+notes))?\b",
    re.IGNORECASE,
)

# ── Identity questions ────────────────────────────────────────────────────────
# OpenAI / Claude refuse to identify people in images even when the answer is
# trivially in their training data. So when the user asks "who is X" / "who was X"
# / "how old is X" — we strip the screenshot and answer from text + web search.
#
# Crucially, questions about what is on screen ("What is this error?", "What does
# this button do?", "What is Docker?", "Tell me about this diagram", "Who is on my screen")
# MUST retain screenshots.
IDENTITY_RE = re.compile(
    r"^\s*"
    r"(who\s+(is|are|was|were)|"
    r"how\s+old\s+is)"
    r"\s+"
    # Negative lookahead: reject screen references
    r"(?!"
    r"this|that|it|my\s+screen|on\s+(my\s+)?screen|going\s+on|"
    r"happening|the\s+screen|here|in\s+this|in\s+that|i\s|i\b)"
    r".+",
    re.IGNORECASE,
)

SCREEN_CONTEXT_TOKENS_RE = re.compile(
    r"\b(screen|this|that|error|button|icon|code|image|picture|diagram|"
    r"chart|window|dialog|line|symbol|syntax|menu|cursor|here)\b",
    re.IGNORECASE,
)


def is_identity_question(q: str) -> bool:
    """Detects 'who is <person>'-style queries that should NOT include a screenshot.
    Returns False whenever visual/screen context is indicated."""
    if not q:
        return False
    stripped = q.strip()
    if SCREEN_CONTEXT_TOKENS_RE.search(stripped):
        return False
    return IDENTITY_RE.match(stripped) is not None


def is_repeat(q: str) -> bool:
    return REPEAT_RE.match(q or "") is not None


def is_journal_today(q: str) -> bool:
    return JOURNAL_TODAY_RE.search(q or "") is not None


def is_journal_week(q: str) -> bool:
    return JOURNAL_WEEK_RE.search(q or "") is not None


def is_quiz_review(q: str) -> bool:
    return QUIZ_REVIEW_RE.search(q or "") is not None


# ── Privacy guard — block sensitive windows from being screenshotted ──────────

PRIVACY_BLOCKLIST = (
    r"\b(password|credential|secret|keepass|bitwarden|1password|lastpass|"
    r"authenticator|banking|sign\s*in|login|\.env)\b"
)
_PRIVACY_RE = re.compile(PRIVACY_BLOCKLIST, re.IGNORECASE)


def is_sensitive_window(title: str) -> bool:
    return bool(title) and _PRIVACY_RE.search(title) is not None


def find_sensitive_windows() -> list[dict]:
    """Enumerate visible top-level windows and return list of sensitive window dicts:
    [{'hwnd': hwnd, 'title': title, 'rect': (left, top, right, bottom)}, ...]
    """
    sensitive = []
    try:
        u = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        def _enum_proc(hwnd, lparam):
            try:
                if not u.IsWindowVisible(hwnd):
                    return True
                if u.IsIconic(hwnd):
                    return True  # Minimized windows do not appear in desktop screenshots
                n = u.GetWindowTextLengthW(hwnd)
                if n <= 0:
                    return True
                buf = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(hwnd, buf, n + 1)
                title = buf.value or ""
                if is_sensitive_window(title):
                    rect = RECT()
                    if u.GetWindowRect(hwnd, ctypes.byref(rect)):
                        l, t, r, b = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
                        sensitive.append({"hwnd": hwnd, "title": title, "rect": (l, t, r, b)})
                    else:
                        sensitive.append({"hwnd": hwnd, "title": title, "rect": None})
            except Exception:
                pass
            return True

        cb = WNDENUMPROC(_enum_proc)
        u.EnumWindows(cb, 0)
    except Exception:
        pass
    return sensitive


def get_sensitive_monitor_indices(screens: list) -> set[int]:
    """Return the set of monitor indices (1-based) containing sensitive windows.
    Fails closed (returns all monitor indices) if the active window is sensitive
    or if any sensitive window's monitor attribution is uncertain."""
    if not screens:
        return set()

    all_indices = {s.index for s in screens}

    # 1. Fail-safe: if the active window itself is sensitive, fail closed for all monitors
    active_title = active_window_title()
    if is_sensitive_window(active_title):
        return all_indices

    # 2. Enumerate all visible windows on the desktop
    sensitive_wins = find_sensitive_windows()
    if not sensitive_wins:
        return set()

    sensitive_monitors = set()
    for win in sensitive_wins:
        rect = win.get("rect")
        if not rect or len(rect) != 4:
            # Uncertain rect -> fail closed immediately
            return all_indices

        wl, wt, wr, wb = rect
        if wr <= wl or wb <= wt:
            # Invalid dimensions -> fail closed
            return all_indices

        hit_monitors = set()
        for s in screens:
            mon_l = getattr(s, "physical_left", 0)
            mon_t = getattr(s, "physical_top", 0)
            mon_r = mon_l + getattr(s, "physical_width", 0)
            mon_b = mon_t + getattr(s, "physical_height", 0)

            # Check physical bounding box overlap
            if min(wr, mon_r) > max(wl, mon_l) and min(wb, mon_b) > max(wt, mon_t):
                hit_monitors.add(s.index)

        if not hit_monitors:
            # Window is visible according to OS, but doesn't intersect any known monitor bounds
            # Uncertain attribution -> fail closed
            return all_indices

        sensitive_monitors.update(hit_monitors)

    return sensitive_monitors
