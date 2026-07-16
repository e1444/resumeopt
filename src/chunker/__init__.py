"""Chunker package: text normalization and per-candidate context windowing.

Replaces line/sentence-based "chunking" as a preprocessing concept. See
window.py's module docstring for the full rationale.

- `normalize.py` (`normalize_whitespace`): collapse all whitespace/newlines
  into single spaces across a whole posting, once, up front.
- `window.py` (`locate_quote`, `build_context_window`): locate a candidate's
  evidence quote in the normalized text and extract a small local window of
  surrounding words, used as that candidate's own context for classification
  and matching - instead of a shared, boundary-dependent "chunk".
"""

from __future__ import annotations

from .normalize import normalize_whitespace
from .window import DEFAULT_WINDOW_WORDS, build_context_window, locate_quote

__all__ = [
    "normalize_whitespace",
    "locate_quote",
    "build_context_window",
    "DEFAULT_WINDOW_WORDS",
]
