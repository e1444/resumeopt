"""Chunker package: text normalization, sentence/bullet-level chunking, and
per-candidate context windowing.

- `normalize.py` (`normalize_whitespace`): collapse all whitespace/newlines
  into single spaces across a whole posting, once, up front.
- `window.py` (`locate_quote`, `build_context_window`): locate a candidate's
  evidence quote in the normalized text and extract a small local window of
  surrounding words.
- `sentence_chunking.py` (`split_into_sentence_chunks`): deterministic
  regex-based sentence/bullet splitter - the fallback chunker.
- `llm_chunking.py` (`split_into_sentence_chunks_via_llm`): LLM-based
  chunker - the DEFAULT, higher-accuracy chunking mechanism used by the
  parser pipeline, with automatic grounding-checked fallback to the
  deterministic splitter above.
"""

from __future__ import annotations

from .llm_chunking import split_into_sentence_chunks_via_llm
from .normalize import normalize_whitespace
from .sentence_chunking import split_into_sentence_chunks
from .window import DEFAULT_WINDOW_WORDS, build_context_window, locate_quote

__all__ = [
    "normalize_whitespace",
    "locate_quote",
    "build_context_window",
    "DEFAULT_WINDOW_WORDS",
    "split_into_sentence_chunks",
    "split_into_sentence_chunks_via_llm",
]
