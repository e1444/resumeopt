"""Deterministic (no LLM) sentence-level chunking - splits a posting into
sentence/bullet-level chunks for the parser pipeline.

Matches this project's "different decomposition method, by sentence"
experiment (see `tests/parser/reasoning_sentence_chunk_benchmark.py`, which
validated this against `tests/evals/sample_big_section_sentence_cases.yaml`).

Deliberately normalizes whitespace FIRST (`normalize_whitespace`) before
splitting - a stray mid-word line break (e.g. "Ja\\nvaScript", common in
PDF-pasted postings) is collapsed into a single space FIRST, so a
sentence-level split afterward never sees the broken line-boundary artifact
at all - only the (much rarer, individually tolerable) mid-word space it
leaves behind.

The sentence-boundary regex is intentionally simple (period/exclamation/
question mark followed by whitespace and an uppercase letter or digit) - it
is not an exhaustive NLP-grade sentence tokenizer (no abbreviation
dictionary beyond what the boundary condition itself naturally avoids, e.g.
"e.g., mathematics" is safe because a lowercase letter follows the comma,
not a capital). This is used as the fallback chunker when LLM-based chunking
(`llm_chunking.split_into_sentence_chunks_via_llm`) fails or returns nothing
grounded - see that module for the higher-accuracy default.
"""

from __future__ import annotations

import re
from typing import List

from .normalize import normalize_whitespace

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentence_chunks(posting_text: str) -> List[str]:
    """Normalize whitespace, then split into sentence-level chunks.

    Returns an empty list for empty/whitespace-only input. Chunks are
    stripped and empty results are dropped.
    """

    normalized = normalize_whitespace(posting_text)
    if not normalized:
        return []

    raw_chunks = _SENTENCE_BOUNDARY_PATTERN.split(normalized)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]
