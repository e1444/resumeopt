"""Data models shared across the parser pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PostingSummary:
    """Stage 0 output: structured, posting-level global context.

    Passed as shared background context to every Stage 1 extraction call -
    without it, every judgment would only ever see a candidate's own local
    excerpt, with no sense of the posting as a whole.
    """

    role_title: str
    seniority: str
    industry_domain: str
    core_requirements: List[str]
    nice_to_have: List[str]
    summary_paragraph: str


@dataclass(frozen=True)
class ChunkSkillVerdict:
    """Final per-candidate decision after Stage 1 (extraction) + Stage 2
    (category) + Stage 3a (keyword-atomicity gate) + Stage 3b (within-chunk
    redundancy check, only for non-atomic terms).

    `category` is one of `categorization.CATEGORIES` (or `"uncategorized"` if
    Stage 2 never returned a verdict for this term - a failed-batch fail-safe,
    treated as excluded).

    `atomic_keyword` (Stage 3a, `keyword_atomicity.py`) is `None` when the
    atomicity gate was never attempted (the term failed Stage 2). `True`
    means the term is an intrinsically standalone, resume-worthy keyword on
    its own merits and BYPASSES the redundancy check entirely (always kept).
    `False` means the term is a purely descriptive category label with no
    independent identity, and is sent on to Stage 3b.

    `redundant_with` (Stage 3b, `redundancy.py`) is `None` when Stage 3b was
    never attempted (the term failed Stage 2, passed the atomicity gate, or
    was the only surviving non-atomic candidate in its chunk - nothing to be
    redundant with) and an empty list when Stage 3b ran and found no
    redundancy. A non-empty list names the more-specific sibling term(s)
    this candidate is redundant with, and is the reason `included` is
    `False` even though `category == "resume_technical_skill"`.

    `included` is `True` only when `category == categorization.
    INCLUDED_CATEGORY` ("resume_technical_skill") AND the term is either
    atomic (`atomic_keyword is True`) or, if non-atomic, was not marked
    redundant by Stage 3b.

    `chunk` is the ORIGINAL sentence/passage this term was extracted from -
    kept for grounding/evidence and as local context for cache matching
    (`SemanticMatcher(context=...)`).
    """

    raw_term: str
    chunk: str
    category: str
    extraction_reason: str
    category_reason: str
    atomic_keyword: Optional[bool] = None
    atomicity_reason: str = ""
    redundant_with: Optional[List[str]] = None
    redundancy_reason: str = ""
    included: bool = False
