"""Skill matcher package: turning an extracted raw skill term into cache-backed matches.

Three independent, separately-testable strategies:
- `exact_alias.py` (`ExactAliasMatcher`): deterministic lookup against a
  skill's exact/alias terms. Free, instant, fully reproducible.
  Kept as the fast first-tier path.
- `semantic.py` (`SemanticMatcher`): embedding-based cosine-similarity
  matching, so a skill doesn't need every phrasing variant (ipynb/jupyter,
  BSc/undergraduate degree, GLM/GBM style abbreviations) hand-enumerated as a
  cache alias to be matched. Still deterministic given a fixed embedding
  model and a fixed cache.
- `grounding.py` (`LLMGroundingMatcher`): LLM-based grounding confirmation for
  borderline matches. Not a `Matcher` in the match-producing sense; it
  validates whether an already-selected canonical match is actually
  supported by the posting text.

Shared types (`base.py`: `MatchCandidate`, `Matcher`), the skill-cache record
shape and scoring constants (`models.py`: `SkillRecord`, `normalize_term`,
`MATCH_PRIORITY`, `BASE_CONFIDENCE`, `BASE_RELEVANCE`), and the persistent
embedding cache (`embedding_cache.py`: `EmbeddingCache`) round out the
package. This is intentionally decoupled from `src/parser/`: matching only
operates on in-memory `SkillRecord` sequences passed in by the caller and has
no knowledge of YAML cache loading, chunking, or extraction - those remain
`src/parser/` responsibilities.
"""

from __future__ import annotations

from .base import MatchCandidate, Matcher
from .embedding_cache import EmbeddingCache
from .exact_alias import ExactAliasMatcher
from .grounding import LLMGroundingMatcher
from .models import BASE_CONFIDENCE, BASE_RELEVANCE, MATCH_PRIORITY, SkillRecord, normalize_term
from .semantic import SemanticMatcher

__all__ = [
    "MatchCandidate",
    "Matcher",
    "EmbeddingCache",
    "ExactAliasMatcher",
    "LLMGroundingMatcher",
    "SemanticMatcher",
    "SkillRecord",
    "normalize_term",
    "MATCH_PRIORITY",
    "BASE_CONFIDENCE",
    "BASE_RELEVANCE",
]
