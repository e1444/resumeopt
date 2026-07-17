"""Deterministic exact/alias string matching against the skill cache.

Free, instant, fully reproducible. Kept as the fast first-tier path before
falling back to `SemanticMatcher`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .base import Matcher, MatchCandidate
from .models import BASE_CONFIDENCE, SkillRecord, normalize_term


class ExactAliasMatcher(Matcher):
    """Deterministic exact/alias string matching against the skill cache."""

    def __init__(self, skills: Sequence[SkillRecord]):
        self.term_lookup: Dict[str, List[Tuple[str, str]]] = {}
        for record in skills:
            self._add_term(record.name, record.name, "exact")
            for alias in record.aliases:
                self._add_term(alias, record.name, "alias")
        self.ordered_terms: List[str] = sorted(self.term_lookup.keys(), key=len, reverse=True)

    def _add_term(self, raw_term: str, canonical_name: str, match_type: str) -> None:
        key = normalize_term(raw_term)
        if not key:
            return
        self.term_lookup.setdefault(key, []).append((canonical_name, match_type))

    def match(self, raw_term: str) -> List[MatchCandidate]:
        key = normalize_term(raw_term)
        entries = self.term_lookup.get(key, [])
        return [
            MatchCandidate(canonical_name=name, match_type=match_type, confidence=BASE_CONFIDENCE[match_type])
            for name, match_type in entries
        ]
