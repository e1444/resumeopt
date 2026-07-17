"""Shared matcher interface and result type.

Three independent, separately-testable strategies implement `Matcher`
(exact_alias.py, semantic.py) or stand alone (grounding.py) - see
matcher/__init__.py for the full package layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, List


@dataclass(frozen=True)
class MatchCandidate:
    """One candidate canonical match produced by a Matcher."""

    canonical_name: str
    match_type: str  # "exact" | "alias" | "semantic"
    confidence: float
    similarity: Optional[float] = None


class Matcher(ABC):
    """Shared interface for turning a raw extracted term into cache matches."""

    @abstractmethod
    def match(self, raw_term: str) -> List[MatchCandidate]:
        """Return zero or more candidate canonical matches for a raw term."""

    def match_batch(self, raw_terms: Sequence[str]) -> List[List[MatchCandidate]]:
        """Match many raw terms at once. Default loops match(); override to batch."""

        return [self.match(raw_term) for raw_term in raw_terms]
