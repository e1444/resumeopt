"""Shared data models and scoring constants for skill matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

MATCH_PRIORITY = {"exact": 3, "alias": 2, "semantic": 0}
BASE_CONFIDENCE = {"exact": 0.98, "alias": 0.90, "semantic": 0.75}
BASE_RELEVANCE = {"exact": 5, "alias": 4, "semantic": 2}


def normalize_term(value: str) -> str:
    """Lowercase and collapse whitespace for stable dict-key comparisons."""

    return " ".join(value.lower().strip().split())


@dataclass(frozen=True)
class SkillRecord:
    """Canonical skill definition from the YAML cache."""

    name: str
    aliases: Tuple[str, ...]
    always_include: bool = False
