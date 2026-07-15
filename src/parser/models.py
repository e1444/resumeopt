"""Shared data models and scoring constants for posting parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

MATCH_PRIORITY = {"exact": 3, "alias": 2, "related": 1}
BASE_CONFIDENCE = {"exact": 0.98, "alias": 0.90, "related": 0.75}
BASE_RELEVANCE = {"exact": 5, "alias": 4, "related": 3}


@dataclass(frozen=True)
class SkillRecord:
    """Canonical skill definition from the YAML cache."""

    name: str
    aliases: Tuple[str, ...]
    related: Tuple[str, ...]
