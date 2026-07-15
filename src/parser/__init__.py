"""Posting parser package.

Layout:
- models.py        shared data models and scoring constants
- base.py           PostingParser ABC + DeterministicPostingParser (cache-backed matching)
- candidate_utils.py  mechanical candidate shape/dedup helpers
- voting.py           self-consistency voting for repeated extraction samples
- selection.py        select_skills / validate_selected_skills (shared final-selection stage)
- orchestra_single_shot.py  OrchestraSingleShotParser (default: deterministic-only
  chunking, per-chunk self-contained extraction+cache-match, self-consistency voted)
- single_shot.py       SingleShotPostingParser (one LLM call per whole posting;
  only safe for already-atomic input)
- factory.py           parse_posting() convenience factory

Note: earlier "multishot" (chunk-by-chunk with LLM-based re-chunking) and
"deterministic-parser-as-primary-strategy" variants were retired after
benchmarking showed `OrchestraSingleShotParser` matches or beats them while
being simpler and cheaper. `DeterministicPostingParser` remains as shared
cache-loading/matching infrastructure and as the offline (no-LLM) fallback.
"""

from __future__ import annotations

from .base import DeterministicPostingParser, PostingParser
from .factory import parse_posting
from .models import SkillRecord
from .orchestra_single_shot import OrchestraSingleShotParser
from .selection import select_skills, validate_selected_skills
from .single_shot import SingleShotPostingParser

__all__ = [
    "PostingParser",
    "DeterministicPostingParser",
    "SkillRecord",
    "SingleShotPostingParser",
    "OrchestraSingleShotParser",
    "select_skills",
    "validate_selected_skills",
    "parse_posting",
]
