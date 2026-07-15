"""Single-shot parser: runs the orchestra parser's extraction and cache-matching
logic exactly once per whole posting, skipping chunk splitting entirely.

Only safe for already-atomic input (a single sentence/bullet); benchmarked to
collapse badly (F1 ~0.2) when given a full multi-bullet posting in one call,
since the model returns coarse per-bullet phrases instead of decomposing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from llm import LLMProvider

from .orchestra_single_shot import OrchestraSingleShotParser


class SingleShotPostingParser(OrchestraSingleShotParser):
    """Extracts and cache-matches the entire posting in one LLM call."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        skills_cache_path: Path = Path("data/skills.yaml"),
        max_workers: int = 8,
    ):
        super().__init__(
            llm_provider=llm_provider,
            skills_cache_path=skills_cache_path,
            max_workers=max_workers,
            num_votes=1,
        )

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        chunk = posting_text.strip() or posting_text
        extraction_candidates = self._extract_terms_llm_batch(chunk)

        records: List[Dict[str, Any]] = []
        record = self._build_record_from_candidates(chunk, extraction_candidates)
        if record is not None:
            records.append(record)
        return records
