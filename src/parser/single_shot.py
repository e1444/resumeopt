"""Single-shot parser: same extraction as OrchestraSingleShotParser but with
self-consistency voting (num_votes) forced to 1.

Historically this class's distinction was "skip chunk splitting, one call for
the whole posting" vs. OrchestraSingleShotParser's per-chunk calls - and was
benchmarked to collapse badly (F1 ~0.2) on multi-bullet postings because the
old single-prompt extraction returned coarse per-bullet phrases instead of
decomposing them. Since chunking was removed 2026-07-15 (see
orchestra_single_shot.py's module docstring), OrchestraSingleShotParser
itself now always processes the whole posting in one decompose+classify
pass too - so the two classes only differ in num_votes now (this one forces
1; the default forces 3). Kept as a distinct, named configuration for
call-count-sensitive callers rather than removed, since `parse_posting(...,
num_votes=1)` is an equally valid way to get the same behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from llm import LLMProvider
from chunker import normalize_whitespace

from .orchestra_single_shot import OrchestraSingleShotParser


class SingleShotPostingParser(OrchestraSingleShotParser):
    """Extracts and cache-matches the entire posting in one LLM call."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        skills_cache_path: Path = Path("data/skills.yaml"),
        max_workers: int = 8,
        use_semantic_matching: bool = True,
        embedding_cache_path: Optional[Path] = Path("build/cache/skill_embeddings_cache.json"),
        classifier_votes: int = 1,
    ):
        super().__init__(
            llm_provider=llm_provider,
            skills_cache_path=skills_cache_path,
            max_workers=max_workers,
            use_semantic_matching=use_semantic_matching,
            embedding_cache_path=embedding_cache_path,
            classifier_votes=classifier_votes,
            num_votes=1,
        )

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        normalized_text = normalize_whitespace(posting_text)
        if not normalized_text:
            return []
        extraction_candidates, debug = self._extract_terms_llm_batch(normalized_text)

        records: List[Dict[str, Any]] = []
        record = self._build_record_from_candidates(normalized_text, extraction_candidates, [debug])
        if record is not None:
            records.append(record)
        return records
