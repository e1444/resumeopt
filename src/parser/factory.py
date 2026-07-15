"""Factory for selecting a posting parser implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from llm import LLMProvider

from .base import DeterministicPostingParser, PostingParser
from .orchestra_single_shot import OrchestraSingleShotParser
from .single_shot import SingleShotPostingParser


def parse_posting(
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    llm_provider: Optional[LLMProvider] = None,
    use_llm: bool = False,
    llm_parser_mode: str = "orchestra_single_shot",
    max_workers: int = 8,
    num_votes: int = 3,
) -> List[Dict[str, Any]]:
    """Parse a job posting with deterministic default behavior.

    llm_parser_mode selects the LLM-backed implementation when use_llm=True:
    "orchestra_single_shot" (default; deterministic-only chunking with an
    independent, self-contained, self-consistency-voted extraction+cache-match
    call per chunk, run concurrently), or "single_shot" (one extraction call
    for the whole posting - benchmarked to fail badly on multi-bullet
    postings; only safe for already-atomic input).

    max_workers caps how many chunk-level LLM calls run concurrently, since
    each chunk's extraction is independent. num_votes controls how many
    independent extraction samples are taken per chunk for self-consistency
    voting (orchestra_single_shot only); set to 1 to disable voting.
    """

    parser: PostingParser
    if use_llm and llm_provider is not None:
        normalized_mode = llm_parser_mode.replace("-", "_").lower().strip()
        if normalized_mode in ("single_shot", "singleshot"):
            parser = SingleShotPostingParser(
                llm_provider=llm_provider, skills_cache_path=skills_cache_path, max_workers=max_workers
            )
        else:
            parser = OrchestraSingleShotParser(
                llm_provider=llm_provider,
                skills_cache_path=skills_cache_path,
                max_workers=max_workers,
                num_votes=num_votes,
            )
    else:
        parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)

    return parser.parse(posting_text)

