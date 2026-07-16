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
    max_workers: int = 24,
    num_votes: int = 3,
    use_semantic_matching: bool = True,
    embedding_cache_path: Optional[Path] = Path("build/cache/skill_embeddings_cache.json"),
    classifier_votes: int = 1,
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
    voting (orchestra_single_shot only); default 3. Since every LLM call
    here is I/O-bound, replicated votes run genuinely concurrently as long
    as max_workers is large enough to hold them all in flight at once -
    empirically, num_votes=3 with max_workers=32 completed a real posting
    FASTER than num_votes=1 with max_workers=8, so voting was restored to
    its previous default once max_workers was raised accordingly instead of
    treating the two as an inherent cost trade-off.

    use_semantic_matching enables the embedding-based SemanticMatcher as a
    second matching tier after exact/alias/related lookup fails (falls back
    to exact/alias-only automatically if llm_provider doesn't support
    embeddings, e.g. Anthropic/Ollama today). embedding_cache_path persists
    cache reference-text embeddings across runs so a stable skills cache only
    pays the embedding cost once; pass None to disable persistent caching.

    classifier_votes controls optional self-consistency voting within each of
    the 4 parallel extraction classifiers (degree_context, domain_vs_technical,
    soft_skill, genericity); benchmarked n=1 vs n=3 (with the original 3
    classifiers) and found no measurable difference, so n=1 is the default
    (see src/parser/parallel_extraction.py).
    """

    parser: PostingParser
    if use_llm and llm_provider is not None:
        normalized_mode = llm_parser_mode.replace("-", "_").lower().strip()
        if normalized_mode in ("single_shot", "singleshot"):
            parser = SingleShotPostingParser(
                llm_provider=llm_provider,
                skills_cache_path=skills_cache_path,
                max_workers=max_workers,
                use_semantic_matching=use_semantic_matching,
                embedding_cache_path=embedding_cache_path,
                classifier_votes=classifier_votes,
            )
        else:
            parser = OrchestraSingleShotParser(
                llm_provider=llm_provider,
                skills_cache_path=skills_cache_path,
                max_workers=max_workers,
                num_votes=num_votes,
                use_semantic_matching=use_semantic_matching,
                embedding_cache_path=embedding_cache_path,
                classifier_votes=classifier_votes,
            )
    else:
        parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)

    return parser.parse(posting_text)

