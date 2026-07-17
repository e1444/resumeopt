"""Factory tying the parser pipeline together into a parser-record-shaped
output consumed by `src/main.py`'s `run_pipeline`
(`matched_skills`/`missing_skills`/`missing_skills_discarded`/`posting_line`/
`extraction_debug_samples`).

`use_llm=False` returns `DeterministicPostingParser`'s cache-only matching
(no LLM calls at all - a fast, free, low-recall baseline). `use_llm=True`
(the default, production path) runs the full pipeline: `chunker.
split_into_sentence_chunks_via_llm` (or the deterministic regex fallback) ->
`summary.generate_posting_summary` (Stage 0) -> `pipeline.
run_parser_pipeline` (Stage 1 extraction, Stage 2 categorization, Stage 3a/3b
atomicity+redundancy) -> grounding check (`chunker.locate_quote`) -> cache
matching (`ExactAliasMatcher` then `SemanticMatcher` fallback) -> one
parser-record-shaped dict.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from chunker import locate_quote, normalize_whitespace
from llm import DEFAULT_MAX_CONCURRENCY, LLMProvider
from matcher import BASE_CONFIDENCE, EmbeddingCache, MatchCandidate, SemanticMatcher

from .base import DeterministicPostingParser
from .pipeline import run_parser_pipeline
from .summary import format_summary_block, generate_posting_summary

# Every included verdict comes from the SAME single "resume_technical_skill"
# category (no finer per-term relevance tiering in this design) - a flat max
# relevance_score for all of them.
_INCLUDED_RELEVANCE_SCORE = 5


def _finalize_match(match: Dict[str, Any]) -> Dict[str, Any]:
    match_type = match["match_type"]
    base_confidence = match.get("base_confidence", BASE_CONFIDENCE.get(match_type, 0.5))
    return {
        "raw_term": match["raw_term"],
        "canonical_name": match["canonical_name"],
        "match_type": match_type,
        "confidence": round(min(1.0, base_confidence), 2),
        "relevance_score": _INCLUDED_RELEVANCE_SCORE,
        "evidence": match["evidence"],
    }


def parse_posting(
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    use_llm: bool = False,
    summary_llm_provider: Optional[LLMProvider] = None,
    reasoning_llm_provider: Optional[LLMProvider] = None,
    use_semantic_matching: bool = True,
    embedding_cache_path: Optional[Path] = Path("build/cache/skill_embeddings_cache.json"),
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    use_llm_chunking: bool = True,
    enable_redundancy_check: bool = True,
) -> List[Dict[str, Any]]:
    """Parse a job posting into cache-matched skills.

    `use_llm=False` (default) returns deterministic cache-only matching
    (`DeterministicPostingParser`), no LLM calls, no `missing_skills`/
    `missing_skills_discarded` (callers should treat those keys as absent/
    empty for this path).

    `use_llm=True` requires both `summary_llm_provider` (Stage 0
    posting summary - typically a stronger/judge-tier model) and
    `reasoning_llm_provider` (chunking, extraction, categorization, and
    Stage 3 atomicity/redundancy - a reasoning-tier model; validated with
    `gpt-5-mini`). Returns a single-element list of one parser-record-shaped
    dict: `matched_skills`/`missing_skills`/`missing_skills_discarded`/
    `posting_line`/`extraction_debug_samples`.
    """

    if not use_llm:
        parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)
        return parser.parse(posting_text)

    if summary_llm_provider is None or reasoning_llm_provider is None:
        raise ValueError("summary_llm_provider and reasoning_llm_provider are both required when use_llm=True")

    normalized_text = normalize_whitespace(posting_text)
    if not normalized_text:
        return []

    posting_summary = asyncio.run(generate_posting_summary(summary_llm_provider, normalized_text))
    summary_block = format_summary_block(posting_summary)

    verdicts = asyncio.run(
        run_parser_pipeline(
            reasoning_llm_provider,
            normalized_text,
            summary_block=summary_block,
            max_concurrency=max_concurrency,
            enable_redundancy_check=enable_redundancy_check,
            use_llm_chunking=use_llm_chunking,
        )
    )

    # Grounding check: discard any candidate whose own raw_term can't actually
    # be located in the posting text (exact/case-insensitive/fuzzy - see
    # chunker.locate_quote) - a factual check catching hallucinated/
    # paraphrased-beyond-recognition terms, not a topical judgment call.
    ungrounded_discarded: List[Dict[str, str]] = []
    grounded_verdicts = {}
    for normalized_term, verdict in verdicts.items():
        if locate_quote(normalized_text, verdict.raw_term) is None:
            ungrounded_discarded.append({"raw_term": verdict.raw_term, "chunk": verdict.chunk})
            continue
        grounded_verdicts[normalized_term] = verdict

    parser_base = DeterministicPostingParser(skills_cache_path=skills_cache_path)
    semantic_matcher: Optional[SemanticMatcher] = None
    if use_semantic_matching:
        try:
            embedding_cache = EmbeddingCache(embedding_cache_path) if embedding_cache_path is not None else None
            semantic_matcher = SemanticMatcher(
                parser_base._skills, reasoning_llm_provider, embedding_cache=embedding_cache
            )
        except NotImplementedError:
            # Provider doesn't support embeddings (e.g. Anthropic, Ollama
            # today) - fall back to exact/alias-only matching.
            semantic_matcher = None

    grouped: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    seen_missing: Set[str] = set()
    discarded: List[Dict[str, Any]] = []

    for verdict in grounded_verdicts.values():
        if not verdict.included:
            if verdict.redundant_with:
                reason = f"redundant_with={verdict.redundant_with}:{verdict.redundancy_reason}"
            elif verdict.atomic_keyword is False:
                reason = f"non_atomic_and_not_redundant:{verdict.atomicity_reason}"
            else:
                reason = f"category={verdict.category}:{verdict.category_reason}"
            discarded.append(
                {
                    "raw_term": verdict.raw_term,
                    "reason": reason,
                    "include_for_resume_skills": False,
                    "include_for_cache_candidate": False,
                    "evidence_quote": verdict.chunk,
                }
            )
            continue

        matches: List[MatchCandidate] = parser_base._exact_alias_matcher.match(verdict.raw_term)
        if not matches and semantic_matcher is not None:
            matches = semantic_matcher.match(verdict.raw_term, context=verdict.chunk)

        if not matches:
            if verdict.raw_term not in seen_missing:
                missing.append(verdict.raw_term)
                seen_missing.add(verdict.raw_term)
            continue

        for match_candidate in matches:
            canonical_name = match_candidate.canonical_name
            existing = grouped.get(canonical_name)
            # No finer relevance tiering to break ties on (see
            # _INCLUDED_RELEVANCE_SCORE) - prefer the higher-confidence match.
            if existing is not None and match_candidate.confidence <= existing.get("base_confidence", 0.0):
                continue

            grouped[canonical_name] = {
                "raw_term": verdict.raw_term,
                "canonical_name": canonical_name,
                "match_type": match_candidate.match_type,
                "base_confidence": match_candidate.confidence,
                "evidence": verdict.chunk,
            }

    matched_skills = [_finalize_match(match) for match in grouped.values()]
    matched_skills.sort(key=lambda item: (-item["relevance_score"], -item["confidence"], item["canonical_name"]))

    debug = {
        "chunks": list({verdict.chunk for verdict in verdicts.values()}),
        "ungrounded_discarded": ungrounded_discarded,
        "chunk_verdicts": {
            verdict.raw_term: {
                "chunk": verdict.chunk,
                "category": verdict.category,
                "extraction_reason": verdict.extraction_reason,
                "category_reason": verdict.category_reason,
                "atomic_keyword": verdict.atomic_keyword,
                "atomicity_reason": verdict.atomicity_reason,
                "redundant_with": verdict.redundant_with,
                "redundancy_reason": verdict.redundancy_reason,
                "included": verdict.included,
            }
            for verdict in verdicts.values()
        },
        "posting_summary": {
            "role_title": posting_summary.role_title,
            "seniority": posting_summary.seniority,
            "industry_domain": posting_summary.industry_domain,
            "core_requirements": posting_summary.core_requirements,
            "nice_to_have": posting_summary.nice_to_have,
            "summary_paragraph": posting_summary.summary_paragraph,
        },
    }

    record = {
        "posting_line": normalized_text,
        "matched_skills": matched_skills,
        "missing_skills": missing,
        "missing_skills_discarded": discarded,
        "extraction_debug_samples": [debug],
    }
    return [record]
