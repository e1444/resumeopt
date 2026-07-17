"""Orchestrates the parser pipeline: chunking (LLM-based by default, see
`chunker.split_into_sentence_chunks_via_llm`) -> Stage 1 (extraction) ->
Stage 2 (categorization) -> Stage 3a (global keyword-atomicity gate) ->
Stage 3b (within-chunk redundancy check, only for non-atomic terms) ->
deduped, final per-term verdicts.

See `extraction.py`, `categorization.py`, `keyword_atomicity.py`,
`redundancy.py`, and `models.py` for each stage's own docstring. This module
only wires them together and resolves cross-chunk duplicates (the same
skill can legitimately be extracted from more than one chunk in a real
posting) - first occurrence wins, keeping that occurrence's chunk as the
term's grounding/evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from chunker import split_into_sentence_chunks, split_into_sentence_chunks_via_llm
from llm import DEFAULT_MAX_CONCURRENCY, LLMProvider

from .categorization import INCLUDED_CATEGORY, categorize_candidates_for_chunks
from .extraction import extract_candidates_for_chunks
from .keyword_atomicity import check_keyword_atomicity
from .models import ChunkSkillVerdict
from .redundancy import check_redundancy_for_chunks


async def run_parser_pipeline(
    reasoning_llm_provider: LLMProvider,
    posting_text: str,
    summary_block: Optional[str] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    enable_redundancy_check: bool = True,
    use_llm_chunking: bool = True,
) -> Dict[str, ChunkSkillVerdict]:
    """Run the full parser pipeline over a whole posting.

    `summary_block` (optional) is the posting's global summary (see
    `parser.summary.generate_posting_summary` + `format_summary_block`),
    attached to every Stage 1 extraction call as background context.

    `use_llm_chunking` (default `True`) uses `chunker.split_into_sentence_
    chunks_via_llm` (same `reasoning_llm_provider`) instead of the
    deterministic regex splitter (`chunker.split_into_sentence_chunks`) -
    validated live against 2 real postings to correctly separate section
    headers from bullets and split punctuation-free bullet lists the regex
    splitter merges into giant blobs. Falls back to the deterministic
    chunker automatically if the LLM call fails or returns nothing grounded.
    Set `False` for the deterministic-only behavior (e.g. for ablation
    comparisons or to avoid the extra call).

    `enable_redundancy_check` (default `True`) runs Stage 3 (keyword-
    atomicity gate, then within-chunk redundancy for non-atomic terms only)
    on Stage 2's survivors. Set `False` to fall back to Stage 1+2-only
    behavior (e.g. for ablation comparisons).

    Returns one verdict per DISTINCT (normalized) term found anywhere in the
    posting - if the same term is extracted from more than one chunk, the
    first occurrence wins.
    """

    if use_llm_chunking:
        chunks = await split_into_sentence_chunks_via_llm(reasoning_llm_provider, posting_text)
    else:
        chunks = split_into_sentence_chunks(posting_text)
    if not chunks:
        return {}

    extraction_results = await extract_candidates_for_chunks(
        reasoning_llm_provider, chunks, summary_block, max_concurrency
    )
    chunk_terms: List[dict] = [
        {"chunk": result["chunk"], "terms": result["terms"]} for result in extraction_results
    ]
    categorization_results = await categorize_candidates_for_chunks(
        reasoning_llm_provider, chunk_terms, max_concurrency
    )

    # Stage 1+2 verdicts first (no atomicity/redundancy info yet) - deduped
    # across chunks (first occurrence wins).
    verdicts: Dict[str, ChunkSkillVerdict] = {}
    survivors_by_chunk: Dict[str, List[str]] = {}
    for extraction, categories in zip(extraction_results, categorization_results):
        chunk = extraction["chunk"]
        for term in extraction["terms"]:
            normalized = term.strip().lower()
            if not normalized or normalized in verdicts:
                continue  # cross-chunk dedupe - first occurrence wins
            category_info = categories.get(term, {})
            category = category_info.get("category") or "uncategorized"
            included = category == INCLUDED_CATEGORY
            verdicts[normalized] = ChunkSkillVerdict(
                raw_term=term,
                chunk=chunk,
                category=category,
                extraction_reason=extraction["term_reasons"].get(term, ""),
                category_reason=category_info.get("reason", ""),
                included=included,
            )
            if included:
                survivors_by_chunk.setdefault(chunk, []).append(term)

    if not enable_redundancy_check or not survivors_by_chunk:
        return verdicts

    # Stage 3a: global, context-free keyword-atomicity gate over every
    # distinct Stage-2 survivor in the whole posting.
    all_survivor_terms = sorted({term for terms in survivors_by_chunk.values() for term in terms})
    atomicity_results = await check_keyword_atomicity(reasoning_llm_provider, all_survivor_terms, max_concurrency=max_concurrency)

    # Stage 3b: within-chunk redundancy check, only for the non-atomic subset.
    non_atomic_terms_by_chunk: Dict[str, List[str]] = {}
    for chunk, terms in survivors_by_chunk.items():
        non_atomic = [
            term
            for term in terms
            if not atomicity_results.get(term, {"atomic_keyword": True}).get("atomic_keyword", True)
        ]
        if non_atomic:
            non_atomic_terms_by_chunk[chunk] = non_atomic

    redundancy_by_chunk: Dict[str, Dict[str, Dict]] = {}
    if non_atomic_terms_by_chunk:
        redundancy_chunk_terms = [
            {"chunk": chunk, "terms": terms} for chunk, terms in non_atomic_terms_by_chunk.items()
        ]
        redundancy_results = await check_redundancy_for_chunks(
            reasoning_llm_provider, redundancy_chunk_terms, max_concurrency
        )
        redundancy_by_chunk = {
            entry["chunk"]: result for entry, result in zip(redundancy_chunk_terms, redundancy_results)
        }

    final_verdicts: Dict[str, ChunkSkillVerdict] = {}
    for normalized, verdict in verdicts.items():
        if not verdict.included:
            final_verdicts[normalized] = verdict
            continue

        # Missing from the result (failed batch) defaults to atomic_keyword=True
        # - a fail-safe toward keeping (bypassing redundancy entirely).
        atomicity_info = atomicity_results.get(verdict.raw_term, {"atomic_keyword": True, "reason": ""})
        atomic = bool(atomicity_info.get("atomic_keyword", True))

        if atomic:
            final_verdicts[normalized] = ChunkSkillVerdict(
                raw_term=verdict.raw_term,
                chunk=verdict.chunk,
                category=verdict.category,
                extraction_reason=verdict.extraction_reason,
                category_reason=verdict.category_reason,
                atomic_keyword=True,
                atomicity_reason=str(atomicity_info.get("reason", "")),
                included=True,
            )
            continue

        chunk_redundancy = redundancy_by_chunk.get(verdict.chunk, {})
        # Missing from the result (failed batch) defaults to keep=True - a
        # fail-safe toward recall.
        redundancy_info = chunk_redundancy.get(verdict.raw_term, {"keep": True, "redundant_with": [], "reason": ""})
        keep = bool(redundancy_info.get("keep", True))
        redundant_with = redundancy_info.get("redundant_with") or []
        final_verdicts[normalized] = ChunkSkillVerdict(
            raw_term=verdict.raw_term,
            chunk=verdict.chunk,
            category=verdict.category,
            extraction_reason=verdict.extraction_reason,
            category_reason=verdict.category_reason,
            atomic_keyword=False,
            atomicity_reason=str(atomicity_info.get("reason", "")),
            redundant_with=redundant_with,
            redundancy_reason=str(redundancy_info.get("reason", "")),
            included=keep,
        )
    return final_verdicts
