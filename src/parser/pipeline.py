"""Orchestrates the parser pipeline: chunking (LLM-based by default, see
`chunker.split_into_sentence_chunks_via_llm`) -> Stage 0.5 (cheap chunk
screening) -> Stage 1 (extraction) -> Stage 2 (categorization) -> Stage 3a
(global keyword-atomicity gate) -> Stage 3b (within-chunk redundancy check,
only for non-atomic terms) -> deduped, final per-term verdicts.

See `chunk_screening.py`, `extraction.py`, `categorization.py`,
`keyword_atomicity.py`, `redundancy.py`, and `models.py` for each stage's own
docstring. This module only wires them together and resolves cross-chunk
duplicates (the same skill can legitimately be extracted from more than one
chunk in a real posting) - first occurrence wins, keeping that occurrence's
chunk as the term's grounding/evidence.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from chunker import split_into_sentence_chunks, split_into_sentence_chunks_via_llm
from llm import DEFAULT_BATCH_SIZE, DEFAULT_MAX_CONCURRENCY, LLMProvider, batch_list

from .categorization import INCLUDED_CATEGORY, CATEGORIZATION_BATCH_SIZE, categorize_candidates_for_chunks
from .chunk_screening import screen_chunks_for_skill_likelihood
from .extraction import EXTRACTION_BATCH_SIZE, extract_candidates_for_chunks
from .keyword_atomicity import check_keyword_atomicity
from .models import ChunkSkillVerdict
from .redundancy import check_redundancy_for_chunks


def _make_batch_counter(
    on_substage: Optional[Callable[[str, int, int], None]], name: str, total: int
) -> Optional[Callable[[], None]]:
    """Builds a closure that reports "N of `total` batches done" for one
    substage, called once per completed batch (see each stage module's
    `on_batch_done` docstring). Returns `None` (no-op) if there's no
    `on_substage` callback to report to, or nothing to batch at all.
    """

    if on_substage is None or total <= 0:
        return None

    completed = 0

    def _on_done() -> None:
        nonlocal completed
        completed += 1
        try:
            on_substage(name, completed, total)
        except Exception:
            pass

    return _on_done


async def run_parser_pipeline(
    reasoning_llm_provider: LLMProvider,
    posting_text: str,
    summary_block: Optional[str] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    enable_redundancy_check: bool = True,
    use_llm_chunking: bool = True,
    screening_llm_provider: Optional[LLMProvider] = None,
    enable_chunk_screening: bool = True,
    extraction_batch_size: int = EXTRACTION_BATCH_SIZE,
    categorization_batch_size: int = CATEGORIZATION_BATCH_SIZE,
    on_substage: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, ChunkSkillVerdict]:
    """Run the full parser pipeline over a whole posting.

    `summary_block` (optional) is the posting's global summary (see
    `parser.summary.generate_posting_summary` + `format_summary_block`),
    attached to every Stage 1 extraction call as background context.

    `screening_llm_provider` (optional) runs Stage 0.5 - a cheap, batched,
    coarse screen that skips chunks unlikely to contain any resume-worthy
    skill at all, so they never reach Stage 1/2. Ideally a cheaper/faster,
    non-reasoning-tier model (e.g. `gpt-4o-mini`) - ordinary judgment, not
    multi-step reasoning, is enough for this coarse a filter. Defaults to
    `reasoning_llm_provider` if not given (still functionally correct, just
    without the cost benefit of a separate cheap model). Set
    `enable_chunk_screening=False` to skip this stage entirely (e.g. for
    ablation comparisons).

    `extraction_batch_size`/`categorization_batch_size` (default to each
    module's own production default of `6`, i.e. several chunks per call)
    let a caller override the batch size for Stage 1/2 - see
    `extraction.py`'s module docstring for the cost/quality trade-off
    analysis behind this default (large, consistent token/call savings, a
    minor accepted recall trade-off on dense postings, and an F1 IMPROVEMENT
    on simpler postings). Pass `1` for one-call-per-chunk behavior.

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

    `on_substage` (optional): called as `(name, completed, total)` once per
    completed batch within each of the 5 batched sub-stages (`chunk_screening`,
    `extraction`, `categorization`, `atomicity`, `redundancy`) - a pure
    progress-reporting hook (e.g. for a caller-facing progress bar) with no
    effect on pipeline behavior. `total` for each sub-stage is known upfront
    (batch counts don't change once a sub-stage's input list is fixed), so
    this gives real, monotonically-increasing sub-progress within what would
    otherwise be one single long, opaque "parsing" step.

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

    if enable_chunk_screening:
        screening_provider = screening_llm_provider or reasoning_llm_provider
        screening_total = len(batch_list(chunks, DEFAULT_BATCH_SIZE))
        screening_results = await screen_chunks_for_skill_likelihood(
            screening_provider,
            chunks,
            max_concurrency=max_concurrency,
            on_batch_done=_make_batch_counter(on_substage, "chunk_screening", screening_total),
        )
        chunks = [chunk for chunk in chunks if screening_results.get(chunk, True)]
    if not chunks:
        return {}

    extraction_total = len(batch_list(chunks, extraction_batch_size))
    extraction_results = await extract_candidates_for_chunks(
        reasoning_llm_provider,
        chunks,
        summary_block,
        max_concurrency,
        batch_size=extraction_batch_size,
        on_batch_done=_make_batch_counter(on_substage, "extraction", extraction_total),
    )
    chunk_terms: List[dict] = [
        {"chunk": result["chunk"], "terms": result["terms"]} for result in extraction_results
    ]
    categorization_total = len(batch_list([entry for entry in chunk_terms if entry["terms"]], categorization_batch_size))
    categorization_results = await categorize_candidates_for_chunks(
        reasoning_llm_provider,
        chunk_terms,
        max_concurrency,
        batch_size=categorization_batch_size,
        on_batch_done=_make_batch_counter(on_substage, "categorization", categorization_total),
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
    atomicity_total = len(batch_list(all_survivor_terms, DEFAULT_BATCH_SIZE))
    atomicity_results = await check_keyword_atomicity(
        reasoning_llm_provider,
        all_survivor_terms,
        max_concurrency=max_concurrency,
        on_batch_done=_make_batch_counter(on_substage, "atomicity", atomicity_total),
    )

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
            reasoning_llm_provider,
            redundancy_chunk_terms,
            max_concurrency,
            on_batch_done=_make_batch_counter(on_substage, "redundancy", len(redundancy_chunk_terms)),
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
