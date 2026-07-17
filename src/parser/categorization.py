"""Stage 2: per-chunk multi-class categorical classification, filtering
Stage 1's (deliberately recall-first, over-inclusive) raw candidates down to
genuine resume technical skills.

Reframing "is this a skill" as ONE multi-class question with an explicit
`degree_or_qualification` bucket - instead of forcing every candidate into a
binary include/exclude decision - was validated live: precision 75.58% ->
85.51% with recall held at 100.00% (see repo memory /
`build/benchmarks/skill_category_stage2_benchmark_gpt-5-mini.json`).

Only `resume_technical_skill` counts as "included" downstream - the other 3
categories (`degree_or_qualification`, `soft_skill`, `non_skill`) are all
exclusions, but keeping them as distinct labels (rather than one generic
"excluded") is itself the fix: it removes the binary framing that caused the
original ambiguity in a naive include/exclude classifier.

BATCHED across chunks (multiple chunks' candidate lists per call, see
`CATEGORIZATION_BATCH_SIZE`) on the same rationale as `extraction.py`.
Benchmarked together in a controlled A/B (2026-07-17, same fixed chunk
list, only batch size varied) alongside Stage 1 - see `extraction.py`'s
module docstring for the full numbers and the term-level follow-up
analysis that found most of the apparent recall drop was phrasing/
granularity variance and over-fragmentation cleanup rather than genuine
losses, plus a separate benchmark showing batching IMPROVING F1 on a
simpler posting. Per explicit user decision, `CATEGORIZATION_BATCH_SIZE`
is the production default, matching Stage 1.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from llm import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REASONING_EFFORT,
    LLMProvider,
    batch_list,
    call_json_with_retry_async,
)

CATEGORIES = ("resume_technical_skill", "degree_or_qualification", "soft_skill", "non_skill")
INCLUDED_CATEGORY = "resume_technical_skill"

CATEGORIZATION_BATCH_SIZE = 6
"""Chunks grouped into a single Stage 2 call. Production default per
explicit user decision (2026-07-17), same rationale as
`extraction.EXTRACTION_BATCH_SIZE`."""

_CATEGORY_JSON_SCHEMA = {
    "name": "skill_category_flags",
    "schema": {
        "type": "object",
        "properties": {
            "excerpts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "verdicts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "term": {"type": "string"},
                                    "category": {"type": "string", "enum": list(CATEGORIES)},
                                    "reason": {"type": "string"},
                                },
                                "required": ["term", "category", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["index", "verdicts"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["excerpts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: below are several NUMBERED excerpts, each with its own local context and its own list of "
    "candidate terms. For EACH excerpt INDEPENDENTLY, classify EACH of that excerpt's candidate terms "
    "into EXACTLY ONE of these 4 categories, using ONLY that excerpt's own local context - do NOT let "
    "one excerpt's context influence another excerpt's candidates.\n"
    "- resume_technical_skill: a specific, named technology/tool/platform/programming language/"
    "algorithm/established academic-technical discipline, OR a recognized professional practice/"
    "methodology (e.g. code review, unit testing, version control) - genuinely something a practitioner "
    "would list in a resume's technical skills section.\n"
    "- degree_or_qualification: an academic major, degree, or credential named as an educational "
    "requirement or as one option inside a 'degree in X, Y, or Z (or equivalent experience)' style "
    "enumeration - this is a QUALIFICATION ALTERNATIVE, not a demonstrated, independently practiced "
    "skill, even if the discipline named is itself a real field.\n"
    "- soft_skill: an interpersonal/behavioral quality (communication, teamwork, problem-solving, "
    "leadership) - not a technical skill.\n"
    "- non_skill: anything else - a job title, organization/product/brand name, compensation/benefit "
    "item, business/industry domain label or company-specific business initiative, a generic "
    "freely-substitutable activity description with no fixed named technique behind it, or filler.\n"
    "Use the LOCAL CONTEXT to decide - the SAME word can belong to a different category depending on how "
    "it is actually used in its own excerpt (e.g. 'statistics' named as one option in a degree "
    "enumeration is degree_or_qualification, but 'statistics' named as a core methodology the role "
    "applies daily is resume_technical_skill).\n"
    "Return one entry per excerpt index below, with a `verdicts` array covering every one of that "
    "excerpt's candidate terms.\n\n"
    "Excerpts:\n{excerpts_block}"
)


def _build_excerpt_block(entries: List[Dict[str, Any]]) -> str:
    parts = []
    for i, entry in enumerate(entries, start=1):
        candidates_block = "\n".join(f"  - term: {term!r}" for term in entry["terms"])
        parts.append(f"{i}. excerpt (local context): {entry['chunk']!r}\n   candidates:\n{candidates_block}")
    return "\n\n".join(parts)


async def _categorize_one_batch(
    llm_provider: LLMProvider,
    batch_entries: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Dict[str, str]]]:
    async with semaphore:
        prompt = _TASK_PROMPT.format(excerpts_block=_build_excerpt_block(batch_entries))
        payload = await call_json_with_retry_async(
            llm_provider,
            "skill_category",
            prompt=prompt,
            system_prompt=(
                "You classify candidate resume-skill terms from several independent, numbered excerpts, "
                "each using only its own local context. Return valid JSON only."
            ),
            temperature=0.1,
            max_tokens=1500 * len(batch_entries),
            json_schema=_CATEGORY_JSON_SCHEMA,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        by_index: Dict[int, List[Dict[str, Any]]] = {}
        for item in (payload or {}).get("excerpts", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int) and 1 <= index <= len(batch_entries):
                by_index[index] = item.get("verdicts", [])

        results: List[Dict[str, Dict[str, str]]] = []
        for i in range(1, len(batch_entries) + 1):
            verdicts = by_index.get(i, [])
            result: Dict[str, Dict[str, str]] = {}
            for item in verdicts:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term", "")).strip()
                category = str(item.get("category", "")).strip()
                if term and category in CATEGORIES:
                    result[term] = {"category": category, "reason": str(item.get("reason", ""))}
            results.append(result)
        return results


async def categorize_candidates_for_chunks(
    llm_provider: LLMProvider,
    chunk_terms: List[Dict[str, Any]],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    batch_size: int = CATEGORIZATION_BATCH_SIZE,
) -> List[Dict[str, Dict[str, str]]]:
    """Run Stage 2 categorization over every chunk's candidates, batched
    `batch_size` chunks per call, batches running concurrently.

    `chunk_terms` is a list of `{"chunk": str, "terms": List[str]}` dicts (one
    per Stage 1 chunk result). Returns one `{term: {"category": ..., "reason": ...}}`
    dict per chunk, same order as `chunk_terms`. Terms missing from a chunk's
    result (e.g. a failed batch, or a partial response missing that chunk's
    index) are simply absent - callers should treat a missing category as
    "not resume_technical_skill" (fail-safe toward exclusion, since this
    stage exists specifically to tighten precision). Chunks with zero
    candidate terms are skipped entirely (no call made for them at all).
    """

    if not chunk_terms:
        return []

    # Chunks with no candidates don't need a call at all - carry them
    # through as empty results without occupying a batch slot.
    non_empty_indices = [i for i, entry in enumerate(chunk_terms) if entry["terms"]]
    non_empty_entries = [chunk_terms[i] for i in non_empty_indices]

    results: List[Dict[str, Dict[str, str]]] = [{} for _ in chunk_terms]
    if not non_empty_entries:
        return results

    batches = batch_list(non_empty_entries, batch_size)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    batch_results = await asyncio.gather(
        *(_categorize_one_batch(llm_provider, batch, semaphore) for batch in batches)
    )

    flat_results: List[Dict[str, Dict[str, str]]] = []
    for batch_result in batch_results:
        flat_results.extend(batch_result)
    for original_index, result in zip(non_empty_indices, flat_results):
        results[original_index] = result
    return results

