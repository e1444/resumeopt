"""Stage 1: per-chunk candidate skill extraction via a reasoning model.

BATCHED across chunks (multiple chunks per call, see `EXTRACTION_BATCH_SIZE`)
on the heuristic that a large, fixed portion of every call's prompt (task
instructions, schema, few-shot-style guidance) is repeated identically
regardless of how many chunks it covers, and modern models have large
enough context windows to hold several short excerpts at once.

Added 2026-07-17, then benchmarked with a controlled A/B (same fixed chunk
list, only batch size varied): batch_size=6 cut tokens ~54% and calls ~79%
vs batch_size=1, but also cut raw Stage 1 extraction output ~24% (146 -> 111
terms) and final included terms ~20% (76 -> 61) on one large, jargon-dense
posting. A follow-up TERM-LEVEL inspection (not just aggregate counts) found
most of that apparent loss was NOT a real recall problem: the same
underlying concepts mostly survived under different phrasing/granularity in
the batched run (e.g. one compound term split into two cleaner atomic
terms, matching this module's own "split compound terms" instruction), and
a good portion of the rest was over-fragmentation/near-duplicate noise in
the UNBATCHED run rather than genuine distinct skills. Only a small,
genuine subset of terms (specific, narrow, no-counterpart losses like
"content-policy violation detection"/"mitigation impact measurement") were
confirmed real misses - accepted as a reasonable cost/recall trade-off. A
separate benchmark on a second, simpler posting (`sample_job_posting_big2.
txt`) found batching IMPROVED both precision and F1 (86.27% -> 94.55%) with
no recall cost at all, alongside the same large token/call savings - the
effect is posting-complexity-dependent, not a blanket downside. Per
explicit user decision (2026-07-17), batching is adopted as the production
default given its consistent, large cost efficiency and the minor,
acceptable recall trade-off. Batches within a call are still concurrent
across BATCHES via `asyncio.gather`. Optionally attaches the posting's
global summary as context. Deliberately RECALL-FIRST and over-includes
(e.g. degree-enumeration items, redundant broader-category restatements) -
`categorization.py` (Stage 2) is responsible for precision.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from llm import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REASONING_EFFORT,
    LLMProvider,
    batch_list,
    call_json_with_retry_async,
)

EXTRACTION_BATCH_SIZE = 6
"""Chunks grouped into a single Stage 1 call. Production default per
explicit user decision (2026-07-17, see module docstring) - large, 
consistent token/call savings, with only a minor accepted recall trade-off
on a live benchmark. Pass `batch_size=1` for the (slower, more expensive,
slightly-higher-recall-on-dense-postings) one-call-per-chunk behavior."""

_EXTRACTION_JSON_SCHEMA = {
    "name": "chunk_skill_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "excerpts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "reasoning": {"type": "string"},
                        "skills": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "term": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["term", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["index", "reasoning", "skills"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["excerpts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: below are several NUMBERED excerpts (each a single sentence or short passage from a job "
    "posting). For EACH excerpt INDEPENDENTLY, read ONLY that excerpt's own text and extract every "
    "skill/technique term that should appear in a resume's SKILLS section, based on that excerpt "
    "specifically - do NOT let one excerpt's content influence another excerpt's extraction.\n"
    "Include a term if it is: a real, named technology/tool/platform/programming language/algorithm; an "
    "established academic/technical/scientific discipline; OR a recognized professional practice/"
    "methodology (e.g. code review, unit testing, version control) - AND it is relevant to the role "
    "(explicitly required, a natural expected skill, or a nice-to-have/asset).\n"
    "Do NOT include: soft skills (communication, teamwork, problem-solving); job titles or an employer/"
    "vendor/client name that is NOT itself a tool (a product/platform name that IS the named tool/"
    "technology being used, e.g. GitHub, Docker, AWS, is a valid technical skill even though it is also a "
    "brand); compensation/benefit items; a business/industry domain label or company-"
    "specific business initiative described using technical-sounding words (a section headline naming a "
    "business function/program, as opposed to a specific technique); a generic, freely-substitutable "
    "description of an activity with no specific named technique behind it (if an ordinary synonym could "
    "replace a word in the phrase without changing its meaning, it is NOT a fixed term); or a term that "
    "ONLY represents one option inside a 'degree in X, Y, or Z (or equivalent experience)' style "
    "enumeration (a qualification alternative, not a practiced skill).\n"
    "A term does not need to be common or familiar to qualify - narrow, specialized, or compound "
    "technical terms are still valid if they are real, specific, named techniques or practices.\n"
    "Split compound/joined lists (e.g. 'GLM/GBM', 'writing tests and producing documentation') into "
    "separate atomic terms.\n"
    "Use the EXACT wording, spelling, and abbreviation form that appears in the excerpt for each term - "
    "do NOT expand an abbreviation into its spelled-out name (write 'NLP' if the excerpt says 'NLP', NOT "
    "'Natural Language Processing (NLP)') and do NOT abbreviate a term that is already spelled out in "
    "full. A later stage verifies every term is an exact, literal substring of the original posting text, "
    "so a paraphrased, expanded, or abbreviated form gets discarded even when the underlying concept is "
    "correct - only the literal wording as written survives.\n"
    "For EACH excerpt, first write a brief `reasoning` narrative explaining your overall read of THAT "
    "excerpt, THEN list each extracted `term` with its own short `reason`. If you are unsure whether "
    "something belongs, still include it - a later stage independently re-checks each candidate more "
    "strictly. Return one entry per excerpt index below, even if its `skills` list is empty.\n"
    "{summary_section}"
    "Excerpts:\n{excerpts_block}"
)


def _build_prompt(batch_chunks: List[str], summary_block: Optional[str]) -> str:
    summary_section = (
        f"Posting summary (for context, not any excerpt itself):\n{summary_block}\n\n" if summary_block else ""
    )
    excerpts_block = "\n".join(f"{i}. excerpt: {chunk!r}" for i, chunk in enumerate(batch_chunks, start=1))
    return _TASK_PROMPT.format(summary_section=summary_section, excerpts_block=excerpts_block)


async def _extract_one_batch(
    llm_provider: LLMProvider,
    batch_chunks: List[str],
    summary_block: Optional[str],
    semaphore: asyncio.Semaphore,
    on_batch_done: Optional[Callable[[], None]] = None,
) -> List[Dict[str, Any]]:
    async with semaphore:
        prompt = _build_prompt(batch_chunks, summary_block)
        payload = await call_json_with_retry_async(
            llm_provider,
            "chunk_extraction",
            prompt=prompt,
            system_prompt=(
                "You extract resume-worthy technical skills from several independent, numbered "
                "job-posting excerpts, one at a time. Return valid JSON only."
            ),
            temperature=0.2,
            max_tokens=1500 * len(batch_chunks),
            json_schema=_EXTRACTION_JSON_SCHEMA,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        by_index: Dict[int, Dict[str, Any]] = {}
        for item in (payload or {}).get("excerpts", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int) and 1 <= index <= len(batch_chunks):
                by_index[index] = item

        results: List[Dict[str, Any]] = []
        for i, chunk in enumerate(batch_chunks, start=1):
            # Missing index (partial response, or the whole call failed) -
            # fail-safe defaults to no terms for that one excerpt, matching
            # the prior one-call-per-chunk behavior's own failure default.
            item = by_index.get(i, {"reasoning": "", "skills": []})
            skills = item.get("skills", [])
            terms = [str(s.get("term", "")).strip() for s in skills if isinstance(s, dict)]
            term_reasons = {
                str(s.get("term", "")).strip(): str(s.get("reason", ""))
                for s in skills
                if isinstance(s, dict) and str(s.get("term", "")).strip()
            }
            results.append(
                {
                    "chunk": chunk,
                    "reasoning": str(item.get("reasoning", "")),
                    "terms": [term for term in terms if term],
                    "term_reasons": term_reasons,
                }
            )
        if on_batch_done is not None:
            try:
                on_batch_done()
            except Exception:
                pass
        return results


async def extract_candidates_for_chunks(
    llm_provider: LLMProvider,
    chunks: List[str],
    summary_block: Optional[str] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    on_batch_done: Optional[Callable[[], None]] = None,
) -> List[Dict[str, Any]]:
    """Run Stage 1 extraction over every chunk, batched `batch_size` chunks
    per call, batches running concurrently.

    Returns one result dict per chunk (same order as `chunks`):
    `{"chunk": str, "reasoning": str, "terms": List[str], "term_reasons": Dict[str, str]}`.

    `on_batch_done` (optional): called once per completed batch, for
    progress-reporting purposes (see `chunk_screening.screen_chunks_for_skill_likelihood`'s
    docstring for the same convention).
    """

    if not chunks:
        return []

    batches = batch_list(chunks, batch_size)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    batch_results = await asyncio.gather(
        *(_extract_one_batch(llm_provider, batch, summary_block, semaphore, on_batch_done) for batch in batches)
    )
    results: List[Dict[str, Any]] = []
    for batch_result in batch_results:
        results.extend(batch_result)
    return results
