"""Stage 0.5: cheap, batch-wise chunk screening - runs BEFORE Stage 1
extraction and skips chunks unlikely to contain any resume-worthy skill at
all, so they never cost a Stage 1 extraction + Stage 2 categorization call
cycle.

Added 2026-07-17 after a deterministic boilerplate-section pre-filter (an
earlier, now-REMOVED approach: dropping whole known sections like "About
Us"/company values by header-pattern matching, before chunking) was found
to not eliminate enough waste on its own, AND was later found to add no
measurable benefit once this stage existed (a live A/B showed this stage
alone converges to the same chunk count and token cost) - it was removed
rather than kept alongside this stage. The remaining problem this stage
solves: many individually-chunked SENTENCES within otherwise-legitimate
sections (e.g. "Tell the story behind the data.", "Operationalize what
works.", "Anticipate analytical tradeoffs.") are vague, motivational, or
process-framing statements with no concrete named skill in them, yet each
still costs a full extraction+categorization cycle and often produces
speculative, low-confidence, or outright hallucinated/ungrounded terms (the
model inferring "MLOps"/"CI/CD" from a one-line "Operationalize what
works." is a stretch, not a real extraction). A boilerplate SECTION filter
can't catch this - it only removes whole known sections, not individual
low-value sentences scattered throughout otherwise-legitimate sections.

Deliberately uses a NON-reasoning-tier model by default (e.g. `gpt-4o-mini`)
- this is a coarse, single yes/no judgment ("could this sentence plausibly
name or imply a concrete skill/tool/technique") that doesn't need multi-step
reasoning, and batching many chunks into few calls (rather than the reasoning
model's one-call-per-chunk convention) keeps this screening step itself
cheap relative to the extraction/categorization calls it's meant to avoid.

Deliberately INCLUSION-BIASED (tie-breaker: if genuinely unsure, keep) -
this stage can only ever SAVE a wasted call, never rescue a true skill that
got wrongly screened out (unlike Stage 3's redundancy check, which still
gets audited downstream); a false negative here is a genuine, silent recall
loss, unlike a false positive's minor unnecessary call. Terms missing from
a batch's response (failed call) default to `likely_to_contain_skills=True`
(kept) for the same reason.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from llm import DEFAULT_BATCH_SIZE, DEFAULT_MAX_CONCURRENCY, LLMProvider, batch_list, call_json_with_retry_async

_SCREENING_JSON_SCHEMA = {
    "name": "chunk_skill_screening",
    "schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "likely_to_contain_skills": {"type": "boolean"},
                    },
                    "required": ["index", "likely_to_contain_skills"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: for EACH numbered excerpt below (a single sentence/bullet from a job posting), decide whether "
    "it plausibly NAMES OR IMPLIES a concrete, resume-worthy technical skill, tool, technology, "
    "methodology, or named practice - something a later, more careful review pass would actually be able "
    "to extract a real skill term from.\n"
    "Label likely_to_contain_skills=true if the excerpt names or clearly implies a specific technology, "
    "tool, programming language, technical discipline, or named professional practice/methodology "
    "(examples: 'Analyze telemetry, usage logs...' -> true; 'SQL, Python, Power BI' -> true; 'Detect and "
    "investigate Copilot abuse.' -> true, since 'abuse detection' is a real named practice).\n"
    "Label likely_to_contain_skills=false if the excerpt is vague, motivational, purely process-framing, "
    "an org-chart/team-name mention, a soft-skill statement, or a generic activity description with no "
    "specific named technique behind it (examples: 'Tell the story behind the data.' -> false (generic "
    "communication framing, not a named skill); 'Operationalize what works.' -> false (vague filler "
    "phrase, no specific named practice); 'Mentor and influence.' -> false (soft skill only); 'Ship to "
    "learn' -> false (company value/slogan, not a named technical practice)).\n"
    "Tie-breaker: if genuinely unsure, label likely_to_contain_skills=true - a later stage independently "
    "re-checks each extracted candidate far more strictly; the cost of wrongly keeping a low-value "
    "excerpt is small, but wrongly discarding a real skill here is unrecoverable.\n\n"
    "Excerpts:\n{candidates_block}"
)


async def _screen_one_batch(
    llm_provider: LLMProvider,
    batch_chunks: List[str],
    semaphore: asyncio.Semaphore,
) -> Dict[int, bool]:
    async with semaphore:
        candidates_block = "\n".join(f"{i}. chunk: {chunk!r}" for i, chunk in enumerate(batch_chunks, start=1))
        prompt = _TASK_PROMPT.format(candidates_block=candidates_block)
        payload = await call_json_with_retry_async(
            llm_provider,
            "chunk_skill_screening",
            prompt=prompt,
            system_prompt=(
                "You do a fast, coarse screen of job-posting excerpts to flag which ones plausibly "
                "contain a concrete resume skill, before a slower, more careful pass. Return valid JSON "
                "only."
            ),
            temperature=0.1,
            max_tokens=1500,
            json_schema=_SCREENING_JSON_SCHEMA,
        )
        verdicts = (payload or {}).get("verdicts", [])
        result: Dict[int, bool] = {}
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int) and 1 <= index <= len(batch_chunks):
                result[index] = bool(item.get("likely_to_contain_skills", True))
        return result


async def screen_chunks_for_skill_likelihood(
    llm_provider: LLMProvider,
    chunks: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Dict[str, bool]:
    """Batched, concurrent, coarse screen of every chunk for skill likelihood.

    Returns `{chunk: likely_to_contain_skills}`. A chunk missing from the
    result (failed batch) should be treated as `True` (kept) by callers - a
    fail-safe toward recall, matching this stage's inclusion-biased design.

    Duplicate chunk text (rare but possible) collapses to a single dict key;
    callers filtering the original `chunks` list should look up by chunk
    text with a `True` default, not iterate this dict directly.
    """

    if not chunks:
        return {}

    batches = batch_list(chunks, batch_size)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    batch_results = await asyncio.gather(
        *(_screen_one_batch(llm_provider, batch, semaphore) for batch in batches)
    )

    merged: Dict[str, bool] = {}
    for batch_chunks, batch_result in zip(batches, batch_results):
        for index, chunk in enumerate(batch_chunks, start=1):
            merged[chunk] = batch_result.get(index, True)
    return merged
