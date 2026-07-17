"""Stage 3a: context-free, sibling-free "intrinsic keyword atomicity" gate -
decomposes the single, conflated "is this redundant" question into two
separable judgments.

Asking "does this term have independent, standalone resume/ATS-keyword
value, evaluated WITHOUT knowledge of what other candidate terms happen to
also be present" removes a specific over-aggression pattern found in the
single-question redundancy design it replaces as the first step (e.g.
`Machine Learning`/`Data Science`/`DevOps` wrongly dropped as "redundant"
with their own sub-techniques, precisely BECAUSE rich postings extract many
sub-techniques for exactly the most foundational fields - the sibling-count
signal perversely penalizes the most important terms).

Deliberately CONTEXT-FREE and GLOBAL (no chunk text, no sibling terms - one
flat, deduplicated list across the WHOLE posting): keeping this judgment
free of any co-occurring-candidate information is the whole point - it's
what prevents the "has many specific siblings present" confound from
leaking into the "is this term intrinsically valuable on its own" answer.

A term found `atomic_keyword=True` here BYPASSES `redundancy.py`'s within-
chunk check entirely (always kept, once it already passed Stage 2). Only
`atomic_keyword=False` terms (pure category/descriptive labels with no
standalone identity) are sent to the redundancy check at all. Benchmarked
against the single-question approach in
`build/benchmarks/keyword_atomicity_benchmark_gpt-5-mini.json`: fixes the
confirmed over-aggression bug and achieves 100% recall (matching the
"no Stage 3 at all" baseline), at a modest precision/F1 cost relative to the
single-question approach (F1 88.01% vs 89.01%) - promoted to production per
explicit user-approved verdict favoring the bug fix and perfect recall.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from llm import DEFAULT_BATCH_SIZE, DEFAULT_MAX_CONCURRENCY, LLMProvider, batch_list, call_json_with_retry_async

_ATOMICITY_JSON_SCHEMA = {
    "name": "keyword_atomicity_flags",
    "schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "atomic_keyword": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["term", "atomic_keyword", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: for EACH candidate term below, decide whether it is, ON ITS OWN, an independently "
    "recognized, resume/ATS-worthy KEYWORD - a named field of study, discipline, widely-recognized "
    "practice/methodology, or specific tool/technology that a hiring manager or applicant-tracking "
    "system would search for and that is worth its own resume line REGARDLESS of whether more "
    "specific related skills might also be listed alongside it.\n"
    "Judge EACH term COMPLETELY INDEPENDENTLY of the others in this list - do not consider what other "
    "candidates are present; a term's answer must not depend on which OTHER terms happen to also be "
    "in this batch.\n"
    "Label atomic_keyword=true if: the term is itself an established field of study, named discipline, "
    "widely-recognized practice/methodology, or a specific product/tool/technology name - something "
    "that functions as a standalone keyword in professional/hiring contexts on its own merits (for "
    "example 'Machine Learning', 'Data Science', 'DevOps', 'Kubernetes', 'Infrastructure as Code', "
    "'Natural Language Processing', 'Git').\n"
    "Label atomic_keyword=false if: the term is a generic, purely descriptive category/activity label "
    "whose ONLY function is to describe or group other, more specific things, with no standalone "
    "identity of its own in professional/hiring contexts (for example 'version control', 'programming', "
    "'automation', 'tools', bare 'monitoring' used as an activity description).\n"
    "Tie-breaker: if genuinely unsure, label atomic_keyword=true - lean toward treating a term as "
    "independently valuable rather than assuming it is merely a generic category label.\n"
    "Examples:\n"
    "- term: 'Machine Learning' -> true. An established field/discipline with its own standalone "
    "identity, independent of whichever specific ML techniques are also listed.\n"
    "- term: 'version control' -> false. A purely descriptive category label with no standalone "
    "identity beyond describing tools like Git/SVN/Mercurial.\n"
    "- term: 'Infrastructure as Code' -> true. A named, established methodology in its own right, "
    "independent of which specific tool (Terraform, CloudFormation, Pulumi) implements it."
)


async def _check_one_batch(
    llm_provider: LLMProvider,
    batch_terms: List[str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, Any]]:
    async with semaphore:
        candidates_block = "\n".join(f"{i}. term: {term!r}" for i, term in enumerate(batch_terms, start=1))
        prompt = f"{_TASK_PROMPT}\n\nCandidates:\n{candidates_block}"
        payload = await call_json_with_retry_async(
            llm_provider,
            "keyword_atomicity",
            prompt=prompt,
            system_prompt=(
                "You judge whether candidate resume-skill terms are independently valuable standalone "
                "keywords, evaluated in isolation from any other candidates. Return valid JSON only."
            ),
            temperature=0.1,
            max_tokens=1500,
            json_schema=_ATOMICITY_JSON_SCHEMA,
        )
        verdicts = (payload or {}).get("verdicts", [])
        result: Dict[str, Dict[str, Any]] = {}
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            if term:
                result[term] = {
                    "atomic_keyword": bool(item.get("atomic_keyword", True)),
                    "reason": str(item.get("reason", "")),
                }
        return result


async def check_keyword_atomicity(
    llm_provider: LLMProvider,
    candidate_terms: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Dict[str, Dict[str, Any]]:
    """Batched, concurrent, CONTEXT-FREE keyword-atomicity check.

    `candidate_terms` should be the FULL, DEDUPLICATED set of Stage-2
    survivors across the whole posting - no chunk grouping, no sibling
    information, by design (see module docstring).

    Returns {term: {"atomic_keyword": bool, "reason": str}}. Terms missing
    from the result (failed batch) default to `atomic_keyword=True` when
    consumed by the pipeline - a fail-safe toward KEEPING (bypassing the
    redundancy check entirely) rather than risking a wrongly-dropped keyword,
    consistent with this stage's inclusion-biased tie-breaker.
    """

    batches = batch_list(candidate_terms, batch_size)
    if not batches:
        return {}

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    batch_results = await asyncio.gather(
        *(_check_one_batch(llm_provider, batch, semaphore) for batch in batches)
    )
    merged: Dict[str, Dict[str, Any]] = {}
    for result in batch_results:
        merged.update(result)
    return merged
