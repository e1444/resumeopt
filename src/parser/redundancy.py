"""Stage 3b: within-chunk redundancy/specificity filter, run only on the
subset of terms that FAILED Stage 3a's keyword-atomicity gate
(`keyword_atomicity.py`) - i.e. terms with no independent standalone
identity, which are the only ones where "is a more specific sibling also
present" is even a meaningful question.

An empirical test (see repo memory) showed embedding cosine similarity
CANNOT be used as the redundancy-detection mechanism itself: it has no
directionality (a true sibling pair like "python"/"javascript" scores
nearly as high as a true parent-child pair like "programming"/"python"),
and its magnitude is inconsistent across domains. Specificity/subsumption
is a world-knowledge judgment ("Grafana is a monitoring tool"), not a
similarity metric, so this stage asks an LLM directly instead of using
embeddings at all.

Scoped WITHIN A CHUNK (not whole-posting), per explicit cost-efficiency
rationale: "if the chunking is good, almost all related skills appear in
the same chunk." Only chunks with >=2 non-atomic surviving candidates are
worth checking - a lone candidate has no sibling to be redundant with.

CRITICAL conditional rule: a general term is only marked redundant if a MORE
SPECIFIC sibling is ALSO present in the same chunk's surviving candidate
list. Tie-breaker is inclusion-biased ("if genuinely unsure, keep").
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from llm import DEFAULT_MAX_CONCURRENCY, DEFAULT_REASONING_EFFORT, LLMProvider, call_json_with_retry_async

_REDUNDANCY_JSON_SCHEMA = {
    "name": "redundancy_flags",
    "schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "redundant_with": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["term", "keep", "redundant_with", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: the candidate skill terms below were all extracted from the SAME excerpt and already "
    "confirmed as legitimate resume technical skills. Some may be a GENERAL/BROADER restatement of "
    "another, MORE SPECIFIC term ALSO in this same list (for example: 'version control' is a general "
    "restatement given 'git' is also present; 'programming' is general given 'Python'/'JavaScript' are "
    "also present; 'monitoring'/'automation' is general given 'Grafana'/'Prometheus' are also present) - "
    "a tight resume skills section should keep the specific named instance and drop the now-redundant "
    "general category.\n"
    "For EACH candidate, decide keep=true or keep=false:\n"
    "- keep=false ONLY if a MORE SPECIFIC sibling term is ALSO present in this exact list, making this "
    "candidate a redundant, less-informative restatement of it. List the specific sibling(s) in "
    "`redundant_with`.\n"
    "- keep=true if no more-specific sibling for this candidate exists in this list - even if the term "
    "sounds general in isolation, it is the best available signal here and must be kept. Also keep=true "
    "for two genuinely DIFFERENT specific things that merely sound similar (e.g. two different named "
    "tools are NOT redundant with each other - only a general category is redundant with a specific "
    "instance of it).\n"
    "Tie-breaker: if genuinely unsure whether one is truly more specific than the other, keep=true - lean "
    "toward keeping both rather than discarding a potentially useful, distinct skill.\n\n"
    "Excerpt (local context):\n{chunk}\n\n"
    "Candidates:\n{candidates_block}"
)


async def _check_one_chunk(
    llm_provider: LLMProvider,
    chunk: str,
    terms: List[str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, Any]]:
    if len(terms) < 2:
        # Nothing to be redundant WITH - skip the call entirely, fail-safe keep.
        return {term: {"keep": True, "redundant_with": [], "reason": "only candidate in this chunk"} for term in terms}

    async with semaphore:
        candidates_block = "\n".join(f"{i}. term: {term!r}" for i, term in enumerate(terms, start=1))
        prompt = _TASK_PROMPT.format(chunk=chunk, candidates_block=candidates_block)
        payload = await call_json_with_retry_async(
            llm_provider,
            "chunk_redundancy",
            prompt=prompt,
            system_prompt=(
                "You identify redundant general-category restatements among resume-skill terms extracted "
                "from the same excerpt, keeping the more specific named instance. Return valid JSON only."
            ),
            temperature=0.1,
            max_tokens=1500,
            json_schema=_REDUNDANCY_JSON_SCHEMA,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        verdicts = (payload or {}).get("verdicts", [])
        result: Dict[str, Dict[str, Any]] = {}
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            redundant_with = item.get("redundant_with", [])
            result[term] = {
                "keep": bool(item.get("keep", True)),
                "redundant_with": [str(s).strip() for s in redundant_with if isinstance(s, (str,)) and str(s).strip()]
                if isinstance(redundant_with, list)
                else [],
                "reason": str(item.get("reason", "")),
            }
        return result


async def check_redundancy_for_chunks(
    llm_provider: LLMProvider,
    chunk_terms: List[Dict[str, Any]],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[Dict[str, Dict[str, Any]]]:
    """Run Stage 3b redundancy checks over each chunk's non-atomic surviving
    candidates concurrently.

    `chunk_terms` is a list of `{"chunk": str, "terms": List[str]}` dicts -
    one per chunk that has >=1 non-atomic Stage-2-included candidate.
    Returns one `{term: {"keep": bool, "redundant_with": List[str], "reason": str}}`
    dict per chunk, same order as `chunk_terms`.

    Terms missing from a chunk's result (e.g. a failed batch) should be
    treated as `keep=True` by callers - a fail-safe toward recall.
    """

    if not chunk_terms:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    return await asyncio.gather(
        *(_check_one_chunk(llm_provider, entry["chunk"], entry["terms"], semaphore) for entry in chunk_terms)
    )
