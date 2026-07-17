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
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from llm import DEFAULT_MAX_CONCURRENCY, LLMProvider, call_json_with_retry_async

CATEGORIES = ("resume_technical_skill", "degree_or_qualification", "soft_skill", "non_skill")
INCLUDED_CATEGORY = "resume_technical_skill"

_CATEGORY_JSON_SCHEMA = {
    "name": "skill_category_flags",
    "schema": {
        "type": "object",
        "properties": {
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
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: classify EACH candidate term below into EXACTLY ONE of these 4 categories, based on the "
    "excerpt it was extracted from (given as local context).\n"
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
    "applies daily is resume_technical_skill).\n\n"
    "Excerpt (local context):\n{chunk}\n\n"
    "Candidates:\n{candidates_block}"
)


async def _categorize_one_chunk(
    llm_provider: LLMProvider,
    chunk: str,
    terms: List[str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, str]]:
    if not terms:
        return {}
    async with semaphore:
        candidates_block = "\n".join(f"{i}. term: {term!r}" for i, term in enumerate(terms, start=1))
        prompt = _TASK_PROMPT.format(chunk=chunk, candidates_block=candidates_block)
        payload = await call_json_with_retry_async(
            llm_provider,
            "skill_category",
            prompt=prompt,
            system_prompt=(
                "You classify candidate resume-skill terms into one of 4 categories based on their local "
                "context. Return valid JSON only."
            ),
            temperature=0.1,
            max_tokens=1500,
            json_schema=_CATEGORY_JSON_SCHEMA,
        )
        verdicts = (payload or {}).get("verdicts", [])
        result: Dict[str, Dict[str, str]] = {}
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            category = str(item.get("category", "")).strip()
            if term and category in CATEGORIES:
                result[term] = {"category": category, "reason": str(item.get("reason", ""))}
        return result


async def categorize_candidates_for_chunks(
    llm_provider: LLMProvider,
    chunk_terms: List[Dict[str, Any]],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[Dict[str, Dict[str, str]]]:
    """Run Stage 2 categorization over every chunk's candidates concurrently.

    `chunk_terms` is a list of `{"chunk": str, "terms": List[str]}` dicts (one
    per Stage 1 chunk result). Returns one `{term: {"category": ..., "reason": ...}}`
    dict per chunk, same order as `chunk_terms`. Terms missing from a chunk's
    result (e.g. a failed batch) are simply absent - callers should treat a
    missing category as "not resume_technical_skill" (fail-safe toward
    exclusion, since this stage exists specifically to tighten precision).
    """

    if not chunk_terms:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    return await asyncio.gather(
        *(
            _categorize_one_chunk(llm_provider, entry["chunk"], entry["terms"], semaphore)
            for entry in chunk_terms
        )
    )
