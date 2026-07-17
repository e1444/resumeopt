"""Stage 1: per-chunk candidate skill extraction via a reasoning model.

One call per chunk (batched across chunks concurrently via `asyncio.gather`).
Optionally attaches the posting's global summary as context. Deliberately
RECALL-FIRST and over-includes (e.g. degree-enumeration items, redundant
broader-category restatements) - `categorization.py` (Stage 2) is
responsible for precision.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from llm import DEFAULT_MAX_CONCURRENCY, LLMProvider, call_json_with_retry_async

_EXTRACTION_JSON_SCHEMA = {
    "name": "chunk_skill_extraction",
    "schema": {
        "type": "object",
        "properties": {
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
        "required": ["reasoning", "skills"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: read ONLY the excerpt below (a single sentence or short passage from a job posting) and "
    "extract every skill/technique term that should appear in a resume's SKILLS section, based on this "
    "excerpt specifically.\n"
    "Include a term if it is: a real, named technology/tool/platform/programming language/algorithm; an "
    "established academic/technical/scientific discipline; OR a recognized professional practice/"
    "methodology (e.g. code review, unit testing, version control) - AND it is relevant to the role "
    "(explicitly required, a natural expected skill, or a nice-to-have/asset).\n"
    "Do NOT include: soft skills (communication, teamwork, problem-solving); job titles or organization/"
    "product/brand names; compensation/benefit items; a business/industry domain label or company-"
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
    "First write a brief `reasoning` narrative explaining your overall read of this excerpt, THEN list "
    "each extracted `term` with its own short `reason`. If you are unsure whether something belongs, "
    "still include it - a later stage independently re-checks each candidate more strictly.\n"
    "{summary_section}"
    "Excerpt:\n{chunk}"
)


def _build_prompt(chunk: str, summary_block: Optional[str]) -> str:
    summary_section = (
        f"Posting summary (for context, not the excerpt itself):\n{summary_block}\n\n" if summary_block else ""
    )
    return _TASK_PROMPT.format(summary_section=summary_section, chunk=chunk)


async def _extract_one_chunk(
    llm_provider: LLMProvider,
    chunk: str,
    summary_block: Optional[str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        prompt = _build_prompt(chunk, summary_block)
        payload = await call_json_with_retry_async(
            llm_provider,
            "chunk_extraction",
            prompt=prompt,
            system_prompt=(
                "You extract resume-worthy technical skills from a single short job-posting excerpt. "
                "Return valid JSON only."
            ),
            temperature=0.2,
            max_tokens=1500,
            json_schema=_EXTRACTION_JSON_SCHEMA,
        )
        payload = payload or {"reasoning": "", "skills": []}
        skills = payload.get("skills", [])
        terms = [str(item.get("term", "")).strip() for item in skills if isinstance(item, dict)]
        term_reasons = {
            str(item.get("term", "")).strip(): str(item.get("reason", ""))
            for item in skills
            if isinstance(item, dict) and str(item.get("term", "")).strip()
        }
        return {
            "chunk": chunk,
            "reasoning": str(payload.get("reasoning", "")),
            "terms": [term for term in terms if term],
            "term_reasons": term_reasons,
        }


async def extract_candidates_for_chunks(
    llm_provider: LLMProvider,
    chunks: List[str],
    summary_block: Optional[str] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """Run Stage 1 extraction over every chunk concurrently.

    Returns one result dict per chunk (same order as `chunks`):
    `{"chunk": str, "reasoning": str, "terms": List[str], "term_reasons": Dict[str, str]}`.
    """

    if not chunks:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    return await asyncio.gather(
        *(_extract_one_chunk(llm_provider, chunk, summary_block, semaphore) for chunk in chunks)
    )
