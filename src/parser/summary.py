"""Stage 0: posting-level summary generation.

One LLM call per posting, producing structured global context
(`PostingSummary`) used as shared background context for Stage 1 extraction.
This gives every later per-chunk call a sense of the posting as a whole
(role, seniority, domain, core/nice-to-have requirements) instead of only
ever seeing its own local excerpt.
"""

from __future__ import annotations

from typing import Any, Dict, List

from llm import LLMProvider, call_json_with_retry_async

from .models import PostingSummary

_POSTING_SUMMARY_JSON_SCHEMA = {
    "name": "posting_summary",
    "schema": {
        "type": "object",
        "properties": {
            "role_title": {"type": "string"},
            "seniority": {"type": "string"},
            "industry_domain": {"type": "string"},
            "core_requirements": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
            "summary_paragraph": {"type": "string"},
        },
        "required": [
            "role_title",
            "seniority",
            "industry_domain",
            "core_requirements",
            "nice_to_have",
            "summary_paragraph",
        ],
        "additionalProperties": False,
    },
}

_PROMPT = (
    "Task: read the job posting below and produce one structured summary of it.\n"
    "role_title: the job's title, as stated (best guess if not literally stated).\n"
    "seniority: the seniority/level implied (for example 'entry-level', 'senior', 'unspecified').\n"
    "industry_domain: the business/industry domain the role sits in (for example 'healthcare "
    "technology', 'insurance', 'general software').\n"
    "core_requirements: the explicit MUST-HAVE skills/qualifications, as short phrases using the "
    "posting's own wording where possible - only items clearly stated as required, not every skill "
    "mentioned anywhere in the posting.\n"
    "nice_to_have: explicit OPTIONAL/preferred skills, in the same style as core_requirements.\n"
    "summary_paragraph: 2-4 sentences describing the role, its seniority, and its domain - do not "
    "enumerate every requirement here, that is what core_requirements/nice_to_have are for.\n\n"
    "Job posting:\n{posting_text}"
)


async def generate_posting_summary(llm_provider: LLMProvider, posting_text: str) -> PostingSummary:
    """Stage 0: produce one `PostingSummary` for the whole posting.

    Falls back to an empty-but-valid summary if the call fails, so the rest
    of the pipeline degrades gracefully instead of crashing the whole run
    over a single failed call.
    """

    payload: Dict[str, Any] = (
        await call_json_with_retry_async(
            llm_provider,
            "generate_posting_summary",
            prompt=_PROMPT.format(posting_text=posting_text),
            system_prompt="You summarize job postings into structured briefs. Return valid JSON only.",
            temperature=0.1,
            max_tokens=1200,
            json_schema=_POSTING_SUMMARY_JSON_SCHEMA,
        )
        or {}
    )

    def _str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return PostingSummary(
        role_title=str(payload.get("role_title", "")).strip(),
        seniority=str(payload.get("seniority", "")).strip(),
        industry_domain=str(payload.get("industry_domain", "")).strip(),
        core_requirements=_str_list(payload.get("core_requirements", [])),
        nice_to_have=_str_list(payload.get("nice_to_have", [])),
        summary_paragraph=str(payload.get("summary_paragraph", "")).strip(),
    )


def format_summary_block(posting_summary: PostingSummary) -> str:
    """Render a `PostingSummary` as shared prompt context for Stage 1."""

    core = "\n".join(f"- {item}" for item in posting_summary.core_requirements) or "(none listed)"
    nice = "\n".join(f"- {item}" for item in posting_summary.nice_to_have) or "(none listed)"
    return (
        f"role_title: {posting_summary.role_title}\n"
        f"seniority: {posting_summary.seniority}\n"
        f"industry_domain: {posting_summary.industry_domain}\n"
        f"core_requirements:\n{core}\n"
        f"nice_to_have:\n{nice}\n"
        f"summary_paragraph: {posting_summary.summary_paragraph}"
    )
