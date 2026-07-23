"""Phase 1: job-posting requirements extraction.

Reuses the existing production parser pipeline (`parser.factory.parse_posting`)
rather than duplicating extraction logic, per AGENTS.md's "reuse existing
modules instead of creating parallel copies of core infrastructure" and the
dev plan's own Phase 1 task 2 wording. This module only reshapes that
pipeline's output into the narrower `JobRequirements` artifact triage
actually needs (role/seniority/domain/core-nice-to-have requirements plus
matched/missing skill terms for later phases) and persists it as
`requirements.json`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm import LLMProvider
from parser import parse_posting

from tailoring.models import JobRequirements, RequirementSentenceMatch

DEFAULT_SKILLS_CACHE_PATH = Path("data/skills.yaml")

ParsePostingFn = Callable[..., Any]


def _requirement_sentences_from_chunk_verdicts(
    chunk_verdicts: Dict[str, Dict[str, Any]]
) -> List[RequirementSentenceMatch]:
    """Group the parser's own `chunk_verdicts` byproduct by sentence.

    Phase 3.9: `chunk_verdicts` (keyed by raw skill term) already carries an
    exact, parser-derived `chunk` (the one posting sentence that term was
    extracted from) for every extracted term, whether ultimately kept or
    not. Only `included=True` terms are kept here - a discarded/redundant/
    miscategorized term was never a genuine signal about what that
    sentence is asking for. Sentence order follows first-appearance order
    across `chunk_verdicts` (stable, not re-sorted); skill-term order
    within a sentence is also first-appearance order.
    """

    terms_by_sentence: "OrderedDict[str, List[str]]" = OrderedDict()
    for raw_term, verdict in chunk_verdicts.items():
        if not verdict.get("included"):
            continue
        sentence = verdict.get("chunk") or ""
        if not sentence:
            continue
        terms_by_sentence.setdefault(sentence, []).append(raw_term)

    return [
        RequirementSentenceMatch(sentence=sentence, skill_terms=tuple(terms))
        for sentence, terms in terms_by_sentence.items()
    ]


def extract_job_requirements(
    posting_text: str,
    summary_llm_provider: LLMProvider,
    reasoning_llm_provider: LLMProvider,
    skills_cache_path: Path = DEFAULT_SKILLS_CACHE_PATH,
    parse_fn: ParsePostingFn = parse_posting,
    **parse_posting_kwargs: Any,
) -> JobRequirements:
    """Extract `JobRequirements` for one job posting.

    `parse_fn` defaults to the real `parser.factory.parse_posting` but is
    injectable so tests can substitute a fake that returns a canned
    parser-record shape without needing to fake the parser's entire
    multi-stage internal LLM schema chain (chunking/extraction/
    categorization/atomicity/redundancy) - this module only cares about the
    parser's OUTPUT shape, not how it got there.

    Raises `ValueError` if the parser returns no records (e.g. empty
    posting text) - there is nothing meaningful to triage against.
    """

    records = parse_fn(
        posting_text,
        skills_cache_path=skills_cache_path,
        use_llm=True,
        summary_llm_provider=summary_llm_provider,
        reasoning_llm_provider=reasoning_llm_provider,
        **parse_posting_kwargs,
    )
    if not records:
        raise ValueError("parse_posting returned no records for this posting - cannot extract requirements")

    record = records[0]
    posting_summary = record["extraction_debug_samples"][0]["posting_summary"]
    chunk_verdicts = record["extraction_debug_samples"][0].get("chunk_verdicts") or {}

    return JobRequirements(
        role_title=posting_summary["role_title"],
        seniority=posting_summary["seniority"],
        industry_domain=posting_summary["industry_domain"],
        core_requirements=tuple(posting_summary["core_requirements"]),
        nice_to_have=tuple(posting_summary["nice_to_have"]),
        summary_paragraph=posting_summary["summary_paragraph"],
        matched_skills=tuple(record.get("matched_skills") or ()),
        missing_skills=tuple(record.get("missing_skills") or ()),
        parser_provenance={
            "use_llm": True,
            "summary_model": getattr(summary_llm_provider, "model", None),
            "reasoning_model": getattr(reasoning_llm_provider, "model", None),
        },
        requirement_sentences=tuple(_requirement_sentences_from_chunk_verdicts(chunk_verdicts)),
    )


def job_requirements_to_dict(requirements: JobRequirements) -> Dict[str, Any]:
    """`requirements.json` shape."""

    return {
        "role_title": requirements.role_title,
        "seniority": requirements.seniority,
        "industry_domain": requirements.industry_domain,
        "core_requirements": list(requirements.core_requirements),
        "nice_to_have": list(requirements.nice_to_have),
        "summary_paragraph": requirements.summary_paragraph,
        "matched_skills": list(requirements.matched_skills),
        "missing_skills": list(requirements.missing_skills),
        "parser_provenance": dict(requirements.parser_provenance),
        "requirement_sentences": [
            {"sentence": match.sentence, "skill_terms": list(match.skill_terms)}
            for match in requirements.requirement_sentences
        ],
    }


def job_requirements_from_dict(data: Dict[str, Any]) -> JobRequirements:
    """Inverse of `job_requirements_to_dict` - loads a persisted `requirements.json`."""

    return JobRequirements(
        role_title=data["role_title"],
        seniority=data["seniority"],
        industry_domain=data["industry_domain"],
        core_requirements=tuple(data.get("core_requirements") or ()),
        nice_to_have=tuple(data.get("nice_to_have") or ()),
        summary_paragraph=data["summary_paragraph"],
        matched_skills=tuple(data.get("matched_skills") or ()),
        missing_skills=tuple(data.get("missing_skills") or ()),
        parser_provenance=dict(data.get("parser_provenance") or {}),
        requirement_sentences=tuple(
            RequirementSentenceMatch(sentence=item["sentence"], skill_terms=tuple(item.get("skill_terms") or ()))
            for item in (data.get("requirement_sentences") or ())
        ),
    )


def write_requirements_json(requirements: JobRequirements, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(job_requirements_to_dict(requirements), handle, indent=2)


def load_requirements_json(path: Path) -> JobRequirements:
    with path.open("r", encoding="utf-8") as handle:
        return job_requirements_from_dict(json.load(handle))
