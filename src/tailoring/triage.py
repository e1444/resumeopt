"""Phase 1: advisory, non-mutating slot triage.

One narrow LLM call per baseline bullet (per AGENTS.md: "LLM classifier
calls should ask exactly one question per call ... split multi-part
judgments into separate single-purpose calls") - triage never batches
multiple bullets into one call, and never assigns a generated claim to a
bullet. It only labels each EXISTING bullet's replacement eligibility
relative to one job posting's requirements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from llm import DEFAULT_REASONING_EFFORT, LLMProvider

from tailoring.models import BaselineBullet, JobRequirements, ProjectBaseline, SlotTriageResult, TriageLabel

_TRIAGE_LABELS = ("keep", "candidate_for_replacement", "deprioritize", "idk")

_TRIAGE_JSON_SCHEMA = {
    "name": "slot_triage_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(_TRIAGE_LABELS)},
            "job_relevance": {"type": "string"},
            "narrative_value": {"type": "string"},
            "replacement_opportunity": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["label", "job_relevance", "narrative_value", "replacement_opportunity", "reason"],
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT = (
    "You triage ONE existing resume bullet point against ONE job posting's requirements. "
    "You make exactly one judgment: is this bullet still worth keeping as-is for this posting, "
    "or is it a plausible candidate to be replaced by a stronger, more targeted alternative? "
    "You do NOT propose or describe any replacement text - only label this bullet's own "
    "replacement eligibility. Choose exactly one label:\n"
    "- keep: this bullet is clearly relevant and well-aligned with the posting; no need to replace it.\n"
    "- candidate_for_replacement: this bullet is off-topic or weakly relevant for this posting; a "
    "more targeted alternative would likely serve this slot better.\n"
    "- deprioritize: this bullet is not a strong match but also not clearly worth actively "
    "replacing - a weaker, more ambiguous case than candidate_for_replacement.\n"
    "- idk: you are genuinely unsure and neither of the above confidently applies.\n"
    "Your `label` MUST be consistent with your own `job_relevance` and `replacement_opportunity` "
    "assessments - do not pick a more conservative label than your own reasoning supports. "
    "Specifically: if `job_relevance` is low/weak AND `replacement_opportunity` is high/clear, "
    "the label MUST be candidate_for_replacement, not deprioritize or keep. Reserve deprioritize "
    "for cases where relevance or replacement opportunity is genuinely moderate/mixed, not as a "
    "generic hedge against candidate_for_replacement. Reserve idk for genuine uncertainty about "
    "which label applies, not as a way to avoid a confident-sounding call your own reasoning "
    "already supports."
)


def _format_requirements_block(requirements: JobRequirements) -> str:
    core = "\n".join(f"- {item}" for item in requirements.core_requirements) or "(none listed)"
    nice = "\n".join(f"- {item}" for item in requirements.nice_to_have) or "(none listed)"
    return (
        f"Role: {requirements.role_title}\n"
        f"Seniority: {requirements.seniority}\n"
        f"Industry/domain: {requirements.industry_domain}\n"
        f"Summary: {requirements.summary_paragraph}\n\n"
        f"Core requirements:\n{core}\n\n"
        f"Nice to have:\n{nice}"
    )


def _build_prompt(bullet: BaselineBullet, project: ProjectBaseline, requirements: JobRequirements) -> str:
    return (
        f"Job posting requirements:\n{_format_requirements_block(requirements)}\n\n"
        f"Existing resume bullet (from project '{project.project_title}', role context "
        f"'{project.role_context}', structural position '{bullet.position}'):\n"
        f'"{bullet.text}"\n\n'
        "Triage this ONE bullet against this ONE posting's requirements."
    )


def triage_bullet(
    bullet: BaselineBullet,
    project: ProjectBaseline,
    requirements: JobRequirements,
    llm_provider: Optional[LLMProvider] = None,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
) -> SlotTriageResult:
    """Triage one baseline bullet against one job posting's requirements.

    Falls back to a deterministic keyword-overlap heuristic when
    `llm_provider` is `None` (dev plan: "deterministic fallback for
    unavailable LLM providers").
    """

    if llm_provider is None:
        return _deterministic_triage(bullet, requirements)

    response = llm_provider.call_json(
        prompt=_build_prompt(bullet, project, requirements),
        system_prompt=_SYSTEM_PROMPT,
        json_schema=_TRIAGE_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )

    label = response.get("label")
    if label not in _TRIAGE_LABELS:
        label = "idk"

    return SlotTriageResult(
        bullet_id=bullet.id,
        project_id=bullet.project_id,
        label=label,  # type: ignore[arg-type]
        job_relevance=response.get("job_relevance"),
        narrative_value=response.get("narrative_value"),
        replacement_opportunity=response.get("replacement_opportunity"),
        reason=response.get("reason", ""),
    )


def _deterministic_triage(bullet: BaselineBullet, requirements: JobRequirements) -> SlotTriageResult:
    """No-LLM fallback: crude keyword-overlap heuristic.

    Not intended to be accurate - only to keep this module fully functional
    (never raises/blocks) when no LLM provider is configured. Always labels
    `idk` with an explicit reason, EXCEPT when overlap is strong enough to
    be an unambiguous `keep` or there is a real risk that it is a
    completely unrelated bullet - see thresholds below.
    """

    bullet_words = set(bullet.text.lower().split())
    requirement_words = set()
    for item in (*requirements.core_requirements, *requirements.nice_to_have):
        requirement_words.update(item.lower().split())

    overlap = bullet_words & requirement_words
    reason = (
        f"deterministic fallback (no LLM provider configured): "
        f"{len(overlap)} overlapping word(s) with posting requirements: {sorted(overlap)}"
    )

    if len(overlap) >= 3:
        label: TriageLabel = "keep"
    elif len(overlap) == 0:
        label = "candidate_for_replacement"
    else:
        label = "idk"

    return SlotTriageResult(
        bullet_id=bullet.id,
        project_id=bullet.project_id,
        label=label,
        job_relevance=None,
        narrative_value=None,
        replacement_opportunity=None,
        reason=reason,
    )


def triage_project_bullets(
    project: ProjectBaseline,
    requirements: JobRequirements,
    llm_provider: Optional[LLMProvider] = None,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
) -> List[SlotTriageResult]:
    """Triage every bullet in one project, one call per bullet.

    Every baseline bullet gets an explicit triage entry, including
    non-actionable ones, per the dev plan ("write slot_triage.json,
    including every slot and an explicit explanation for non-actionable
    slots").
    """

    return [
        triage_bullet(bullet, project, requirements, llm_provider=llm_provider, reasoning_effort=reasoning_effort)
        for bullet in project.bullets
    ]


def slot_triage_to_dicts(results: Sequence[SlotTriageResult]) -> List[dict]:
    return [
        {
            "bullet_id": result.bullet_id,
            "project_id": result.project_id,
            "label": result.label,
            "job_relevance": result.job_relevance,
            "narrative_value": result.narrative_value,
            "replacement_opportunity": result.replacement_opportunity,
            "reason": result.reason,
        }
        for result in results
    ]


def write_slot_triage_json(results: Sequence[SlotTriageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(slot_triage_to_dicts(results), handle, indent=2)
