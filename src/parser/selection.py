"""Final skill selection and post-parse validation, shared across parser implementations.

This stage applies a deliberate, cache-level curation policy (soft skills are
never shown in the final resume skills section even if a parser matched them)
and validates that selected skills are cache-backed, confident, and grounded
in the posting text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm import LLMProvider
from llm.schemas import GROUNDING_JSON_SCHEMA

from .base import DeterministicPostingParser
from .models import MATCH_PRIORITY

# Canonical skills that are intentionally excluded from the final resume skills
# section even when a parser matches them, because they describe soft skills
# rather than hard/technical skills. This is a curation policy over the cache
# content itself, applied uniformly regardless of which parser produced the match.
SOFT_SKILL_TERM_HINTS = (
    "problem solving",
    "time management",
    "communication",
    "organizational",
    "stakeholder communication",
    "collaboration",
    "collaborating",
    "leadership",
    "framing problems",
    "explaining results and limitations",
)


def select_skills(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one strongest match per canonical skill from parsed chunk records."""

    strongest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for match in record.get("matched_skills", []):
            canonical_name = match.get("canonical_name")
            if not isinstance(canonical_name, str) or not canonical_name.strip():
                continue

            canonical_key = canonical_name.strip().lower()
            if any(hint in canonical_key for hint in SOFT_SKILL_TERM_HINTS):
                continue

            existing = strongest.get(canonical_name)
            if existing is None:
                strongest[canonical_name] = match
                continue

            existing_rank = (
                int(existing.get("relevance_score", 0)),
                float(existing.get("confidence", 0.0)),
                MATCH_PRIORITY.get(str(existing.get("match_type", "related")), 0),
            )
            candidate_rank = (
                int(match.get("relevance_score", 0)),
                float(match.get("confidence", 0.0)),
                MATCH_PRIORITY.get(str(match.get("match_type", "related")), 0),
            )
            if candidate_rank > existing_rank:
                strongest[canonical_name] = match

    selected = list(strongest.values())
    selected.sort(
        key=lambda item: (
            -int(item.get("relevance_score", 0)),
            -float(item.get("confidence", 0.0)),
            item.get("canonical_name", ""),
        )
    )
    return selected


def validate_selected_skills(
    records: Sequence[Dict[str, Any]],
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    min_confidence: float = 0.7,
    max_unique_skills: int = 12,
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Validate final selected skills against cache, confidence, and grounding constraints."""

    selected_skills = select_skills(records)
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)
    canonical_cache = {parser._normalize(skill.name) for skill in parser._skills}
    cache_by_canonical = {parser._normalize(skill.name): skill for skill in parser._skills}
    posting_lower = posting_text.lower()

    if not selected_skills:
        issues.append({"type": "empty_selection", "message": "No skills selected"})

    if len(selected_skills) > max_unique_skills:
        issues.append(
            {
                "type": "too_many_skills",
                "count": len(selected_skills),
                "max_allowed": max_unique_skills,
            }
        )

    for match in selected_skills:
        canonical_name = str(match.get("canonical_name", "")).strip()
        canonical_key = parser._normalize(canonical_name)
        if canonical_key not in canonical_cache:
            issues.append(
                {
                    "type": "unsupported_skill",
                    "canonical_name": canonical_name,
                }
            )
            continue

        confidence = float(match.get("confidence", 0.0))
        if confidence < min_confidence:
            issues.append(
                {
                    "type": "weak_match",
                    "canonical_name": canonical_name,
                    "confidence": confidence,
                    "minimum_confidence": min_confidence,
                }
            )

        evidence = str(match.get("evidence", "")).strip().lower()
        raw_term = str(match.get("raw_term", "")).strip().lower()
        canonical_lower = canonical_name.lower()

        grounded_deterministic = (
            (bool(evidence) and evidence in posting_lower)
            or (bool(raw_term) and raw_term in posting_lower)
            or (bool(canonical_lower) and canonical_lower in posting_lower)
        )

        if grounded_deterministic:
            continue

        llm_grounded = False
        if llm_provider is not None:
            skill_record = cache_by_canonical.get(canonical_key)
            llm_grounded = _llm_validate_skill_grounding(
                llm_provider=llm_provider,
                posting_text=posting_text,
                canonical_name=canonical_name,
                aliases=list(skill_record.aliases) if skill_record else [],
                related=list(skill_record.related) if skill_record else [],
                raw_term=raw_term,
                evidence=evidence,
            )

        if llm_grounded:
            notes.append(
                f"LLM grounding accepted edge-case match for '{canonical_name}'"
            )
            continue

        if not evidence and not raw_term:
            issues.append(
                {
                    "type": "missing_grounding",
                    "canonical_name": canonical_name,
                    "message": "Match lacks evidence and raw term",
                }
            )
        elif (
            evidence
            and evidence not in posting_lower
            and raw_term not in posting_lower
            and canonical_lower not in posting_lower
        ):
            issues.append(
                {
                    "type": "missing_grounding",
                    "canonical_name": canonical_name,
                    "message": "Evidence or term not found in posting text and LLM grounding did not confirm",
                }
            )

    if issues:
        notes.append("Validation failed with one or more issues")
        return {
            "status": "fail",
            "notes": notes,
            "issues": issues,
            "selected_skills": selected_skills,
        }

    notes.append("Selected skills pass cache, confidence, and grounding checks")
    return {
        "status": "pass",
        "notes": notes,
        "issues": [],
        "selected_skills": selected_skills,
    }


def _llm_validate_skill_grounding(
    llm_provider: LLMProvider,
    posting_text: str,
    canonical_name: str,
    aliases: Sequence[str],
    related: Sequence[str],
    raw_term: str,
    evidence: str,
) -> bool:
    """Use an LLM to validate semantic grounding for borderline matches."""

    prompt = (
        "Determine whether the skill is actually supported by the posting text. "
        f"Posting Text:\n{posting_text}\n\n"
        f"Skill Canonical Name: {canonical_name}\n"
        f"Skill Aliases: {list(aliases)}\n"
        f"Skill Related Terms: {list(related)}\n"
        f"Parser Raw Term: {raw_term}\n"
        f"Parser Evidence: {evidence}\n"
        "Consider alias and related-term edge cases such as ipynb indicating Jupyter."
    )

    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt=(
                "You validate grounding for resume skills. "
                "Return valid JSON only and do not infer unsupported skills."
            ),
            temperature=0.1,
            max_tokens=300,
            json_schema=GROUNDING_JSON_SCHEMA,
        )
    except Exception:
        return False

    return bool(payload.get("is_grounded", False))
