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
from matcher import MATCH_PRIORITY, LLMGroundingMatcher

from .base import DeterministicPostingParser

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


def select_skills(
    records: Sequence[Dict[str, Any]],
    max_unique_skills: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select one strongest match per canonical skill from parsed chunk records.

    If `max_unique_skills` is given, truncates to the top N after sorting by
    relevance/confidence/match-type strength (already the sort order below) -
    so a posting with more genuine skills than fit in a tight resume section
    keeps its strongest matches rather than failing outright.
    """

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
    if max_unique_skills is not None:
        selected = selected[:max_unique_skills]
    return selected


def validate_selected_skills(
    records: Sequence[Dict[str, Any]],
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    min_confidence: float = 0.7,
    max_unique_skills: int = 12,
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Validate final selected skills against cache, confidence, and grounding constraints.

    `max_unique_skills` truncates the candidate pool to its top N (by relevance/
    confidence/match-type strength) before validating anything else, rather than
    failing outright when a posting has more genuine skills than fit in a tight
    resume section - a posting simply having many real skills isn't itself a
    quality problem to raise on; per-skill confidence/grounding checks still
    apply to whichever skills survive truncation.
    """

    all_selected_skills = select_skills(records)
    selected_skills = select_skills(records, max_unique_skills=max_unique_skills)
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)
    canonical_cache = {parser._normalize(skill.name) for skill in parser._skills}
    cache_by_canonical = {parser._normalize(skill.name): skill for skill in parser._skills}
    posting_lower = posting_text.lower()

    if not selected_skills:
        issues.append({"type": "empty_selection", "message": "No skills selected"})

    if len(all_selected_skills) > max_unique_skills:
        notes.append(
            f"{len(all_selected_skills)} skills matched; kept the strongest "
            f"{max_unique_skills} for a tight resume section, dropped "
            f"{len(all_selected_skills) - max_unique_skills} weaker/lower-relevance match(es)"
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
            llm_grounded = LLMGroundingMatcher(llm_provider).confirm_grounding(
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
