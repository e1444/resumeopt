"""Phase 0 source-data validation for the bullet-tailoring baseline resources.

No LLM calls. Validates the shape/integrity of durable, human-authored source
data (`<project>_fact_atoms.yaml`) and the manually-prepared baseline data
(`<project>_bullets.yaml`), and derives triage-based protection state. Per
the dev plan's Phase 0 validation gate:

- a deterministic test resolves every `fact_id` referenced by every baseline
  bullet;
- a deterministic test derives protection consistently (`keep`/`idk`
  protected + reserve facts; `candidate_for_replacement`/`deprioritize`
  eligible, do not reserve facts - unless the bullet's `position` is
  `start`/`end`, which always protects it regardless of triage label);
- invalid source data fails with actionable errors: duplicate ids, unknown
  references, invalid project ownership, non-atomic fact-shape warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence

from tailoring.models import BaselineBullet, FactAtom, Position, ProtectionState, TriageLabel

Severity = Literal["error", "warning"]

_VALID_POSITIONS = {"start", "middle", "end"}
_PROTECTED_LABELS = {"keep", "idk"}
_ELIGIBLE_LABELS = {"candidate_for_replacement", "deprioritize"}

# Deliberately simple, deterministic heuristics (no LLM) for flagging a fact
# atom that may bundle more than one distinct claim - a WARNING, not a hard
# failure, since atomicity is ultimately a human editorial judgment.
_NON_ATOMIC_CONJUNCTIONS = (" and ", "; ", ", and ")

# A period followed by whitespace + an uppercase letter is a genuine
# sentence break; a bare period count would also match decimal numbers
# (e.g. "0.0025 ECE vs. 0.03 baseline"), producing false positives on this
# resume's numeric-heavy facts.
_SENTENCE_BREAK_RE = re.compile(r"\.\s+[A-Z]")


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation finding."""

    severity: Severity
    code: str
    message: str


def validate_fact_atoms(fact_atoms: Sequence[FactAtom]) -> List[ValidationIssue]:
    """Check duplicate ids and non-atomic fact-shape warnings.

    Duplicate ids are a hard error (ambiguous downstream references).
    Non-atomic shape is a warning only.
    """

    issues: List[ValidationIssue] = []
    seen: Dict[str, int] = {}
    for atom in fact_atoms:
        seen[atom.id] = seen.get(atom.id, 0) + 1
        issues.extend(_non_atomic_warnings(atom))

    for fact_id, count in seen.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_fact_id",
                    message=f"fact id '{fact_id}' is defined {count} times; ids must be unique",
                )
            )
    return issues


def _non_atomic_warnings(atom: FactAtom) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    lowered = atom.fact.lower()
    if any(conj in lowered for conj in _NON_ATOMIC_CONJUNCTIONS):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="possibly_non_atomic_fact",
                message=(
                    f"fact '{atom.id}' contains a conjunction ('and'/multiple clauses); "
                    "consider splitting into separate atomic facts"
                ),
            )
        )
    elif _SENTENCE_BREAK_RE.search(atom.fact):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="possibly_non_atomic_fact",
                message=(
                    f"fact '{atom.id}' contains multiple sentences; "
                    "consider splitting into separate atomic facts"
                ),
            )
        )
    return issues


def validate_baseline_bullets(
    bullets: Sequence[BaselineBullet],
    known_fact_ids: Iterable[str],
    expected_project_id: str,
) -> List[ValidationIssue]:
    """Check duplicate bullet ids, invalid positions, wrong project ownership,
    and unknown `fact_id` references against `known_fact_ids`.
    """

    issues: List[ValidationIssue] = []
    known_fact_id_set = set(known_fact_ids)
    seen: Dict[str, int] = {}

    for bullet in bullets:
        seen[bullet.id] = seen.get(bullet.id, 0) + 1

        if bullet.position not in _VALID_POSITIONS:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_position",
                    message=(
                        f"bullet '{bullet.id}' has invalid position "
                        f"'{bullet.position}'; must be one of {sorted(_VALID_POSITIONS)}"
                    ),
                )
            )

        if bullet.project_id != expected_project_id:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_project_ownership",
                    message=(
                        f"bullet '{bullet.id}' declares project_id "
                        f"'{bullet.project_id}', expected '{expected_project_id}'"
                    ),
                )
            )

        for fact_id in bullet.fact_ids:
            if fact_id not in known_fact_id_set:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="unknown_fact_reference",
                        message=(
                            f"bullet '{bullet.id}' references unknown fact id '{fact_id}'"
                        ),
                    )
                )

    for bullet_id, count in seen.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_bullet_id",
                    message=f"bullet id '{bullet_id}' is defined {count} times; ids must be unique",
                )
            )
    return issues


_PROTECTED_POSITIONS = {"start", "end"}


def derive_protection_states(
    bullets: Sequence[BaselineBullet],
    triage_by_bullet_id: Dict[str, TriageLabel],
) -> List[ProtectionState]:
    """Derive protection state for every bullet from its triage label.

    `keep`/`idk` bullets are protected and reserve their linked facts.
    `candidate_for_replacement`/`deprioritize` bullets are eligible and do
    NOT reserve their linked facts (per the dev plan, replaceable points
    don't hold their facts back from generation). A bullet with no triage
    entry at all is treated as protected (fail safe: never reserve nothing,
    never assume eligibility without an explicit label).

    A `start` or `end` positioned bullet is ALWAYS protected, regardless of
    its triage label - this overrides an otherwise-eligible
    `candidate_for_replacement`/`deprioritize` label (per the dev plan's
    "Future consideration" note: an opening point carries scope-setting
    exposition/context and a final point is the last displayed point, and
    neither can be safely substituted by a narrower, technology-specific
    generated replacement). Only `middle` bullets can ever be eligible.
    """

    states: List[ProtectionState] = []
    for bullet in bullets:
        label = triage_by_bullet_id.get(bullet.id)
        protected = bullet.position in _PROTECTED_POSITIONS or label not in _ELIGIBLE_LABELS
        states.append(
            ProtectionState(
                bullet_id=bullet.id,
                project_id=bullet.project_id,
                protected=protected,
                reserved_fact_ids=bullet.fact_ids if protected else (),
            )
        )
    return states


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)
