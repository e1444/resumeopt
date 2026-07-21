"""Phase 3: local claim proposal and ranking.

ONE structured generation call over a project's whole bounded fact pool
(not per-slot, not per-fact) discovers 0-6 coherent claim molecules, each
citing its own supporting fact ids, per the dev plan's claim-generation
prompt contract. Ranking/selection is a separate, DETERMINISTIC step (the
dev plan permits "a separate single-purpose step or deterministic scorer" -
deterministic chosen here: cheaper, no extra LLM calls, and ranking only
needs to reliably promote 1-2 viable candidates, not a perfect total
order).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from llm import DEFAULT_REASONING_EFFORT, LLMProvider

from tailoring.models import CoreClaimMolecule, FactAtom

MIN_CLAIMS = 0
MAX_CLAIMS = 6
MIN_SUPPORTING_FACTS = 1
MAX_SUPPORTING_FACTS = 10
DEFAULT_MAX_SELECTED = 2

_CLAIM_GENERATION_JSON_SCHEMA = {
    "name": "core_claim_molecules",
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "maxItems": MAX_CLAIMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string"},
                        "supporting_fact_ids": {"type": "array", "items": {"type": "string"}},
                        "target_skills": {"type": "array", "items": {"type": "string"}},
                        "primary_proof": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "claim_text",
                        "supporting_fact_ids",
                        "target_skills",
                        "primary_proof",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT = (
    "You discover coherent, resume-bullet-worthy accomplishment CLAIMS by grouping a project's own fact atoms. "
    "You do NOT judge these claims against any particular job posting or resume slot - you only find genuinely "
    "coherent groupings within the supplied facts.\n\n"
    "Rules:\n"
    f"- Propose {MIN_CLAIMS} to {MAX_CLAIMS} claims. Prefer FEWER, NARROWER claims over broad synthesis - "
    "return a smaller claim, or no claims at all, rather than forcing facts together that don't genuinely belong.\n"
    f"- Each claim must be directly supported by {MIN_SUPPORTING_FACTS} to {MAX_SUPPORTING_FACTS} of the given "
    "fact atoms, cited by their exact given id. Stay close to the facts' own wording; do not infer unstated "
    "details, numbers, or outcomes.\n"
    "- A claim represents exactly ONE accomplishment. Two or more facts belong in the same claim only if they "
    "combine to describe a single concrete deliverable or outcome that could not be truthfully claimed from "
    "either fact alone - not merely because they share a similar general activity category, theme, or domain. "
    "A shared category is a WEAK signal and is NEVER sufficient by itself; ask whether the facts jointly "
    "produced ONE identifiable thing (a system, a result, a shipped artifact), not just whether they FEEL "
    "similar in kind. This applies equally to substantive technical work and to minor/personal facts - do not "
    "relax this rule just because the facts are small or informal. For example: a frontend UI accomplishment "
    "and a separate backend infrastructure accomplishment must never be combined into one claim, even if both "
    "are in the same pool. Two administrative or interpersonal tasks that only share a loose category (helping "
    "plan a team offsite and reviewing internship applications - both broadly \"team-oriented\" but producing "
    "no shared deliverable) must NOT be merged. Three unrelated personal hobbies (restoring a bicycle, "
    "reupholstering a chair, taking a pottery class - each its own separate, unconnected activity with no "
    "combined output) must NOT be merged into one \"hobbies\" claim just because they are all hobbies. In every "
    "such case, treat each fact as its own thin, standalone claim or omit it, rather than inventing a combined "
    "narrative for facts that never actually produced anything together.\n"
    "- Do not reuse the same fact id as a primary basis for more than one claim.\n"
    "- For each claim, also return: target_skills (the skills this claim demonstrates), primary_proof (the "
    "single strongest piece of concrete evidence for this claim - a metric, named tool, or specific outcome), "
    "and a brief rationale explaining why these particular facts belong together as one accomplishment.\n"
    "- If the fact pool has no coherent grouping at all (for example only unrelated, one-off facts), return an "
    "empty claims list. Do not force a claim just to produce output."
)


def _format_fact_pool(fact_atoms: Sequence[FactAtom]) -> str:
    lines = [f'- id={atom.id} | fact="{atom.fact}" | skill_tags={list(atom.skill_tags)}' for atom in fact_atoms]
    return "\n".join(lines)


def generate_core_claim_molecules(
    project_id: str,
    fact_atoms: Sequence[FactAtom],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
) -> List[CoreClaimMolecule]:
    """Discover 0-6 coherent claim molecules from one project's bounded fact pool.

    Every returned claim's `supporting_fact_ids` are validated against
    `fact_atoms` - a claim citing any unknown fact id is marked with a
    `non_advancement_reason` (not silently trusted, not silently dropped),
    per this project's "never silently drop" convention.
    """

    if not fact_atoms:
        return []

    known_fact_ids = {atom.id for atom in fact_atoms}
    prompt = (
        f"Project fact pool ({len(fact_atoms)} facts):\n{_format_fact_pool(fact_atoms)}\n\n"
        "Discover coherent accomplishment claims from this fact pool."
    )

    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        json_schema=_CLAIM_GENERATION_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )

    raw_claims = response.get("claims") or []
    molecules: List[CoreClaimMolecule] = []
    for index, raw_claim in enumerate(raw_claims):
        supporting_fact_ids = tuple(raw_claim.get("supporting_fact_ids") or ())
        unknown_ids = [fact_id for fact_id in supporting_fact_ids if fact_id not in known_fact_ids]
        molecule_id = f"{project_id}_claim_{index + 1:02d}"

        molecules.append(
            CoreClaimMolecule(
                id=molecule_id,
                project_id=project_id,
                claim_text=raw_claim.get("claim_text", ""),
                supporting_fact_ids=supporting_fact_ids,
                target_skills=tuple(raw_claim.get("target_skills") or ()),
                primary_proof=raw_claim.get("primary_proof", ""),
                rationale=raw_claim.get("rationale", ""),
                non_advancement_reason=(f"unsupported_fact_ids:{unknown_ids}" if unknown_ids else None),
            )
        )

    return molecules


def _score_claim(claim: CoreClaimMolecule, reserved_fact_ids: Set[str]) -> float:
    """Deterministic advisory score - direct skill coverage, fact support,
    narrowness, local novelty (unused-fact fraction), and expansion
    headroom. Does not need to produce a perfect total order (dev plan);
    it only needs to reliably surface 1-2 viable candidates.
    """

    available_facts = [fact_id for fact_id in claim.supporting_fact_ids if fact_id not in reserved_fact_ids]
    if not available_facts:
        return float("-inf")

    skill_coverage = len(set(claim.target_skills))
    fact_support = len(claim.supporting_fact_ids)
    novelty_fraction = len(available_facts) / max(fact_support, 1)
    # Prefer narrower groupings (per the dev plan) without penalizing a
    # claim that still legitimately needs several facts to be supported.
    narrowness_bonus = 1.0 if fact_support <= 4 else 0.5
    expansion_headroom = MAX_SUPPORTING_FACTS - fact_support

    return (
        skill_coverage * 2.0
        + fact_support * 1.0
        + novelty_fraction * 3.0
        + narrowness_bonus
        + expansion_headroom * 0.1
    )


def rank_core_claim_molecules(
    claims: Sequence[CoreClaimMolecule],
    max_selected: int = DEFAULT_MAX_SELECTED,
) -> List[CoreClaimMolecule]:
    """Deterministically rank/select project-level claim molecules.

    Greedily accepts up to `max_selected` claims, reserving each accepted
    claim's supporting facts before scoring the remaining candidates - a
    second accepted claim must still have non-overlapping fact support
    left (dev plan: "reserve primary facts after an accepted candidate and
    repeat only while non-overlapping facts remain"). Claims that already
    failed generation validation (`non_advancement_reason` set at
    generation time, e.g. unsupported fact ids) are left untouched and
    never selected. Returns every input claim, so non-advancing candidates
    stay visible per the dev plan's persistence requirement.
    """

    already_invalid_ids = {claim.id for claim in claims if claim.non_advancement_reason}
    candidates = [claim for claim in claims if claim.id not in already_invalid_ids]
    reserved_fact_ids: Set[str] = set()
    selected: List[CoreClaimMolecule] = []

    while candidates and len(selected) < max_selected:
        scored: List[Tuple[CoreClaimMolecule, float]] = [
            (claim, _score_claim(claim, reserved_fact_ids)) for claim in candidates
        ]
        scored = [pair for pair in scored if pair[1] > float("-inf")]
        if not scored:
            break
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best, _ = scored[0]
        selected.append(best)
        reserved_fact_ids.update(best.supporting_fact_ids)
        candidates = [claim for claim in candidates if claim.id != best.id]

    selected_rank_by_id = {claim.id: rank for rank, claim in enumerate(selected, start=1)}

    results: List[CoreClaimMolecule] = []
    for claim in claims:
        if claim.id in already_invalid_ids:
            results.append(claim)
        elif claim.id in selected_rank_by_id:
            results.append(replace(claim, rank=selected_rank_by_id[claim.id]))
        else:
            results.append(replace(claim, non_advancement_reason="not_selected_this_round"))
    return results


def core_claim_molecules_to_dicts(claims: Sequence[CoreClaimMolecule]) -> List[dict]:
    return [
        {
            "id": claim.id,
            "project_id": claim.project_id,
            "claim_text": claim.claim_text,
            "supporting_fact_ids": list(claim.supporting_fact_ids),
            "target_skills": list(claim.target_skills),
            "primary_proof": claim.primary_proof,
            "rationale": claim.rationale,
            "rank": claim.rank,
            "non_advancement_reason": claim.non_advancement_reason,
        }
        for claim in claims
    ]


def write_unranked_core_claim_molecules_json(claims: Sequence[CoreClaimMolecule], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(core_claim_molecules_to_dicts(claims), handle, indent=2)


def write_core_claim_molecules_json(claims: Sequence[CoreClaimMolecule], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(core_claim_molecules_to_dicts(claims), handle, indent=2)
