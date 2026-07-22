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

# Phase 3.9: live-validated (narrow case + held-out case, gpt-5-mini) that
# "low" effort is unreliable for classify_claim_concreteness specifically -
# it rationalizes routine instrumentation details (e.g. "batch-level
# progress") as differentiating. "medium" fixed this on both the narrow
# and held-out cases without switching model. Deliberately NOT the
# project-wide DEFAULT_REASONING_EFFORT ("minimal") or this module's own
# claim-generation default.
CONCRETENESS_REASONING_EFFORT = "medium"

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
                        "why": {"type": "string"},
                        "result": {"type": "string"},
                    },
                    "required": [
                        "claim_text",
                        "supporting_fact_ids",
                        "target_skills",
                        "primary_proof",
                        "rationale",
                        "why",
                        "result",
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
    "- Before writing claim_text, first identify this claim's NUCLEUS: an abstract `why` (the underlying "
    "motivation or theme this accomplishment serves - for example, a build/release pipeline's why might be "
    "'ensuring reliable, repeatable releases', or an authentication service's why might be 'protecting user "
    "data'). Then decide whether a SEPARATE, concrete `result` exists - a distinct payoff the facts actually "
    "state beyond the why itself (a measured outcome, a clear before/after change, a shipped capability). Many "
    "genuine accomplishments have `why` and `result` collapse into the exact same idea (for example why='making "
    "the system tolerate sudden traffic spikes without failing' with no separate numeric result beyond that "
    "same reliability claim) - this is a common, entirely legitimate outcome, NOT a failure to find one. Return "
    "`result` as an EMPTY STRING whenever no separate result is genuinely stated by the facts - never invent one "
    "just to fill the field. Use this why/result nucleus to frame claim_text around a clear center of gravity "
    "(what this accomplishment fundamentally proves - capability, impact, judgment, or scale) rather than a "
    "flat list of everything the facts state.\n"
    "- If the fact pool has no coherent grouping at all (for example only unrelated, one-off facts), return an "
    "empty claims list. Do not force a claim just to produce output."
)


def _format_fact_pool(fact_atoms: Sequence[FactAtom]) -> str:
    lines = [f'- id={atom.id} | fact="{atom.fact}" | skill_tags={list(atom.skill_tags)}' for atom in fact_atoms]
    return "\n".join(lines)


# Prefixes used for `non_advancement_reason` values that reflect a genuine
# GENERATION-time validation failure (the claim itself is malformed/
# untrustworthy), as opposed to `rank_core_claim_molecules`'s own
# "not_selected_this_round" marker, which just means "valid but not chosen
# this round" and must remain eligible for a future rerank. Keeping this as
# an explicit, checkable set (rather than treating ANY non-None reason as
# permanently invalid) is what makes reranking idempotent.
_UNSUPPORTED_FACT_IDS_PREFIX = "unsupported_fact_ids:"
_INVALID_SUPPORTING_FACT_COUNT_PREFIX = "invalid_supporting_fact_count:"
GENERATION_VALIDATION_FAILURE_PREFIXES: Tuple[str, ...] = (
    _UNSUPPORTED_FACT_IDS_PREFIX,
    _INVALID_SUPPORTING_FACT_COUNT_PREFIX,
)


def is_generation_validation_failure(claim: CoreClaimMolecule) -> bool:
    """True only for a genuine generation-time validation failure (unknown
    fact ids, or an out-of-bounds supporting-fact count) - NOT for a merely
    advisory reason such as `rank_core_claim_molecules`'s own
    "not_selected_this_round" marker.
    """

    return bool(claim.non_advancement_reason) and claim.non_advancement_reason.startswith(
        GENERATION_VALIDATION_FAILURE_PREFIXES
    )


def generate_core_claim_molecules(
    project_id: str,
    fact_atoms: Sequence[FactAtom],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
    requirement_sentence: Optional[str] = None,
) -> List[CoreClaimMolecule]:
    """Discover 0-6 coherent claim molecules from one project's bounded fact pool.

    Every returned claim's `supporting_fact_ids` are de-duplicated (order
    preserved) and validated against `fact_atoms` and the configured
    min/max supporting-fact bounds - a claim citing any unknown fact id, or
    citing an out-of-bounds number of (deduplicated) facts, is marked with
    a specific `non_advancement_reason` (not silently trusted, not
    silently dropped, and not folded into a generic reason later), per
    this project's "never silently drop" convention.

    Phase 3.9: `requirement_sentence`, when given, is the ONE job-posting
    requirement this `fact_atoms` pool was matched against (e.g. one
    sentence from a posting's own requirement/responsibility lines). This
    is passed as grounding context, not a new schema or a separate forced-
    single-nucleus judgment - the SAME 0-N-claim, atomicity-preserving
    generation runs as usual, but now aware of what specific requirement
    the pool was matched for, so it can (a) exclude a matched-but-actually-
    unrelated fact instead of stitching it into an off-theme claim, and
    (b) address only the genuinely-supported part of a compound/multi-part
    requirement rather than force-claiming a part with no evidence. Live-
    validated: a broad posting sentence matching facts from 4 unrelated
    sub-systems correctly dropped the one genuinely unrelated cluster
    entirely (rather than needing a separate post-hoc relevance filter),
    and a compound "messaging AND asynchronous processing" requirement
    with facts supporting only the second half produced a claim addressing
    only that half, never fabricating the messaging part. `None` (the
    default) reproduces the exact prior prompt/behavior unchanged.
    """

    if not fact_atoms:
        return []

    known_fact_ids = {atom.id for atom in fact_atoms}
    if requirement_sentence:
        prompt = (
            f'This fact pool was matched against ONE job-posting requirement: "{requirement_sentence}"\n\n'
            f"Project fact pool ({len(fact_atoms)} facts):\n{_format_fact_pool(fact_atoms)}\n\n"
            "Discover coherent accomplishment claims from this fact pool that genuinely address this specific "
            "requirement. Do not include a fact, or force a claim, just because it was matched into this pool - "
            "some matched facts may turn out not to genuinely relate to this specific requirement. If the "
            "requirement has multiple distinct parts and the facts only genuinely support some of them, address "
            "only the genuinely-supported part(s) - never claim a part with no supporting evidence."
        )
    else:
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
        # dict.fromkeys de-duplicates while preserving first-seen order.
        supporting_fact_ids = tuple(dict.fromkeys(raw_claim.get("supporting_fact_ids") or ()))
        unknown_ids = [fact_id for fact_id in supporting_fact_ids if fact_id not in known_fact_ids]
        molecule_id = f"{project_id}_claim_{index + 1:02d}"

        if unknown_ids:
            non_advancement_reason: Optional[str] = f"{_UNSUPPORTED_FACT_IDS_PREFIX}{unknown_ids}"
        elif not (MIN_SUPPORTING_FACTS <= len(supporting_fact_ids) <= MAX_SUPPORTING_FACTS):
            non_advancement_reason = f"{_INVALID_SUPPORTING_FACT_COUNT_PREFIX}{len(supporting_fact_ids)}"
        else:
            non_advancement_reason = None

        molecules.append(
            CoreClaimMolecule(
                id=molecule_id,
                project_id=project_id,
                claim_text=raw_claim.get("claim_text", ""),
                supporting_fact_ids=supporting_fact_ids,
                target_skills=tuple(raw_claim.get("target_skills") or ()),
                primary_proof=raw_claim.get("primary_proof", ""),
                rationale=raw_claim.get("rationale", ""),
                non_advancement_reason=non_advancement_reason,
                why=raw_claim.get("why", ""),
                result=raw_claim.get("result", ""),
            )
        )

    return molecules


_CONCRETENESS_JSON_SCHEMA = {
    "name": "claim_concreteness",
    "schema": {
        "type": "object",
        "properties": {
            "concrete": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["concrete", "reasoning"],
        "additionalProperties": False,
    },
}

_CONCRETENESS_SYSTEM_PROMPT = (
    "You check exactly one thing: does this resume-bullet claim's own evidence include something concrete and "
    "differentiating - a specific metric, quantified outcome, named distinguishing technique or process, or "
    "otherwise memorable detail - rather than only describing routine, expected engineering practice with no "
    "distinguishing specifics?\n\n"
    "Answer `concrete=true` if the evidence would make this bullet stand out from other similar candidates "
    "(this can be a hard number, OR a specific, non-generic process/technique that most candidates would not "
    "otherwise be able to claim). Answer `concrete=false` if it only describes routine, expected practice "
    "(e.g. 'added progress tracking', 'wrote tests') with no distinguishing detail beyond the fact that it was "
    "done at all."
)


def classify_claim_concreteness(
    claim: CoreClaimMolecule,
    fact_atoms_by_id: dict,
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = CONCRETENESS_REASONING_EFFORT,
) -> bool:
    """Phase 3.9: does this claim's own evidence include something concrete
    and differentiating, as opposed to only routine/generic practice?

    This is a QUALITATIVE signal, not a fact-count threshold - live
    validation (narrow case + a held-out case from a different sentence)
    found that fact count does not reliably predict this: a 2-fact claim
    citing a hard metric was correctly judged concrete, while a
    same-count claim citing only generic instrumentation details was not.
    Intended as an advisory ranking input (e.g. into `_score_claim`), not
    a hard filter - a thin-but-generic claim should simply rank lower,
    never be silently discarded.

    `reasoning_effort` defaults to `CONCRETENESS_REASONING_EFFORT`
    ("medium") specifically - live-validated that "low" is unreliable for
    this exact judgment (it rationalized routine instrumentation wording
    as differentiating on a real example), while "medium" resolved it on
    both a narrow and a held-out case without changing model.
    """

    fact_texts = [
        fact_atoms_by_id[fact_id].fact for fact_id in claim.supporting_fact_ids if fact_id in fact_atoms_by_id
    ]
    prompt = f'Claim why: "{claim.why}"\nClaim result: "{claim.result}"\n\nCited facts:\n{_format_fact_list(fact_texts)}'
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_CONCRETENESS_SYSTEM_PROMPT,
        json_schema=_CONCRETENESS_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    return bool(response.get("concrete"))


def _format_fact_list(fact_texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in fact_texts) or "(none)"


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
    repeat only while non-overlapping facts remain"). Only claims that
    failed GENERATION validation (`is_generation_validation_failure` -
    unsupported fact ids or an out-of-bounds supporting-fact count) are
    left untouched and permanently excluded from candidates; every other
    claim, including one this function itself previously marked
    `"not_selected_this_round"`, remains eligible. Returns every input
    claim, so non-advancing candidates stay visible per the dev plan's
    persistence requirement.

    This is safe to call repeatedly on the SAME claim set (e.g. after
    `max_selected` changes) - it is idempotent, not one-way: a
    previously-selected claim that loses its slot has its `rank` cleared
    (not left stale) alongside the new `"not_selected_this_round"` reason,
    and a previously-not-selected claim that gets selected this time has
    its old `non_advancement_reason` cleared alongside its new `rank`.
    """

    invalid_ids = {claim.id for claim in claims if is_generation_validation_failure(claim)}
    candidates = [claim for claim in claims if claim.id not in invalid_ids]
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
        if claim.id in invalid_ids:
            results.append(claim)
        elif claim.id in selected_rank_by_id:
            results.append(replace(claim, rank=selected_rank_by_id[claim.id], non_advancement_reason=None))
        else:
            results.append(replace(claim, rank=None, non_advancement_reason="not_selected_this_round"))
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
            "why": claim.why,
            "result": claim.result,
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
