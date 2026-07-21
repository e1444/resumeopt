"""Phase 6: slot competition (local candidate ranking) and advisory global
diversity filtering.

Two DISTINCT steps, per the dev plan:

1. `rank_local_candidates` - a DETERMINISTIC, per-project scorer (relevance
   to the job's target skills, fact support, primary-proof specificity,
   and LOCAL primary-proof distinctness) over every verified proposal plus
   every triage-eligible original bullet for that project. Never prunes
   anything (dev plan Task 2: "containing EVERY verified project-level
   alternative and each eligible original point") - ranking only decides
   ORDER/rationale, not membership. Deterministic by design (no LLM call),
   matching `tailoring.claims.rank_core_claim_molecules`'s own precedent
   that ranking "does not need to produce a perfect total order."

2. `build_global_recommendation` - an ADVISORY, resume-WIDE greedy filter
   that walks every project's already-ranked candidates in round-robin
   priority order and recommends AT MOST ONE proposal per project: a
   candidate is skipped (not recommended) only if it is judged to
   represent the SAME primary accomplishment as an already-accepted,
   higher-priority candidate from another project - never for merely
   sharing a skill, framework, or fact. That judgment is delegated to
   `_classify_primary_proof_overlap`, a single narrow LLM classifier
   (yes/no/idk + a required primary_dimension + reasoning) - its own
   output IS the dev plan's Task 5 "overlap validator" explanation, reused
   directly rather than issuing a second, redundant call for the same
   question (per AGENTS.md: "reuse already-computed context ... instead of
   issuing a new LLM call for a redundant judgment"). An `idk` verdict is
   treated as non-overlapping (inclusion-biased), matching this codebase's
   long-established "if in doubt, do NOT exclude" convention (see e.g.
   `parser.parallel_extraction`'s classifier prompts) - but every verdict,
   including `idk`, is still recorded as a `ProofOverlapDecision` for human
   review, never silently defaulted.

   This step is PURELY ADVISORY: it never prunes `SlotCandidateSet.
   verified_proposal_ids` or `eligible_original_bullet_ids`, and never
   feeds into automatic resume mutation (dev plan Task 6) - callers apply
   or ignore `recommended_proposal_id` entirely at their own discretion.

   Edge case: n projects competing for a slot may genuinely have only
   m < n distinct underlying accomplishments (e.g. the same real-world
   achievement got written up as a bullet under 2 different projects).
   No special-case handling is needed for this - the greedy walk simply
   ends up recommending nothing for the (n - m) excess projects once
   every one of their remaining candidates is judged to overlap with an
   already-accepted pick, and `recommendation_reason` records every
   conflict it attempted. As a cheap deterministic sanity net on top of
   that (`_find_duplicate_recommendation_warnings`, in case the LLM
   classifier itself misses an overlap), an exact-text match between 2
   different projects' recommended primary-proof strings still produces
   a plain warning string for human review; it never blocks or removes a
   recommendation, since this whole module is advisory only.

Per the Phase 3.7 hygiene rule, every anchor example in this module's
prompts is a fully invented scenario, not copied or paraphrased from this
phase's own fixtures (tests/evals/tailoring/competition/) or the real
project's data.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm import LLMProvider

from tailoring.models import AnnotatedProposal, ProofOverlapDecision, SlotCandidateSet, SlotTriageResult

# Reasoning-tier classifiers in this pipeline default to "low" (not the
# project-wide "minimal" default) per the Phase 3.6/3.7/5 lesson: "minimal"
# can intermittently fail to credit information already present in its own
# input, a reliability gap this advisory-but-still-consequential judgment
# cannot afford either.
COMPETITION_REASONING_EFFORT = "low"

# Only these 2 triage labels are eligible for replacement (dev plan Task 3
# / Phase 1's own semantics) - `keep`/`idk` bullets are protected, their
# facts are reserved, and they never compete in Phase 6 at all.
_ELIGIBLE_TRIAGE_LABELS = ("candidate_for_replacement", "deprioritize")

_OVERLAP_DIMENSIONS = ("system_boundary", "responsibility", "constraint", "outcome", "evidence_type")

_OVERLAP_JSON_SCHEMA = {
    "name": "primary_proof_overlap_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no", "idk"]},
            "primary_dimension": {"type": "string", "enum": list(_OVERLAP_DIMENSIONS)},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "primary_dimension", "reasoning"],
        "additionalProperties": False,
    },
}

_OVERLAP_SYSTEM_PROMPT = (
    "You check exactly one thing: do these two resume-bullet PRIMARY ACCOMPLISHMENT PROOFS describe the SAME "
    "real accomplishment restated across two different projects, rather than two genuinely different "
    "accomplishments - even if they mention the same tool, skill, or technology?\n\n"
    "Explain your verdict using exactly ONE of these 5 dimensions as the PRIMARY basis for your answer: "
    "`system_boundary` (do they concern the same system/service, or genuinely different ones), "
    "`responsibility` (is the same underlying action/responsibility being claimed), "
    "`constraint` (do they operate under the same limiting condition), "
    "`outcome` (is the same kind of measured result being claimed), or "
    "`evidence_type` (is the same kind of proof/evidence being cited).\n\n"
    "Example (yes): \"Reduced average checkout page load time from 4s to 1.2s by compressing and lazy-loading "
    "product images.\" vs. \"Cut the homepage's load time from 3.5s to 1s by compressing and lazy-loading hero "
    "images.\" -> yes, primary_dimension=responsibility (both claim the same image-compression-and-lazy-loading "
    "technique) and outcome (both a page-load-time-reduction metric) - a different page/project name alone does "
    "not make this a different accomplishment.\n"
    "Example (no): \"Migrated the billing service's database to a managed cloud provider.\" vs. \"Set up "
    "automated integration tests for the billing service's payment webhook.\" -> no, "
    "primary_dimension=responsibility (migrating a database vs. writing integration tests are different "
    "responsibilities), even though both concern the same billing service - a shared system is never enough on "
    "its own.\n\n"
    "Answer `no` if they are genuinely different accomplishments, `yes` if they are the same accomplishment "
    "restated, or `idk` only if you genuinely cannot tell either way."
)


def _normalize_proof(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _score_candidate(
    proposal: AnnotatedProposal,
    primary_proof: str,
    job_target_skills: Sequence[str],
    is_local_duplicate: bool,
) -> float:
    """Deterministic advisory score: relevance (target-skill overlap with
    the job), fact support (citation count), specificity (a weak proxy -
    a more detailed primary_proof string), and LOCAL primary-proof
    distinctness (a proposal whose primary_proof exactly duplicates
    another candidate already in THIS project's own pool scores lower -
    cross-project duplication is a separate, LLM-judged concern handled
    entirely by `build_global_recommendation`, not this function). Does
    not need to produce a perfect total order (dev plan) - only to
    reliably avoid an obviously perverse ranking.
    """

    normalized_job_skills = {skill.strip().lower() for skill in job_target_skills}
    relevance = len({skill.strip().lower() for skill in proposal.target_skills} & normalized_job_skills)
    support = len(proposal.supporting_fact_ids)
    specificity = min(len(primary_proof) / 40.0, 3.0)
    distinctness = 0.0 if is_local_duplicate else 1.0

    return relevance * 3.0 + support * 1.0 + specificity + distinctness * 2.0


def rank_local_candidates(
    project_id: str,
    triage_results: Sequence[SlotTriageResult],
    proposals: Sequence[AnnotatedProposal],
    primary_proof_by_core_claim_id: Dict[str, str],
    job_target_skills: Sequence[str],
) -> SlotCandidateSet:
    """Build one project's `SlotCandidateSet`: every triage-eligible
    original bullet id, plus every one of the project's own verified
    proposals, ordered by `_score_candidate`. Never filters/prunes either
    list - only orders the proposals.
    """

    eligible_bullet_ids = tuple(
        result.bullet_id
        for result in triage_results
        if result.project_id == project_id and result.label in _ELIGIBLE_TRIAGE_LABELS
    )
    project_proposals = [proposal for proposal in proposals if proposal.project_id == project_id]

    proof_by_proposal_id = {
        proposal.id: primary_proof_by_core_claim_id.get(proposal.core_claim_id, "") for proposal in project_proposals
    }
    normalized_proof_counts: Dict[str, int] = {}
    for text in proof_by_proposal_id.values():
        normalized = _normalize_proof(text)
        normalized_proof_counts[normalized] = normalized_proof_counts.get(normalized, 0) + 1

    scored: List[Tuple[AnnotatedProposal, float]] = []
    for proposal in project_proposals:
        own_proof = proof_by_proposal_id[proposal.id]
        own_normalized = _normalize_proof(own_proof)
        is_local_duplicate = own_normalized != "" and normalized_proof_counts[own_normalized] > 1
        score = _score_candidate(proposal, own_proof, job_target_skills, is_local_duplicate)
        scored.append((proposal, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    ranked_proposal_ids = tuple(proposal.id for proposal, _ in scored)

    if ranked_proposal_ids:
        rationale = (
            "Ranked by target-skill relevance, supporting-fact count, primary-proof specificity, and local "
            "primary-proof distinctness (deterministic scorer; not a perfect total order)."
        )
    else:
        rationale = "No verified proposals available for this project - only the original bullets are available."

    return SlotCandidateSet(
        project_id=project_id,
        eligible_original_bullet_ids=eligible_bullet_ids,
        verified_proposal_ids=ranked_proposal_ids,
        ranking_rationale=rationale,
    )


def _classify_primary_proof_overlap(
    proof_a: str,
    proof_b: str,
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    prompt = f'Proof A: "{proof_a}"\n\nProof B: "{proof_b}"\n\nDo these describe the same real accomplishment?'
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_OVERLAP_SYSTEM_PROMPT,
        json_schema=_OVERLAP_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    verdict = response.get("verdict")
    if verdict not in ("yes", "no", "idk"):
        verdict = "idk"
    dimension = response.get("primary_dimension")
    if dimension not in _OVERLAP_DIMENSIONS:
        dimension = None
    return {"verdict": verdict, "primary_dimension": dimension, "reasoning": response.get("reasoning", "")}


def _find_duplicate_recommendation_warnings(
    recommended_by_project: Dict[str, str],
    proposals_by_id: Dict[str, AnnotatedProposal],
    primary_proof_by_core_claim_id: Dict[str, str],
) -> List[str]:
    """Cheap deterministic sanity net, run AFTER the LLM-judged greedy
    filter finishes: flag any 2 different projects' recommended proposals
    whose primary_proof text is IDENTICAL once normalized. This should
    never trigger if _classify_primary_proof_overlap judged every
    accepted pair correctly - it exists only to surface an obvious
    duplicate the classifier missed (e.g. an unexpected no/idk verdict
    for text that is, on its face, the exact same string), which becomes
    more likely as the number of competing projects grows past the
    number of genuinely unique underlying accomplishments. Advisory
    only: never blocks or removes a recommendation, only adds a warning
    string for human review.
    """

    warnings: List[str] = []
    seen_by_normalized_proof: Dict[str, Tuple[str, str]] = {}
    for project_id, proposal_id in recommended_by_project.items():
        proposal = proposals_by_id.get(proposal_id)
        if proposal is None:
            continue
        normalized = _normalize_proof(primary_proof_by_core_claim_id.get(proposal.core_claim_id, ''))
        if not normalized:
            continue
        if normalized in seen_by_normalized_proof:
            other_project_id, other_proposal_id = seen_by_normalized_proof[normalized]
            warnings.append(
                'Possible duplicate recommendation: '
                + f'{proposal_id!r} (project {project_id!r}) and {other_proposal_id!r} '
                + f'(project {other_project_id!r}) have identical primary_proof text after normalization, '
                + 'despite not being classified as overlapping. This is expected when there are fewer '
                + 'genuinely unique underlying accomplishments than projects competing for a slot - review '
                + 'before treating this default recommendation as final.'
            )
        else:
            seen_by_normalized_proof[normalized] = (project_id, proposal_id)
    return warnings


def build_global_recommendation(
    candidate_sets: Sequence[SlotCandidateSet],
    proposals_by_id: Dict[str, AnnotatedProposal],
    primary_proof_by_core_claim_id: Dict[str, str],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = COMPETITION_REASONING_EFFORT,
) -> Tuple[List[SlotCandidateSet], List[ProofOverlapDecision], List[str]]:
    """Advisory resume-wide greedy diversity filter over already-ranked
    SlotCandidateSets. Recommends AT MOST ONE proposal per project;
    never mutates verified_proposal_ids/eligible_original_bullet_ids.

    Returns (updated_candidate_sets, overlap_decisions, duplicate_warnings)
    - duplicate_warnings is the deterministic sanity-net output of
    _find_duplicate_recommendation_warnings (see its docstring); it is
    normally empty and never affects the recommendation itself.
    """

    max_len = max((len(cs.verified_proposal_ids) for cs in candidate_sets), default=0)
    global_priority_order: List[Tuple[str, str]] = []
    for position in range(max_len):
        for candidate_set in candidate_sets:
            if position < len(candidate_set.verified_proposal_ids):
                global_priority_order.append((candidate_set.project_id, candidate_set.verified_proposal_ids[position]))

    accepted: List[Tuple[str, str]] = []
    recommended_by_project: Dict[str, str] = {}
    attempted_conflict_reasons: Dict[str, List[str]] = {}
    decisions: List[ProofOverlapDecision] = []

    for project_id, proposal_id in global_priority_order:
        if project_id in recommended_by_project:
            continue

        proposal = proposals_by_id.get(proposal_id)
        proof = primary_proof_by_core_claim_id.get(proposal.core_claim_id, "") if proposal else ""

        conflict_reason: Optional[str] = None
        for accepted_project_id, accepted_proposal_id in accepted:
            accepted_proposal = proposals_by_id.get(accepted_proposal_id)
            accepted_proof = (
                primary_proof_by_core_claim_id.get(accepted_proposal.core_claim_id, "") if accepted_proposal else ""
            )
            result = _classify_primary_proof_overlap(proof, accepted_proof, llm_provider, reasoning_effort)
            decisions.append(
                ProofOverlapDecision(
                    proposal_id_a=proposal_id,
                    proposal_id_b=accepted_proposal_id,
                    verdict=result["verdict"],
                    primary_dimension=result["primary_dimension"],
                    reasoning=result["reasoning"],
                )
            )
            if result["verdict"] == "yes":
                conflict_reason = (
                    f"Not recommended: overlaps with {accepted_proposal_id} from project "
                    f"'{accepted_project_id}' ({result['primary_dimension']}): {result['reasoning']}"
                )
                break

        if conflict_reason is None:
            accepted.append((project_id, proposal_id))
            recommended_by_project[project_id] = proposal_id
        else:
            attempted_conflict_reasons.setdefault(project_id, []).append(conflict_reason)

    updated_candidate_sets: List[SlotCandidateSet] = []
    for candidate_set in candidate_sets:
        recommended_id = recommended_by_project.get(candidate_set.project_id)
        if recommended_id is not None:
            reason = "Recommended: top-ranked local candidate with no detected primary-proof overlap."
        elif not candidate_set.verified_proposal_ids:
            reason = "No verified proposals available for this project - the original bullet(s) remain the only option."
        elif candidate_set.project_id in attempted_conflict_reasons:
            reason = " ".join(attempted_conflict_reasons[candidate_set.project_id])
        else:
            reason = None
        updated_candidate_sets.append(
            replace(candidate_set, recommended_proposal_id=recommended_id, recommendation_reason=reason)
        )

    duplicate_warnings = _find_duplicate_recommendation_warnings(
        recommended_by_project, proposals_by_id, primary_proof_by_core_claim_id
    )

    return updated_candidate_sets, decisions, duplicate_warnings


def slot_candidate_sets_to_dicts(candidate_sets: Sequence[SlotCandidateSet]) -> List[dict]:
    return [
        {
            "project_id": candidate_set.project_id,
            "eligible_original_bullet_ids": list(candidate_set.eligible_original_bullet_ids),
            "verified_proposal_ids": list(candidate_set.verified_proposal_ids),
            "ranking_rationale": candidate_set.ranking_rationale,
            "recommended_proposal_id": candidate_set.recommended_proposal_id,
            "recommendation_reason": candidate_set.recommendation_reason,
        }
        for candidate_set in candidate_sets
    ]


def write_project_candidate_sets_json(candidate_sets: Sequence[SlotCandidateSet], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slot_candidate_sets_to_dicts(candidate_sets), indent=2), encoding="utf-8")


def overlap_decisions_to_dicts(decisions: Sequence[ProofOverlapDecision]) -> List[dict]:
    return [
        {
            "proposal_id_a": decision.proposal_id_a,
            "proposal_id_b": decision.proposal_id_b,
            "verdict": decision.verdict,
            "primary_dimension": decision.primary_dimension,
            "reasoning": decision.reasoning,
        }
        for decision in decisions
    ]


def write_default_resume_recommendation_json(
    candidate_sets: Sequence[SlotCandidateSet],
    decisions: Sequence[ProofOverlapDecision],
    path: Path,
    duplicate_warnings: Sequence[str] = (),
) -> None:
    """Advisory only (dev plan Task 6) - never feed this into automatic
    resume mutation. Consolidates every project's recommendation decision,
    the full pairwise overlap-decision audit trail, and any deterministic
    duplicate-recommendation warnings (see `_find_duplicate_recommendation_
    warnings`) in one artifact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_sets": slot_candidate_sets_to_dicts(candidate_sets),
        "overlap_decisions": overlap_decisions_to_dicts(decisions),
        "duplicate_warnings": list(duplicate_warnings),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
