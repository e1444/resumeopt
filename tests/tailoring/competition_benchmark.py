"""Live validation: slot competition ranking + advisory global diversity
filter (Phase 6).

Run via `python -m tests.tailoring.competition_benchmark` from the repo
root (PYTHONPATH=src). Makes real, billed gpt-5-mini calls
(`reasoning_effort="low"`, per the Phase 3.6/3.7/5 lesson).

Uses `tests/evals/tailoring/competition/competition_scenario.yaml`'s
pre-authored (human-reviewed, approved) fixture directly - per AGENTS.md's
"test each module directly with its persisted/fixture inputs" - not a
live upstream pipeline run. Checks:

1. Local candidate-set membership (eligible bullets, verified proposal
   membership) per `expected_outcomes.yaml` section 1 - exact/mechanical.
2. The 4 named cross-project primary-proof overlap pairs, each run
   OVERLAP_TRIALS times directly against `_classify_primary_proof_overlap`,
   printed alongside the fixture's expected verdict (including the
   deliberately AMBIGUOUS pair, where either verdict is acceptable) - for
   manual/human inspection, not a hard mechanical gate.
3. The full `build_global_recommendation` greedy walk's hard constraints
   from `expected_outcomes.yaml` section 3 - mechanically checked where
   the fixture states an unconditional requirement (never recommend both
   duplicate onboarding proposals; never drop a non-conflicting proposal;
   never penalize the shared-skill-only pairs; every eligible original
   bullet remains selectable).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import yaml
from llm import get_llm_provider

from tailoring.competition import (
    _classify_primary_proof_overlap,
    build_global_recommendation,
    overlap_decisions_to_dicts,
    rank_local_candidates,
    slot_candidate_sets_to_dicts,
)
from tailoring.models import AnnotatedProposal, SlotCandidateSet, SlotTriageResult

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "competition"
SCENARIO_PATH = FIXTURE_DIR / "competition_scenario.yaml"
EXPECTED_PATH = FIXTURE_DIR / "expected_outcomes.yaml"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase6_competition_benchmark.json"

REASONING_MODEL = "gpt-5-mini"
OVERLAP_TRIALS = 3


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_scenario(scenario: Dict[str, Any]):
    job_target_skills = scenario["job_target_skills"]
    triage_results: List[SlotTriageResult] = []
    proposals: List[AnnotatedProposal] = []
    primary_proof_by_core_claim_id: Dict[str, str] = {}
    proposals_by_id: Dict[str, AnnotatedProposal] = {}

    for project in scenario["projects"]:
        project_id = project["project_id"]
        for bullet in project["baseline_bullets"]:
            triage_results.append(
                SlotTriageResult(bullet_id=bullet["id"], project_id=project_id, label=bullet["triage_label"], reason="")
            )
        for raw_proposal in project["verified_proposals"]:
            proposal = AnnotatedProposal(
                id=raw_proposal["id"],
                project_id=project_id,
                core_claim_id=raw_proposal["core_claim_id"],
                proposal_text=raw_proposal["proposal_text"].strip(),
                supporting_fact_ids=tuple(raw_proposal["supporting_fact_ids"]),
                target_skills=tuple(raw_proposal["target_skills"]),
            )
            proposals.append(proposal)
            proposals_by_id[proposal.id] = proposal
            primary_proof_by_core_claim_id[proposal.core_claim_id] = raw_proposal["primary_proof"].strip()

    return job_target_skills, triage_results, proposals, primary_proof_by_core_claim_id, proposals_by_id


def _check_local_candidate_sets(scenario, expected, candidate_sets) -> None:
    print("\n=== Part 1: local candidate-set membership ===")
    expected_by_project = {entry["project_id"]: entry for entry in expected["local_candidate_sets"]}
    for candidate_set in candidate_sets:
        exp = expected_by_project[candidate_set.project_id]
        eligible_match = set(candidate_set.eligible_original_bullet_ids) == set(exp["eligible_original_bullet_ids"])
        proposals_match = set(candidate_set.verified_proposal_ids) == set(exp["verified_proposal_ids"])
        print(f"--- {candidate_set.project_id} ---")
        print(f"eligible_original_bullet_ids: {list(candidate_set.eligible_original_bullet_ids)} -> {'PASS' if eligible_match else 'FAIL'}")
        print(f"verified_proposal_ids (set): {sorted(candidate_set.verified_proposal_ids)} -> {'PASS' if proposals_match else 'FAIL'}")
        print(f"ranked order (not fixture-pinned, for inspection): {list(candidate_set.verified_proposal_ids)}")


def _check_overlap_pairs(scenario, expected, reasoning_llm) -> List[Dict[str, Any]]:
    print("\n=== Part 2: named cross-project primary-proof overlap pairs ===")
    proof_by_proposal_id: Dict[str, str] = {}
    for project in scenario["projects"]:
        for raw_proposal in project["verified_proposals"]:
            proof_by_proposal_id[raw_proposal["id"]] = raw_proposal["primary_proof"].strip()

    results = []
    for pair_case in expected["primary_proof_overlap_pairs"]:
        id_a, id_b = pair_case["pair"]
        proof_a, proof_b = proof_by_proposal_id[id_a], proof_by_proposal_id[id_b]
        print(f"\n--- {id_a} vs {id_b} (expected overlap={pair_case['overlap']!r}) ---")
        trial_verdicts = []
        for _ in range(OVERLAP_TRIALS):
            verdict = _classify_primary_proof_overlap(proof_a, proof_b, reasoning_llm, "low")
            trial_verdicts.append(verdict)
            print(f"  verdict={verdict['verdict']!r} dimension={verdict['primary_dimension']!r} - {verdict['reasoning'][:160]}")
        results.append({"pair": [id_a, id_b], "expected_overlap": pair_case["overlap"], "trials": trial_verdicts})
    return results


def _check_global_recommendation(scenario, expected, candidate_sets, proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm):
    print("\n=== Part 3: advisory global greedy recommendation (full ranked candidate sets) ===")
    calls_before = reasoning_llm.usage_totals["call_count"]
    updated_sets, decisions, warnings = build_global_recommendation(
        candidate_sets, proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm, "low"
    )
    calls_used = reasoning_llm.usage_totals["call_count"] - calls_before
    print(f"llm calls used: {calls_used}")

    recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated_sets}
    print(f"recommendations: {recommended}")
    for decision in decisions:
        print(f"  overlap decision: {decision.proposal_id_a} vs {decision.proposal_id_b} -> {decision.verdict} ({decision.primary_dimension})")
    if warnings:
        for warning in warnings:
            print(f"  duplicate-recommendation warning: {warning}")
    else:
        print("  duplicate-recommendation warnings: none")

    onboarding_ids = {"itp_proposal_onboarding", "cad_proposal_onboarding"}
    recommended_ids = set(v for v in recommended.values() if v is not None)
    both_onboarding_recommended = onboarding_ids <= recommended_ids
    print(
        f"hard constraint 'never both onboarding proposals recommended': "
        f"{'FAIL' if both_onboarding_recommended else 'PASS'}"
    )

    ci_penalized = any(
        d.verdict == "yes" and {d.proposal_id_a, d.proposal_id_b} == {"itp_proposal_k8s", "cad_proposal_k8s_ci"}
        for d in decisions
    ) or any(
        d.verdict == "yes" and {d.proposal_id_a, d.proposal_id_b} == {"cad_proposal_k8s_migration", "cad_proposal_k8s_ci"}
        for d in decisions
    )
    print(f"hard constraint 'cad_proposal_k8s_ci never penalized for shared kubernetes tag': {'FAIL' if ci_penalized else 'PASS'}")

    eligible_preserved = all(
        set(updated.eligible_original_bullet_ids) == set(original.eligible_original_bullet_ids)
        for updated, original in zip(updated_sets, candidate_sets)
    )
    print(f"hard constraint 'every eligible original bullet remains selectable': {'PASS' if eligible_preserved else 'FAIL'}")

    return updated_sets, decisions, warnings


def _check_forced_onboarding_conflict(proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm) -> Dict[str, Any]:
    """On the full ranked fixture, both onboarding proposals rank LAST in
    their own project's local pool (their primary_proof text is shorter/
    less detailed than the k8s/dashboard proposals, so the deterministic
    specificity score naturally deprioritizes them) - meaning the full
    Part 3 walk never actually needs to resolve a real conflict, since
    each project's TOP candidate already succeeds. This forces the
    adversarial shape the fixture was specifically designed to test: give
    each project ONLY its onboarding proposal as its sole candidate, so
    the greedy walk MUST compare them directly against a real LLM call.
    """

    print("\n=== Part 4: forced conflict - both projects' ONLY candidate is the duplicate onboarding proposal ===")
    forced_sets = [
        SlotCandidateSet(project_id="internal_tooling_platform", verified_proposal_ids=("itp_proposal_onboarding",)),
        SlotCandidateSet(project_id="customer_analytics_dashboard", verified_proposal_ids=("cad_proposal_onboarding",)),
    ]
    calls_before = reasoning_llm.usage_totals["call_count"]
    updated_sets, decisions, warnings = build_global_recommendation(
        forced_sets, proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm, "low"
    )
    calls_used = reasoning_llm.usage_totals["call_count"] - calls_before
    recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated_sets}
    print(f"llm calls used: {calls_used}")
    print(f"recommendations: {recommended}")
    for decision in decisions:
        print(f"  overlap decision: {decision.proposal_id_a} vs {decision.proposal_id_b} -> {decision.verdict} ({decision.primary_dimension}) - {decision.reasoning[:160]}")
    if warnings:
        for warning in warnings:
            print(f"  duplicate-recommendation warning: {warning}")
    else:
        print("  duplicate-recommendation warnings: none")

    recommended_ids = [v for v in recommended.values() if v is not None]
    exactly_one = len(recommended_ids) == 1
    print(f"hard constraint 'exactly one of the two duplicate onboarding proposals is recommended': {'PASS' if exactly_one else 'FAIL'}")
    return {
        "recommendations": recommended,
        "decisions": overlap_decisions_to_dicts(decisions),
        "warnings": warnings,
        "calls_used": calls_used,
    }


def _check_three_projects_two_unique_points(proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm) -> Dict[str, Any]:
    """Live check for the n-projects-but-m<n-unique-points edge case
    (n=3 projects, only m=2 genuinely distinct underlying
    accomplishments): a THIRD synthetic project is added alongside the
    2 real duplicate onboarding proposals, with its own candidate being
    a genuinely distinct proposal reused from the fixture. No special
    handling is required for this to work correctly - the third project
    should get its own recommendation untouched, while exactly one of
    the 2 duplicate onboarding proposals is recommended (never both,
    never a crash), and the deterministic duplicate-warning sanity net
    should report nothing extra since the classifier itself already
    correctly excludes the loser.
    """

    print("\n=== Part 5: n=3 projects, m=2 unique points (third project's candidate is genuinely distinct) ===")
    forced_sets = [
        SlotCandidateSet(project_id="internal_tooling_platform", verified_proposal_ids=("itp_proposal_onboarding",)),
        SlotCandidateSet(project_id="customer_analytics_dashboard", verified_proposal_ids=("cad_proposal_onboarding",)),
        SlotCandidateSet(project_id="synthetic_third_project", verified_proposal_ids=("itp_proposal_flags",)),
    ]
    calls_before = reasoning_llm.usage_totals["call_count"]
    updated_sets, decisions, warnings = build_global_recommendation(
        forced_sets, proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm, "low"
    )
    calls_used = reasoning_llm.usage_totals["call_count"] - calls_before
    recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated_sets}
    print(f"llm calls used: {calls_used}")
    print(f"recommendations: {recommended}")
    for decision in decisions:
        print(f"  overlap decision: {decision.proposal_id_a} vs {decision.proposal_id_b} -> {decision.verdict} ({decision.primary_dimension}) - {decision.reasoning[:160]}")
    if warnings:
        for warning in warnings:
            print(f"  duplicate-recommendation warning: {warning}")
    else:
        print("  duplicate-recommendation warnings: none")

    recommended_ids = [v for v in recommended.values() if v is not None]
    exactly_one_onboarding = len({v for v in recommended_ids if "onboarding" in (v or "")}) == 1
    third_project_untouched = recommended.get("synthetic_third_project") == "itp_proposal_flags"
    print(
        f"hard constraint 'exactly one onboarding duplicate recommended': "
        f"{'PASS' if exactly_one_onboarding else 'FAIL'}"
    )
    print(
        f"hard constraint 'third (genuinely distinct) project's recommendation is untouched': "
        f"{'PASS' if third_project_untouched else 'FAIL'}"
    )
    return {
        "recommendations": recommended,
        "decisions": overlap_decisions_to_dicts(decisions),
        "warnings": warnings,
        "calls_used": calls_used,
    }


def main() -> None:
    scenario = _load_yaml(SCENARIO_PATH)
    expected = _load_yaml(EXPECTED_PATH)

    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)

    job_target_skills, triage_results, proposals, primary_proof_by_core_claim_id, proposals_by_id = _build_scenario(scenario)

    candidate_sets = [
        rank_local_candidates(
            project["project_id"], triage_results, proposals, primary_proof_by_core_claim_id, job_target_skills
        )
        for project in scenario["projects"]
    ]

    _check_local_candidate_sets(scenario, expected, candidate_sets)

    start = time.time()
    overlap_pair_results = _check_overlap_pairs(scenario, expected, reasoning_llm)
    updated_sets, decisions, warnings = _check_global_recommendation(
        scenario, expected, candidate_sets, proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm
    )
    forced_conflict_result = _check_forced_onboarding_conflict(
        proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm
    )
    three_projects_result = _check_three_projects_two_unique_points(
        proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm
    )
    elapsed = time.time() - start

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(
        json.dumps(
            {
                "local_candidate_sets": slot_candidate_sets_to_dicts(candidate_sets),
                "overlap_pair_trials": overlap_pair_results,
                "global_recommendation": slot_candidate_sets_to_dicts(updated_sets),
                "overlap_decisions": overlap_decisions_to_dicts(decisions),
                "duplicate_warnings": warnings,
                "forced_onboarding_conflict": forced_conflict_result,
                "three_projects_two_unique_points": three_projects_result,
                "elapsed_seconds": elapsed,
                "reasoning_model_usage": reasoning_llm.usage_totals,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
