"""Live validation: proposal synthesis, verification, and typed repair (Phase 5).

Run via `python -m tests.tailoring.verification_benchmark` from the repo
root (PYTHONPATH=src). Makes real, billed gpt-5-mini calls (reasoning-tier,
`reasoning_effort="low"` per the Phase 3.6/3.7 lesson) plus embedding calls
for the real project's Phase 2/4 retrieval/expansion steps.

Part 1: runs `verify_proposal` (and `repair_proposal` where a repairable
failure is expected) against all 6 fixture cases in
tests/evals/tailoring/verification/, using each case's pre-authored
`proposal_text` directly (ground truth, not synthesized live - per
AGENTS.md's "test each module directly with its persisted/fixture
inputs"). Each case is run CLASSIFIER_TRIALS times for an agreement rate,
not a single sample. Hard constraints from `expected_outcomes.yaml` are
printed alongside the actual result for manual inspection (they are prose,
not mechanically parsed), except the 2 constraints that ARE mechanically
checkable: `protected_fact_reuse_unresolvable` must make zero LLM calls,
and every repair attempt's before/after text must actually differ from the
proposal's original text (a repair that "succeeds" by returning the
unchanged input would be a false pass).

Part 2: runs the real project's real Phase 3/4 pipeline (retrieval ->
generation -> ranking -> expansion) to get real `ExpandedClaimMolecule`s,
treats one real baseline bullet's own cited facts as protected (simulating
a "keep"-triaged bullet, per `ProtectionState`), then runs
`synthesize_proposal` -> `verify_proposal` -> `repair_proposal` for real,
CLASSIFIER_TRIALS times per claim, and prints the actual synthesized and
(if applicable) repaired text for inspection - not just verdicts.
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

from tailoring.claims import generate_core_claim_molecules, rank_core_claim_molecules
from tailoring.expansion import apply_verbosity_prefilter, build_support_pool, expand_claim_molecule
from tailoring.loaders import load_fact_atoms
from tailoring.models import AnnotatedProposal, BaselineBullet, CoreClaimMolecule, FactAtom
from tailoring.requirements import load_requirements_json
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements
from tailoring.verification import repair_proposal, synthesize_proposal, verify_proposal

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "verification"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"
PROJECT_ID = "constrained_optimization_for_generative_classification"
FACT_ATOMS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
BULLETS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_bullets.yaml"
REAL_POSTING_ID = "ml_research"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase5_verification_benchmark.json"

REASONING_MODEL = "gpt-5-mini"
CLASSIFIER_TRIALS = 3


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_fixture_case(reasoning_llm, case: Dict[str, Any], fact_atoms_by_id: Dict[str, FactAtom]) -> Dict[str, Any]:
    case_id = case["case_id"]
    print(f"\n--- {case_id} ---")

    raw_claim = case["core_claim"]
    core_claim = CoreClaimMolecule(
        id=raw_claim["id"],
        project_id=raw_claim["project_id"],
        claim_text=raw_claim["claim_text"],
        supporting_fact_ids=tuple(raw_claim["supporting_fact_ids"]),
        target_skills=tuple(raw_claim["target_skills"]),
        primary_proof=raw_claim["primary_proof"],
        rationale=raw_claim["rationale"],
    )
    supporting_fact_ids = tuple(dict.fromkeys((*core_claim.supporting_fact_ids, *case["expansion_added_fact_ids"])))
    proposal = AnnotatedProposal(
        id=f"{core_claim.id}_proposal",
        project_id=core_claim.project_id,
        core_claim_id=core_claim.id,
        proposal_text=case["proposal_text"].strip(),
        supporting_fact_ids=supporting_fact_ids,
        target_skills=core_claim.target_skills,
    )

    protected_fact_ids = {"demo_verification_project_fact_101"}
    protected_baseline_bullets = [
        BaselineBullet(
            id="demo_verification_project_bullet_kept_01",
            project_id="demo_verification_project",
            order=0,
            text="Built and shipped the checkout service's core REST API using FastAPI.",
            position="start",
            fact_ids=("demo_verification_project_fact_101",),
        )
    ]

    trial_results = []
    calls_before = reasoning_llm.usage_totals["call_count"]
    for _ in range(CLASSIFIER_TRIALS):
        result = verify_proposal(
            proposal,
            fact_atoms_by_id,
            protected_fact_ids,
            protected_baseline_bullets,
            core_claim.target_skills,
            reasoning_llm,
        )
        trial_results.append(result)
    calls_used = reasoning_llm.usage_totals["call_count"] - calls_before

    statuses = [f"{r.status}:{r.failure_type}" for r in trial_results]
    print(f"proposal: {proposal.proposal_text}")
    print(f"verify_proposal trials: {statuses} (llm calls used: {calls_used})")

    majority_status, majority_failure_type = max(set(statuses), key=statuses.count).split(":", 1)
    majority_failure_type = None if majority_failure_type == "None" else majority_failure_type

    repair_report = None
    if majority_status == "fail" and majority_failure_type in ("hallucination", "bad_flow", "bad_wording"):
        majority_result = next(r for r in trial_results if r.status == "fail" and r.failure_type == majority_failure_type)
        repaired_proposal, final_result = repair_proposal(
            proposal,
            majority_result,
            fact_atoms_by_id,
            protected_fact_ids,
            protected_baseline_bullets,
            core_claim.target_skills,
            reasoning_llm,
        )
        text_actually_changed = any(step.before_text != step.after_text for step in final_result.repair_steps)
        print(
            f"repair steps: "
            f"{[(s.repair_type, s.resolution, list(s.removed_fact_ids), s.reverified_status) for s in final_result.repair_steps]}"
        )
        print(f"repaired text: {repaired_proposal.proposal_text}")
        print(f"repaired supporting_fact_ids: {list(repaired_proposal.supporting_fact_ids)}")
        print(f"text actually changed: {text_actually_changed}")
        repair_report = {
            "final_status": final_result.status,
            "final_failure_type": final_result.failure_type,
            "repair_steps": [
                {
                    "repair_type": step.repair_type,
                    "before_text": step.before_text,
                    "after_text": step.after_text,
                    "reverified_status": step.reverified_status,
                    "resolution": step.resolution,
                    "removed_fact_ids": list(step.removed_fact_ids),
                }
                for step in final_result.repair_steps
            ],
            "text_actually_changed": text_actually_changed,
        }

    expected = case.get("_expected", {})
    if expected.get("hard_constraints"):
        print("expected hard constraints (for manual review):")
        for line in expected["hard_constraints"]:
            print(f"  - {line}")

    expected_resolution = case.get("expected_resolution")
    if expected_resolution is not None and repair_report is not None:
        actual_resolutions = [step["resolution"] for step in repair_report["repair_steps"]]
        actual_removed = [fid for step in repair_report["repair_steps"] for fid in step["removed_fact_ids"]]
        expected_removed = case.get("expected_removed_fact_ids") or []
        resolution_match = expected_resolution in actual_resolutions or (
            expected_resolution == "unresolvable" and repair_report["final_failure_type"] == "unresolvable"
        )
        removed_match = set(expected_removed) <= set(actual_removed) if expected_removed else True
        print(
            f"expected_resolution={expected_resolution!r} vs actual={actual_resolutions} "
            f"-> {'PASS' if resolution_match else 'FAIL'}"
        )
        if expected_removed:
            print(
                f"expected_removed_fact_ids={expected_removed} vs actual={actual_removed} "
                f"-> {'PASS' if removed_match else 'FAIL'}"
            )

    return {
        "case_id": case_id,
        "trial_statuses": statuses,
        "llm_calls_used": calls_used,
        "majority_status": majority_status,
        "majority_failure_type": majority_failure_type,
        "repair": repair_report,
    }


def _run_real_project(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print("\n\n=== Part 2: real project's real claims -> expansion -> synthesis -> verification ===")

    fact_atoms = load_fact_atoms(FACT_ATOMS_PATH)
    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    fact_atoms_by_project = {PROJECT_ID: fact_atoms}

    bullets_data = _load_yaml(BULLETS_PATH)
    kept_bullet_data = bullets_data["bullets"][0]
    kept_bullet = BaselineBullet(
        id=kept_bullet_data["id"],
        project_id=PROJECT_ID,
        order=kept_bullet_data["order"],
        text=kept_bullet_data["text"],
        position=kept_bullet_data["position"],
        fact_ids=tuple(kept_bullet_data["fact_ids"]),
    )
    protected_fact_ids = set(kept_bullet.fact_ids)
    print(f"Treating {kept_bullet.id} as protected/'keep' - reserved facts: {sorted(protected_fact_ids)}")

    requirements = load_requirements_json(JOB_POSTINGS_DIR / f"{REAL_POSTING_ID}_requirements.json")
    target_skills = target_skills_from_requirements(requirements)

    matches = retrieve_project_fact_pool(
        PROJECT_ID, fact_atoms_by_project, protected_fact_ids, target_skills, llm_provider=embedding_llm
    )
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]

    claims = generate_core_claim_molecules(PROJECT_ID, pool, reasoning_llm)
    ranked = rank_core_claim_molecules(claims)
    selected = sorted((c for c in ranked if c.rank is not None), key=lambda c: c.rank)
    print(f"Selected {len(selected)} claim(s) from a {len(pool)}-fact job-relevant pool.")

    claim_reports = []
    for claim in selected:
        support_pool = build_support_pool(claim, pool, llm_provider=embedding_llm)
        expansion = expand_claim_molecule(claim, support_pool, fact_atoms_by_id, reasoning_llm)
        expansion = apply_verbosity_prefilter(claim, expansion)
        print(f"\nClaim {claim.id}: added={list(expansion.added_support_fact_ids)}")

        trial_reports = []
        for trial in range(CLASSIFIER_TRIALS):
            proposal = synthesize_proposal(claim, expansion, fact_atoms_by_id, reasoning_llm)
            result = verify_proposal(
                proposal,
                fact_atoms_by_id,
                protected_fact_ids,
                [kept_bullet],
                target_skills,
                reasoning_llm,
            )
            print(f"  trial {trial}: proposal_text={proposal.proposal_text!r}")
            print(f"  trial {trial}: status={result.status} failure_type={result.failure_type}")

            if result.status == "fail" and result.failure_type in ("hallucination", "bad_flow", "bad_wording"):
                _, result = repair_proposal(
                    proposal, result, fact_atoms_by_id, protected_fact_ids, [kept_bullet], target_skills, reasoning_llm
                )
                print(f"  trial {trial}: after repair status={result.status} final_text={result.final_text!r}")

            trial_reports.append({"proposal_text": proposal.proposal_text, "status": result.status, "failure_type": result.failure_type})

        claim_reports.append({"claim_id": claim.id, "trials": trial_reports})

    return {"selected_claim_count": len(selected), "claims": claim_reports}


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")  # only .embed() is used here

    start = time.monotonic()
    print("=== Part 1: dev-plan fixture cases ===")

    fact_atoms_data = _load_yaml(FIXTURE_DIR / "demo_verification_project_fact_atoms.yaml")
    fact_atoms_by_id = {
        item["id"]: FactAtom(id=item["id"], fact=item["fact"], skill_tags=tuple(item.get("skill_tags") or ()))
        for item in fact_atoms_data["fact_atoms"]
    }

    cases_data = _load_yaml(FIXTURE_DIR / "verification_cases.yaml")["cases"]
    expected_data = {entry["case_id"]: entry for entry in _load_yaml(FIXTURE_DIR / "expected_outcomes.yaml")["cases"]}
    for case in cases_data:
        case["_expected"] = expected_data.get(case["case_id"], {})

    fixture_reports = [_run_fixture_case(reasoning_llm, case, fact_atoms_by_id) for case in cases_data]

    protected_case = next(r for r in fixture_reports if r["case_id"] == "protected_fact_reuse_unresolvable")
    print(
        f"\nprotected_fact_reuse_unresolvable zero-LLM-call check: "
        f"{'PASS' if protected_case['llm_calls_used'] == 0 else 'FAIL'} "
        f"({protected_case['llm_calls_used']} calls used)"
    )

    real_report = _run_real_project(reasoning_llm, embedding_llm)
    elapsed = time.monotonic() - start

    print(f"\n\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "elapsed_seconds": round(elapsed, 1),
                "fixture_cases": fixture_reports,
                "real_project": real_report,
                "usage_totals": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
