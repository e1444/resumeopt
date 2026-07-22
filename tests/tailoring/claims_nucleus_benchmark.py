"""Live validation: Phase 3.8 nucleus-first claim formulation (Approach 1).

Run via `python -m tests.tailoring.claims_nucleus_benchmark` from the repo
root (PYTHONPATH=src). Makes real, billed `gpt-5-mini` calls plus embedding
calls for the real-project fact-retrieval comparison.

Part 1: runs `generate_core_claim_molecules` (now nucleus-aware) against
the 4 fixtures in `tests/evals/tailoring/claims_nucleus/` and mechanically
checks each case's one hard constraint (the unrelated aside fact must
never merge with the core group). The generated `why`/`result`/`claim_text`
are printed in full for TERM-LEVEL human review against each fixture's
`nucleus_shape`/`allowed_ambiguity` - per this project's explicit "not an
evaluatable fixture" stance on bullet quality, nucleus wording itself is
NOT mechanically scored here, only inspected.

Part 2: re-runs claim generation against the SAME real project/posting
data the post-Phase-6 end-to-end run already used
(`benchmark_driven_llm_workflow_orchestration` / `llm_ml_infra`), for a
direct before/after comparison against that run's already-persisted
`core_claim_molecules.json` (pre-nucleus prompt) - both are printed side
by side for human review.

Atomicity regression (never merging genuinely different deliverables) is
NOT re-tested in this script - it is checked by re-running the existing,
already-reviewed `tests/tailoring/claims_benchmark.py` Part 1 unchanged,
per the dev plan's Phase 3.8 Validation Gate.
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
from tailoring.requirements import load_requirements_json
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements
from tailoring.verification import synthesize_proposal

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "claims_nucleus"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"

REAL_PROJECT_ID = "benchmark_driven_llm_workflow_orchestration"
REAL_FACT_ATOMS_PATH = REPO_ROOT / "data" / "experience" / REAL_PROJECT_ID / f"{REAL_PROJECT_ID}_fact_atoms.yaml"
REAL_POSTING_ID = "llm_ml_infra"
BEFORE_ARTIFACT_PATH = (
    REPO_ROOT
    / "build"
    / "tailoring_e2e_runs"
    / "benchmark_driven_llm_workflow_orchestration__llm_ml_infra__20260721T225552"
    / "core_claim_molecules.json"
)
BEFORE_PROPOSALS_PATH = (
    REPO_ROOT
    / "build"
    / "tailoring_e2e_runs"
    / "benchmark_driven_llm_workflow_orchestration__llm_ml_infra__20260721T225552"
    / "annotated_proposal_set.json"
)

REASONING_MODEL = "gpt-5-mini"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase3_8_nucleus_benchmark.json"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_fixture_case(reasoning_llm, case: Dict[str, Any]) -> Dict[str, Any]:
    fact_atoms_data = _load_yaml(REPO_ROOT / case["fact_atoms_path"])
    fact_atoms = load_fact_atoms(REPO_ROOT / case["fact_atoms_path"])

    print(f"\n--- {case['id']} (expected nucleus_shape={case.get('nucleus_shape')}) ---")

    claims = generate_core_claim_molecules(fact_atoms_data["project_id"], fact_atoms, reasoning_llm)

    all_cited: List[str] = []
    for claim in claims:
        all_cited.extend(claim.supporting_fact_ids)
        print(f"  claim_id={claim.id}")
        print(f"    why:    {claim.why!r}")
        print(f"    result: {claim.result!r}")
        print(f"    text:   {claim.claim_text!r}")
        print(f"    facts:  {list(claim.supporting_fact_ids)}")

    # Mechanical hard-constraint check: the aside fact (last fact_atom in
    # every fixture, by construction) must never be merged with any other
    # fact into the same claim.
    aside_fact_id = fact_atoms[-1].id
    violated = any(
        aside_fact_id in claim.supporting_fact_ids and len(claim.supporting_fact_ids) > 1 for claim in claims
    )
    print(f"  hard constraint 'aside fact never merged': {'FAIL' if violated else 'PASS'}")

    return {
        "case_id": case["id"],
        "expected_nucleus_shape": case.get("nucleus_shape"),
        "claims": [
            {
                "claim_id": claim.id,
                "why": claim.why,
                "result": claim.result,
                "claim_text": claim.claim_text,
                "supporting_fact_ids": list(claim.supporting_fact_ids),
            }
            for claim in claims
        ],
        "hard_constraint_pass": not violated,
    }


def _run_real_project_comparison(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print("\n\n=== Part 2: real project before/after comparison ===")

    fact_atoms = load_fact_atoms(REAL_FACT_ATOMS_PATH)
    fact_atoms_by_project = {REAL_PROJECT_ID: fact_atoms}

    requirements = load_requirements_json(JOB_POSTINGS_DIR / f"{REAL_POSTING_ID}_requirements.json")
    target_skills = target_skills_from_requirements(requirements)

    matches = retrieve_project_fact_pool(
        REAL_PROJECT_ID, fact_atoms_by_project, set(), target_skills, llm_provider=embedding_llm
    )
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]

    claims = generate_core_claim_molecules(REAL_PROJECT_ID, pool, reasoning_llm)
    ranked = rank_core_claim_molecules(claims)
    selected = sorted((c for c in ranked if c.rank is not None), key=lambda c: c.rank)

    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}

    print(f"\nAFTER (nucleus-aware prompt) - {len(selected)} selected claim(s):")
    after_claims = []
    for claim in selected:
        support_pool = build_support_pool(claim, pool, llm_provider=embedding_llm)
        expansion = expand_claim_molecule(claim, support_pool, fact_atoms_by_id, reasoning_llm)
        expansion = apply_verbosity_prefilter(claim, expansion)
        proposal = synthesize_proposal(claim, expansion, fact_atoms_by_id, reasoning_llm)

        print(f"  [{claim.rank}] {claim.id}")
        print(f"    why:          {claim.why!r}")
        print(f"    result:       {claim.result!r}")
        print(f"    claim_text:   {claim.claim_text!r}  (background rationale only, not the bullet)")
        print(f"    proposal_text (the actual synthesized bullet): {proposal.proposal_text!r}")
        after_claims.append(
            {
                "claim_id": claim.id,
                "why": claim.why,
                "result": claim.result,
                "claim_text": claim.claim_text,
                "proposal_text": proposal.proposal_text,
                "supporting_fact_ids": list(claim.supporting_fact_ids),
            }
        )

    before_claims = []
    if BEFORE_ARTIFACT_PATH.exists():
        before_data = json.loads(BEFORE_ARTIFACT_PATH.read_text(encoding="utf-8"))
        before_selected = [c for c in before_data if c.get("rank") is not None]
        before_proposals_by_claim_id = {}
        if BEFORE_PROPOSALS_PATH.exists():
            for proposal in json.loads(BEFORE_PROPOSALS_PATH.read_text(encoding="utf-8")):
                before_proposals_by_claim_id[proposal["core_claim_id"]] = proposal["proposal_text"]
        print(
            f"\nBEFORE (pre-nucleus prompt, from the e2e run's own persisted artifacts) - "
            f"{len(before_selected)} selected claim(s):"
        )
        for claim in before_selected:
            before_proposal_text = before_proposals_by_claim_id.get(claim["id"])
            print(f"  [{claim['rank']}] {claim['id']}")
            print(f"    claim_text:   {claim['claim_text']!r}")
            print(f"    proposal_text (the actual synthesized bullet): {before_proposal_text!r}")
            before_claims.append({**claim, "proposal_text": before_proposal_text})
    else:
        print(f"\n(No BEFORE artifact found at {BEFORE_ARTIFACT_PATH} - skipping before/after comparison.)")

    return {"after_claims": after_claims, "before_claims": before_claims}


def main() -> None:
    expected = _load_yaml(FIXTURE_DIR / "expected_outcomes.yaml")
    cases = expected["cases"]

    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")

    start = time.monotonic()
    print("=== Part 1: nucleus-formulation fixtures ===")
    fixture_reports = [_run_fixture_case(reasoning_llm, case) for case in cases]

    real_project_report = _run_real_project_comparison(reasoning_llm, embedding_llm)
    elapsed = time.monotonic() - start

    print(f"\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(
        json.dumps(
            {
                "fixture_cases": fixture_reports,
                "real_project_comparison": real_project_report,
                "elapsed_seconds": round(elapsed, 1),
                "reasoning_model_usage": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
