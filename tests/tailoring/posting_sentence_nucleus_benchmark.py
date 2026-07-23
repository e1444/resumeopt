"""Live validation: Phase 3.9 posting-sentence-seeded nucleus generation +
the concreteness ranking signal, against a human-approved, invented
fixture package (tests/evals/tailoring/posting_sentence_nucleus/).

Run via `python -m tests.tailoring.posting_sentence_nucleus_benchmark`
from the repo root (PYTHONPATH=src). Makes real, billed `gpt-5-mini`
calls (reasoning-tier; `medium` effort for the concreteness classifier
per its own live-validated default, the project default otherwise).

This is the FIXTURE-BASED generalization check for the mechanism already
validated against real project/posting data in `scratch/phase3_9_spike*.py`
(throwaway, not committed) - it exists to confirm the SAME validated
behaviors (broad-sentence atomicity + off-theme exclusion, compound-
sentence partial support, zero-candidate short-circuit, concreteness
discrimination) hold on independently-designed, invented data, not just
the one real project used to develop the mechanism.
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

from tailoring.claims import classify_claim_concreteness, generate_core_claim_molecules
from tailoring.loaders import load_fact_atoms

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "posting_sentence_nucleus"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase3_9_posting_sentence_benchmark.json"

REASONING_MODEL = "gpt-5-mini"
PROJECT_ID = "demo_platform_project"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_sentence_cases(reasoning_llm, fact_atoms_by_id) -> List[Dict[str, Any]]:
    sentence_data = _load_yaml(FIXTURE_DIR / "sentence_cases.yaml")
    print("=== Sentence-seeded generation cases ===")

    reports = []
    for case in sentence_data["cases"]:
        candidates = [fact_atoms_by_id[fid] for fid in case["candidate_fact_ids"]]
        print(f"\n--- {case['id']}: {case['requirement_sentence'].strip()!r} ---")
        print(f"Candidate facts ({len(candidates)}): {[a.id for a in candidates]}")

        calls_before = reasoning_llm.usage_totals["call_count"]
        if not candidates:
            claims = []
            print("  (0 candidates - skipping LLM call entirely)")
        else:
            claims = generate_core_claim_molecules(
                PROJECT_ID, candidates, reasoning_llm, requirement_sentence=case["requirement_sentence"]
            )
        calls_used = reasoning_llm.usage_totals["call_count"] - calls_before

        cited_fact_ids = set()
        for claim in claims:
            cited_fact_ids.update(claim.supporting_fact_ids)
            print(f"    - {claim.id}: why={claim.why!r}")
            print(f"      result: {claim.result!r}")
            print(f"      facts: {list(claim.supporting_fact_ids)}")
        print(f"  llm calls used: {calls_used}")

        reports.append(
            {
                "case_id": case["id"],
                "candidate_fact_ids": case["candidate_fact_ids"],
                "cited_fact_ids": sorted(cited_fact_ids),
                "claims": [
                    {"id": c.id, "why": c.why, "result": c.result, "facts": list(c.supporting_fact_ids)}
                    for c in claims
                ],
                "llm_calls_used": calls_used,
            }
        )

    # Mechanical hard-constraint checks.
    by_case = {r["case_id"]: r for r in reports}
    aside_id = "demo_platform_project_fact_008"

    narrow_pass = aside_id not in by_case["narrow_well_covered"]["cited_fact_ids"]
    print(f"\nhard constraint 'narrow_well_covered never cites the offsite aside': {'PASS' if narrow_pass else 'FAIL'}")

    broad_no_aside = aside_id not in by_case["broad_catch_all"]["cited_fact_ids"]
    print(f"hard constraint 'broad_catch_all never cites the offsite aside': {'PASS' if broad_no_aside else 'FAIL'}")
    deliverable_groups = {
        "backend": "demo_platform_project_fact_003",
        "pipeline": "demo_platform_project_fact_004",
        "dashboard": "demo_platform_project_fact_005",
        "search": "demo_platform_project_fact_006",
    }
    broad_no_merge = True
    for claim in by_case["broad_catch_all"]["claims"]:
        present = [name for name, fid in deliverable_groups.items() if fid in claim["facts"]]
        if len(present) > 1:
            broad_no_merge = False
    print(f"hard constraint 'broad_catch_all never merges 2+ distinct deliverables into 1 claim': {'PASS' if broad_no_merge else 'FAIL'}")

    compound_no_fabrication = len(set(by_case["compound_partial_support"]["cited_fact_ids"]) - {aside_id, "demo_platform_project_fact_006", "demo_platform_project_fact_007"}) == 0
    print(f"hard constraint 'compound_partial_support never fabricates messaging evidence': {'PASS' if compound_no_fabrication else 'FAIL'}")

    zero_no_calls = by_case["zero_candidates"]["llm_calls_used"] == 0
    print(f"hard constraint 'zero_candidates makes 0 LLM calls': {'PASS' if zero_no_calls else 'FAIL'}")

    return reports


def check_concreteness(reasoning_llm, fact_atoms_by_id) -> List[Dict[str, Any]]:
    from tailoring.models import CoreClaimMolecule

    expected = _load_yaml(FIXTURE_DIR / "expected_outcomes.yaml")
    print("\n\n=== Concreteness classifier cases ===")

    reports = []
    for case in expected["concreteness_cases"]:
        claim = CoreClaimMolecule(
            id=case["id"],
            project_id=PROJECT_ID,
            claim_text="",
            supporting_fact_ids=tuple(case["fact_ids"]),
            target_skills=(),
            primary_proof="",
            rationale="",
            why=case["why"],
            result=case["result"],
        )
        result = classify_claim_concreteness(claim, fact_atoms_by_id, reasoning_llm)
        expected_concrete = case["expected_concrete"]
        match = "PASS" if result == expected_concrete else "FAIL"
        print(f"\n[{match}] {case['id']} - expected concrete={expected_concrete}, got concrete={result}")
        reports.append({"case_id": case["id"], "expected_concrete": expected_concrete, "actual_concrete": result})

    return reports


def main() -> None:
    fact_atoms = load_fact_atoms(FIXTURE_DIR / "demo_platform_project_fact_atoms.yaml")
    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}

    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)

    start = time.time()
    sentence_reports = check_sentence_cases(reasoning_llm, fact_atoms_by_id)
    concreteness_reports = check_concreteness(reasoning_llm, fact_atoms_by_id)
    elapsed = time.time() - start

    print(f"\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(
        json.dumps(
            {
                "sentence_cases": sentence_reports,
                "concreteness_cases": concreteness_reports,
                "elapsed_seconds": elapsed,
                "reasoning_model_usage": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
