"""Live validation: claim generation/ranking on synthetic + real data (Phase 3).

Run via `python -m tests.tailoring.claims_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed gpt-5-mini calls (reasoning-tier,
`reasoning_effort="minimal"` per this project's established default).

Part 1: runs `generate_core_claim_molecules` against the 3 synthetic
fixtures in tests/evals/tailoring/claims/ and checks each fixture's
SEPARABLE HARD CONSTRAINT (frontend/backend non-merge, noise-unused,
no-coherent-grouping single-fact-only) programmatically - these are
mechanical checks, not classifier judgments, because the constraints
themselves are mechanical (do supporting_fact_ids ever span two groups?).

Part 2: runs the real project's real fact pool (Phase 2's
`retrieve_project_fact_pool`, `protected_fact_ids=set()` - same
mechanism-only rationale as retrieval_benchmark.py, since this script
validates claim GENERATION/RANKING, not the triage-protection interplay)
through generation + deterministic ranking, then evaluates the RANKED
(selected) claims with 4 small, single-purpose LLM classifiers per
AGENTS.md's Phase 3 validation gate: fact support, single-accomplishment
coherence, local novelty (vs. the project's other selected claims), and
job-posting relevance. Per AGENTS.md, classifier agreement is evidence for
review, not ground truth - all verdicts/reasoning are printed and
persisted, not collapsed into a single pass/fail number.
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
from tailoring.loaders import load_fact_atoms
from tailoring.models import CoreClaimMolecule, FactAtom
from tailoring.requirements import load_requirements_json
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "claims"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"
PROJECT_ID = "constrained_optimization_for_generative_classification"
FACT_ATOMS_PATH = (
    REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
)
REAL_POSTING_ID = "ml_research"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase3_claims_benchmark.json"

REASONING_MODEL = "gpt-5-mini"

_VERDICT_JSON_SCHEMA = {
    "name": "single_purpose_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
}


def _check_frontend_backend_no_merge(claims: List[CoreClaimMolecule]) -> Dict[str, Any]:
    frontend = {
        "demo_fullstack_project_fact_001",
        "demo_fullstack_project_fact_002",
        "demo_fullstack_project_fact_003",
    }
    backend = {
        "demo_fullstack_project_fact_004",
        "demo_fullstack_project_fact_005",
        "demo_fullstack_project_fact_006",
    }
    violations = []
    for claim in claims:
        ids = set(claim.supporting_fact_ids)
        if ids & frontend and ids & backend:
            violations.append(claim.id)
    return {"passed": not violations, "violating_claim_ids": violations}


def _check_noise_unused(claims: List[CoreClaimMolecule]) -> Dict[str, Any]:
    noise_ids = {"demo_noise_project_fact_004", "demo_noise_project_fact_005"}
    violations = [claim.id for claim in claims if set(claim.supporting_fact_ids) & noise_ids]
    return {"passed": not violations, "violating_claim_ids": violations}


def _check_no_coherent_grouping(claims: List[CoreClaimMolecule]) -> Dict[str, Any]:
    violations = [claim.id for claim in claims if len(set(claim.supporting_fact_ids)) > 1]
    return {"passed": not violations, "violating_claim_ids": violations}


_FIXTURE_CHECKS = {
    "frontend_backend_no_merge": _check_frontend_backend_no_merge,
    "noise_unused": _check_noise_unused,
    "no_coherent_grouping": _check_no_coherent_grouping,
}


def _run_synthetic_fixtures(reasoning_llm) -> Dict[str, Any]:
    print("=== Part 1: synthetic fixture hard-constraint checks ===")
    results: Dict[str, Any] = {}
    for case_id, check_fn in _FIXTURE_CHECKS.items():
        fact_atoms_path = FIXTURE_DIR / f"{case_id}_fact_atoms.yaml"
        data = yaml.safe_load(fact_atoms_path.read_text(encoding="utf-8"))
        fact_atoms = [
            FactAtom(id=item["id"], fact=item["fact"], skill_tags=tuple(item.get("skill_tags") or ()))
            for item in data["fact_atoms"]
        ]

        claims = generate_core_claim_molecules(data["project_id"], fact_atoms, reasoning_llm)
        check_result = check_fn(claims)

        print(f"\n[{case_id}] {len(claims)} claim(s) generated")
        for claim in claims:
            print(f"  - {claim.id}: facts={list(claim.supporting_fact_ids)} :: {claim.claim_text}")
        marker = "PASS" if check_result["passed"] else "FAIL"
        print(f"  hard constraint: {marker} {check_result}")

        results[case_id] = {
            "claims": [
                {"id": c.id, "supporting_fact_ids": list(c.supporting_fact_ids), "claim_text": c.claim_text}
                for c in claims
            ],
            "hard_constraint_check": check_result,
        }
    return results


def _classify(reasoning_llm, system_prompt: str, prompt: str) -> Dict[str, Any]:
    response = reasoning_llm.call_json(
        prompt=prompt,
        system_prompt=system_prompt,
        json_schema=_VERDICT_JSON_SCHEMA,
        reasoning_effort="minimal",
    )
    return {"verdict": bool(response.get("verdict")), "reasoning": response.get("reasoning", "")}


def _classify_fact_support(reasoning_llm, claim: CoreClaimMolecule, fact_texts: Dict[str, str]) -> Dict[str, Any]:
    cited = "\n".join(f"- {fact_texts.get(fid, '(unknown fact id)')}" for fid in claim.supporting_fact_ids)
    system = (
        "You check exactly one thing: does a resume claim ever state a specific detail (a number, named tool, "
        "or outcome) that is NOT already present in its cited source facts? A claim that only restates, "
        "rephrases, or combines details that already appear verbatim in the cited facts IS fully supported - "
        "restating a fact's own number or tool name is not fabrication, and the facts do not need to "
        "independently corroborate or prove the claim beyond what they already state. "
        "Answer with a single boolean verdict (true = every detail already appears in the cited facts, "
        "false = the claim adds/embellishes a detail absent from the cited facts)."
    )
    prompt = f'Claim: "{claim.claim_text}"\n\nCited source facts:\n{cited}\n\nIs this claim fully supported?'
    return _classify(reasoning_llm, system, prompt)


def _classify_single_accomplishment(reasoning_llm, claim: CoreClaimMolecule) -> Dict[str, Any]:
    system = (
        "You check exactly one thing: does this resume claim describe exactly ONE coherent accomplishment, "
        "rather than two or more genuinely different, unrelated actions or outcomes merged together? "
        "Answer with a single boolean verdict (true = one coherent accomplishment, false = merged/incoherent)."
    )
    prompt = f'Claim: "{claim.claim_text}"\n\nDoes this describe exactly one coherent accomplishment?'
    return _classify(reasoning_llm, system, prompt)


def _classify_local_novelty(
    reasoning_llm, claim: CoreClaimMolecule, other_claims: List[CoreClaimMolecule]
) -> Dict[str, Any]:
    others = "\n".join(f'- "{c.claim_text}"' for c in other_claims if c.id != claim.id) or "(no other selected claims)"
    system = (
        "You check exactly one thing: compared to a project's OTHER selected resume claims, does this claim add "
        "genuinely new information rather than substantially restating one of the others? "
        "Answer with a single boolean verdict (true = adds genuinely new information, false = largely redundant)."
    )
    prompt = f'Claim: "{claim.claim_text}"\n\nOther selected claims for this project:\n{others}\n\nIs this claim novel?'
    return _classify(reasoning_llm, system, prompt)


def _classify_job_relevance(reasoning_llm, claim: CoreClaimMolecule, target_skills: List[str]) -> Dict[str, Any]:
    skills = ", ".join(target_skills) or "(none listed)"
    system = (
        "You check exactly one thing: is this resume claim relevant to a job posting's target skills - would "
        "it plausibly help demonstrate at least one of the listed target skills to a hiring reader? "
        "Answer with a single boolean verdict (true = relevant, false = not relevant)."
    )
    prompt = f'Claim: "{claim.claim_text}"\nClaim\'s own target_skills: {list(claim.target_skills)}\n\nJob posting target skills: {skills}\n\nIs this claim relevant to this posting?'
    return _classify(reasoning_llm, system, prompt)


def _run_real_project(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print("\n\n=== Part 2: real project fact pool -> generation -> ranking -> 4 classifiers ===")

    fact_atoms = load_fact_atoms(FACT_ATOMS_PATH)
    fact_texts = {atom.id: atom.fact for atom in fact_atoms}
    fact_atoms_by_project = {PROJECT_ID: fact_atoms}

    requirements = load_requirements_json(JOB_POSTINGS_DIR / f"{REAL_POSTING_ID}_requirements.json")
    target_skills = target_skills_from_requirements(requirements)

    matches = retrieve_project_fact_pool(
        PROJECT_ID, fact_atoms_by_project, set(), target_skills, llm_provider=embedding_llm
    )
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]
    print(f"Real fact pool: {len(pool)}/{len(fact_atoms)} facts included for posting {REAL_POSTING_ID!r}")
    for atom in pool:
        print(f"  - {atom.id}: {atom.fact}")

    claims = generate_core_claim_molecules(PROJECT_ID, pool, reasoning_llm)
    ranked = rank_core_claim_molecules(claims)
    selected = [c for c in ranked if c.rank is not None]
    selected.sort(key=lambda c: c.rank)

    print(f"\nGenerated {len(claims)} claim(s), selected {len(selected)} after ranking:")
    for claim in selected:
        print(f"  rank {claim.rank}: {claim.id} facts={list(claim.supporting_fact_ids)}")
        print(f"    text: {claim.claim_text}")

    classifier_results: List[Dict[str, Any]] = []
    for claim in selected:
        fact_support = _classify_fact_support(reasoning_llm, claim, fact_texts)
        coherence = _classify_single_accomplishment(reasoning_llm, claim)
        novelty = _classify_local_novelty(reasoning_llm, claim, selected)
        relevance = _classify_job_relevance(reasoning_llm, claim, list(target_skills))

        print(f"\nClaim {claim.id}: {claim.claim_text}")
        for name, result in (
            ("fact_support", fact_support),
            ("single_accomplishment_coherence", coherence),
            ("local_novelty", novelty),
            ("job_relevance", relevance),
        ):
            print(f"  {name}: {result['verdict']} - {result['reasoning']}")

        classifier_results.append(
            {
                "claim_id": claim.id,
                "claim_text": claim.claim_text,
                "supporting_fact_ids": list(claim.supporting_fact_ids),
                "fact_support": fact_support,
                "single_accomplishment_coherence": coherence,
                "local_novelty": novelty,
                "job_relevance": relevance,
            }
        )

    return {
        "posting_id": REAL_POSTING_ID,
        "target_skill_count": len(target_skills),
        "pool_fact_ids": sorted(included_ids),
        "generated_claim_count": len(claims),
        "selected_claim_count": len(selected),
        "classifier_results": classifier_results,
    }


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")  # only .embed() is used here

    start = time.monotonic()
    synthetic_report = _run_synthetic_fixtures(reasoning_llm)
    real_report = _run_real_project(reasoning_llm, embedding_llm)
    elapsed = time.monotonic() - start

    print(f"\n\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    all_hard_constraints_passed = all(
        case["hard_constraint_check"]["passed"] for case in synthetic_report.values()
    )
    print(f"\nAll synthetic hard constraints passed: {all_hard_constraints_passed}")

    report = {
        "synthetic_fixtures": synthetic_report,
        "real_project": real_report,
        "elapsed_seconds": round(elapsed, 2),
        "reasoning_model": REASONING_MODEL,
        "reasoning_usage_totals": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
        "all_synthetic_hard_constraints_passed": all_hard_constraints_passed,
    }

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
