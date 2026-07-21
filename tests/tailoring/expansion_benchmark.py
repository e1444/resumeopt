"""Live validation: bounded support expansion + verbosity prefilter (Phase 4).

Run via `python -m tests.tailoring.expansion_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed gpt-5-mini calls (reasoning-tier,
`reasoning_effort="minimal"`) plus embedding calls for support-pool ranking.

Part 1: runs `build_support_pool` + `expand_claim_molecule` against the
dev plan's 3 required fixture cases (tests/evals/tailoring/expansion/) and
checks the 2 HARD constraints mechanically (the adjacent-frontend and
cross-project facts must never be added), while printing the model's own
decision/reasoning for the deliberately AMBIGUOUS subtle case for
inspection (not enforced either way, per the fixture's allowed_ambiguity).

Part 2: runs the real project's real Phase 3 pipeline (retrieval ->
generation -> ranking) to get its 2 real selected claims, builds a support
pool from the SAME job-relevant retrieved pool (not the whole project - a
support fact should still be job-relevant, since it strengthens a claim
being proposed FOR this posting) for each, expands each claim, applies the
verbosity prefilter, then evaluates core-only vs. expanded candidates with
4 small single-purpose classifiers per the dev plan's Phase 4 validation
gate: clarity, target-skill coverage, same-claim integrity, and fact
support. Anchor examples + explicit edge-case rules are included in every
classifier prompt FROM THE START this time (Phase 3 had to learn this the
hard way after a live run exposed an unanchored prompt's false negative).
Each classifier runs CLASSIFIER_TRIALS times per candidate for a majority
verdict + agreement rate, not a single sample.

Since `ExpandedClaimMolecule` has no text field (Phase 4 does not author
the final expanded bullet - see `tailoring.expansion` module docstring),
the "expanded candidate" shown to classifiers here is a clearly-labeled
NAIVE mechanical stitch (core claim_text + added facts' own wording) for
EVALUATION PURPOSES ONLY - never treated as production output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import yaml
from llm import get_llm_provider

from tailoring.claims import generate_core_claim_molecules, rank_core_claim_molecules
from tailoring.expansion import (
    apply_verbosity_prefilter,
    build_support_pool,
    estimate_expanded_line_count,
    expand_claim_molecule,
)
from tailoring.loaders import load_fact_atoms
from tailoring.models import CoreClaimMolecule, ExpandedClaimMolecule, FactAtom
from tailoring.requirements import load_requirements_json
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "expansion"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"
PROJECT_ID = "constrained_optimization_for_generative_classification"
FACT_ATOMS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
REAL_POSTING_ID = "ml_research"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase4_expansion_benchmark.json"

REASONING_MODEL = "gpt-5-mini"
CLASSIFIER_TRIALS = 3

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


def _load_fixture_case() -> Dict[str, Any]:
    return yaml.safe_load((FIXTURE_DIR / "backend_claim_fact_atoms.yaml").read_text(encoding="utf-8"))


def _run_fixture_case(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print("=== Part 1: dev-plan fixture case (adjacent-frontend / subtle / irrelevant) ===")
    data = _load_fixture_case()
    fact_atoms = [
        FactAtom(id=item["id"], fact=item["fact"], skill_tags=tuple(item.get("skill_tags") or ()))
        for item in data["fact_atoms"]
    ]
    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    raw_claim = data["core_claim"]
    claim = CoreClaimMolecule(
        id=raw_claim["id"],
        project_id=raw_claim["project_id"],
        claim_text=raw_claim["claim_text"],
        supporting_fact_ids=tuple(raw_claim["supporting_fact_ids"]),
        target_skills=tuple(raw_claim["target_skills"]),
        primary_proof=raw_claim["primary_proof"],
        rationale=raw_claim["rationale"],
    )

    pool = build_support_pool(claim, fact_atoms, llm_provider=embedding_llm)
    print(f"Support pool ({len(pool)} candidates, ranked): {[atom.id for atom in pool]}")

    expansion = expand_claim_molecule(claim, pool, fact_atoms_by_id, reasoning_llm)
    print(f"added_support_fact_ids: {list(expansion.added_support_fact_ids)}")
    print(f"excluded_fact_ids: {list(expansion.excluded_fact_ids)}")
    for fact_id, reason in zip(expansion.excluded_fact_ids, expansion.exclusion_reasons):
        print(f"  - {fact_id}: {reason}")
    print(f"stop_reason: {expansion.stop_reason}")

    adjacent_frontend_id = "demo_expansion_project_fact_004"
    irrelevant_id = "demo_expansion_project_fact_006"
    subtle_id = "demo_expansion_project_fact_005"

    hard_constraint_violations = [
        fact_id
        for fact_id in (adjacent_frontend_id, irrelevant_id)
        if fact_id in expansion.added_support_fact_ids
    ]
    passed = not hard_constraint_violations
    print(f"hard constraints (fact_004/fact_006 never added): {'PASS' if passed else 'FAIL'}")

    subtle_decision = (
        "add_support"
        if subtle_id in expansion.added_support_fact_ids
        else next(
            (reason for fact_id, reason in zip(expansion.excluded_fact_ids, expansion.exclusion_reasons) if fact_id == subtle_id),
            "(not evaluated - stopped/capped before reaching it)",
        )
    )
    print(f"subtle case (fact_005) decision/reasoning: {subtle_decision}")

    return {
        "support_pool": [atom.id for atom in pool],
        "added_support_fact_ids": list(expansion.added_support_fact_ids),
        "excluded_fact_ids": list(expansion.excluded_fact_ids),
        "exclusion_reasons": list(expansion.exclusion_reasons),
        "stop_reason": expansion.stop_reason,
        "hard_constraints_passed": passed,
        "subtle_case_decision": subtle_decision,
    }


# --- Single-purpose LLM classifiers (anchored from the start, per the
# Phase 3 lesson) --------------------------------------------------------


def _classify(reasoning_llm, system_prompt: str, prompt: str) -> Dict[str, Any]:
    response = reasoning_llm.call_json(
        prompt=prompt,
        system_prompt=system_prompt,
        json_schema=_VERDICT_JSON_SCHEMA,
        reasoning_effort="minimal",
    )
    return {"verdict": bool(response.get("verdict")), "reasoning": response.get("reasoning", "")}


def _run_classifier_trials(classify_call: Callable[[], Dict[str, Any]], trials: int = CLASSIFIER_TRIALS) -> Dict[str, Any]:
    trial_results = [classify_call() for _ in range(trials)]
    true_count = sum(1 for result in trial_results if result["verdict"])
    return {
        "majority_verdict": true_count * 2 >= trials,
        "agreement_rate": round(max(true_count, trials - true_count) / trials, 3),
        "true_count": true_count,
        "trials": trial_results,
    }


def _classify_clarity(reasoning_llm, core_text: str, expanded_text: str) -> Dict[str, Any]:
    system = (
        "You check exactly one thing: is the EXPANDED resume-bullet candidate at least as clear and readable as "
        "the CORE-ONLY candidate - not confusing, not a run-on list of disconnected details?\n\n"
        "Example (true): core \"Built a REST API using FastAPI.\" expanded \"Built a REST API using FastAPI, "
        "adding pagination to handle large result sets.\" -> true, the addition reads naturally as one clause.\n"
        "Example (false): core \"Built a REST API using FastAPI.\" expanded \"Built a REST API using FastAPI. "
        "Also completed a first-aid certification. Also reviewed intern applications.\" -> false, an unrelated "
        "list makes it confusing/run-on.\n\n"
        "Answer with a single boolean verdict (true = at least as clear, false = expansion made it less clear)."
    )
    prompt = f'Core-only: "{core_text}"\n\nExpanded: "{expanded_text}"\n\nIs the expanded version at least as clear?'
    return _classify(reasoning_llm, system, prompt)


def _classify_target_skill_coverage(
    reasoning_llm, core_text: str, expanded_text: str, target_skills: List[str]
) -> Dict[str, Any]:
    skills = ", ".join(target_skills) or "(none listed)"
    system = (
        "You check exactly one thing: does the EXPANDED resume-bullet candidate cover at least as many of the "
        "listed target skills as the CORE-ONLY candidate, without diluting or crowding out the original ones?\n\n"
        "Example (true): core covers \"FastAPI\"; expanded still clearly covers \"FastAPI\" AND now also touches "
        "\"pagination\"/\"API design\" -> true, coverage maintained or improved.\n"
        "Example (false): core clearly covers \"FastAPI\"; expanded buries it under unrelated details so it no "
        "longer reads as a clear demonstration of that skill -> false, coverage got worse.\n\n"
        "Answer with a single boolean verdict (true = coverage maintained or improved, false = coverage got worse)."
    )
    prompt = (
        f'Target skills: {skills}\n\nCore-only: "{core_text}"\n\nExpanded: "{expanded_text}"\n\n'
        "Does the expanded version maintain or improve target-skill coverage?"
    )
    return _classify(reasoning_llm, system, prompt)


def _classify_same_claim_integrity(reasoning_llm, core_text: str, expanded_text: str) -> Dict[str, Any]:
    system = (
        "You check exactly one thing: does the EXPANDED resume-bullet candidate still describe the SAME single "
        "accomplishment as the CORE-ONLY candidate, without introducing a second, different accomplishment?\n\n"
        "Example (true): core \"Built a REST API using FastAPI.\" expanded \"Built a REST API using FastAPI, "
        "publishing an OpenAPI schema for client generation.\" -> true, the addition is still part of the same "
        "API deliverable.\n"
        "Example (false): core \"Built a REST API using FastAPI.\" expanded \"Built a REST API using FastAPI and "
        "built a React dashboard for viewing orders.\" -> false, a second, separate accomplishment was "
        "introduced.\n\n"
        "Answer with a single boolean verdict (true = still one accomplishment, false = a second accomplishment "
        "was introduced)."
    )
    prompt = f'Core-only: "{core_text}"\n\nExpanded: "{expanded_text}"\n\nIs this still exactly one accomplishment?'
    return _classify(reasoning_llm, system, prompt)


def _classify_fact_support(reasoning_llm, expanded_text: str, cited_facts: List[str]) -> Dict[str, Any]:
    cited = "\n".join(f"- {text}" for text in cited_facts)
    system = (
        "You check exactly one thing: does the EXPANDED resume-bullet candidate ever state a specific detail (a "
        "number, named tool, or outcome) that is NOT already present in its cited source facts? Restating a "
        "cited fact's own detail is support, not fabrication.\n\n"
        "Example (true): expanded \"Built a REST API using FastAPI, adding pagination for large result sets.\" "
        "cited facts include both the API fact and a pagination fact -> true, both details are restatements.\n"
        "Example (false): expanded \"...reducing latency by 40%.\" with no cited fact mentioning any latency "
        "figure -> false, a fabricated detail.\n\n"
        "Answer with a single boolean verdict (true = every detail already appears in the cited facts, "
        "false = a detail is fabricated/embellished)."
    )
    prompt = f'Expanded candidate: "{expanded_text}"\n\nCited source facts:\n{cited}\n\nIs this fully supported?'
    return _classify(reasoning_llm, system, prompt)


def _naive_expanded_text(claim: CoreClaimMolecule, expansion: ExpandedClaimMolecule, fact_atoms_by_id: Dict[str, FactAtom]) -> str:
    """Eval-only naive stitch - see module docstring. NEVER production text."""

    added_texts = [fact_atoms_by_id[fid].fact for fid in expansion.added_support_fact_ids if fid in fact_atoms_by_id]
    if not added_texts:
        return claim.claim_text
    return claim.claim_text + " Additional context: " + " ".join(added_texts)


def _evaluate_candidate(
    reasoning_llm, claim: CoreClaimMolecule, expansion: ExpandedClaimMolecule, fact_atoms_by_id: Dict[str, FactAtom]
) -> Dict[str, Any]:
    core_text = claim.claim_text
    expanded_text = _naive_expanded_text(claim, expansion, fact_atoms_by_id)
    cited_facts = [fact_atoms_by_id[fid].fact for fid in claim.supporting_fact_ids if fid in fact_atoms_by_id]
    cited_facts += [fact_atoms_by_id[fid].fact for fid in expansion.added_support_fact_ids if fid in fact_atoms_by_id]

    clarity = _run_classifier_trials(lambda: _classify_clarity(reasoning_llm, core_text, expanded_text))
    coverage = _run_classifier_trials(
        lambda: _classify_target_skill_coverage(reasoning_llm, core_text, expanded_text, list(claim.target_skills))
    )
    integrity = _run_classifier_trials(lambda: _classify_same_claim_integrity(reasoning_llm, core_text, expanded_text))
    fact_support = _run_classifier_trials(lambda: _classify_fact_support(reasoning_llm, expanded_text, cited_facts))

    print(f"\nClaim {claim.id}")
    print(f"  core:     {core_text}")
    print(f"  expanded: {expanded_text}  (eval-only naive stitch, not production text)")
    for name, result in (
        ("clarity", clarity),
        ("target_skill_coverage", coverage),
        ("same_claim_integrity", integrity),
        ("fact_support", fact_support),
    ):
        print(f"  {name}: majority={result['majority_verdict']} agreement={result['true_count']}/{CLASSIFIER_TRIALS}")

    return {
        "claim_id": claim.id,
        "core_text": core_text,
        "expanded_text": expanded_text,
        "added_support_fact_ids": list(expansion.added_support_fact_ids),
        "clarity": clarity,
        "target_skill_coverage": coverage,
        "same_claim_integrity": integrity,
        "fact_support": fact_support,
    }


def _run_real_project(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print("\n\n=== Part 2: real project's 2 selected claims -> expansion -> 4 classifiers ===")

    fact_atoms = load_fact_atoms(FACT_ATOMS_PATH)
    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    fact_atoms_by_project = {PROJECT_ID: fact_atoms}

    requirements = load_requirements_json(JOB_POSTINGS_DIR / f"{REAL_POSTING_ID}_requirements.json")
    target_skills = target_skills_from_requirements(requirements)

    matches = retrieve_project_fact_pool(
        PROJECT_ID, fact_atoms_by_project, set(), target_skills, llm_provider=embedding_llm
    )
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]

    claims = generate_core_claim_molecules(PROJECT_ID, pool, reasoning_llm)
    ranked = rank_core_claim_molecules(claims)
    selected = sorted((c for c in ranked if c.rank is not None), key=lambda c: c.rank)
    print(f"Selected {len(selected)} claim(s) from a {len(pool)}-fact job-relevant pool.")

    evaluations = []
    for claim in selected:
        support_pool = build_support_pool(claim, pool, llm_provider=embedding_llm)
        print(f"\nClaim {claim.id} support pool: {[atom.id for atom in support_pool]}")
        expansion = expand_claim_molecule(claim, support_pool, fact_atoms_by_id, reasoning_llm)
        expansion = apply_verbosity_prefilter(claim, expansion)
        print(
            f"  added={list(expansion.added_support_fact_ids)} excluded={list(expansion.excluded_fact_ids)} "
            f"stop_reason={expansion.stop_reason}"
        )
        print(
            f"  estimated lines: core={estimate_expanded_line_count(claim, 0)} "
            f"expanded={estimate_expanded_line_count(claim, len(expansion.added_support_fact_ids))}"
        )
        evaluations.append(_evaluate_candidate(reasoning_llm, claim, expansion, fact_atoms_by_id))

    return {"selected_claim_count": len(selected), "evaluations": evaluations}


def _classifier_pass_rate(evaluations: List[Dict[str, Any]], classifier_name: str) -> float:
    if not evaluations:
        return 0.0
    passed = sum(1 for entry in evaluations if entry[classifier_name]["majority_verdict"])
    return passed / len(evaluations)


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")  # only .embed() is used here

    start = time.monotonic()
    fixture_report = _run_fixture_case(reasoning_llm, embedding_llm)
    real_report = _run_real_project(reasoning_llm, embedding_llm)
    elapsed = time.monotonic() - start

    print(f"\n\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    classifier_pass_rates = {
        name: _classifier_pass_rate(real_report["evaluations"], name)
        for name in ("clarity", "target_skill_coverage", "same_claim_integrity", "fact_support")
    }
    print(f"\nFixture hard constraints passed: {fixture_report['hard_constraints_passed']}")
    print(f"Classifier majority-verdict pass rates (real project): {classifier_pass_rates}")

    report = {
        "fixture_case": fixture_report,
        "real_project": real_report,
        "elapsed_seconds": round(elapsed, 2),
        "reasoning_model": REASONING_MODEL,
        "classifier_trials": CLASSIFIER_TRIALS,
        "reasoning_usage_totals": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
        "classifier_pass_rates": classifier_pass_rates,
    }

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
