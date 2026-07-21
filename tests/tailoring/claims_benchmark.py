"""Live validation: claim generation/ranking on synthetic + real data (Phase 3).

Run via `python -m tests.tailoring.claims_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed gpt-5-mini calls (reasoning-tier,
`reasoning_effort="minimal"` per this project's established default).

Part 1: runs `generate_core_claim_molecules` against 7 synthetic fixtures
in tests/evals/tailoring/claims/, each GENERATION_TRIALS times (checking
consistency across trials, not just a single sample), and checks each
fixture's SEPARABLE HARD CONSTRAINT programmatically via
`_check_group_isolation` - these are mechanical checks, not classifier
judgments, because the constraints themselves are mechanical (do
supporting_fact_ids ever span a forbidden group, or ever merge a
should-be-standalone/never-cited fact with anything else?). The fixtures
deliberately vary in size and grouping difficulty (per explicit design):
4 atoms/3 obvious+1 outlier, 12 atoms/3 obvious+9 outliers, 12 atoms/3
AMBIGUOUS+9 obvious-different-discipline outliers, and 12 atoms with TWO
distinct non-mergeable same-discipline groups + 5 outliers.

Part 2: runs the real project's real fact pool (Phase 2's
`retrieve_project_fact_pool`, `protected_fact_ids=set()` - same
mechanism-only rationale as retrieval_benchmark.py, since this script
validates claim GENERATION/RANKING, not the triage-protection interplay)
through generation + deterministic ranking, then evaluates the RANKED
(selected) claims with 4 small, single-purpose LLM classifiers per
AGENTS.md's Phase 3 validation gate: fact support, single-accomplishment
coherence, local novelty (vs. the project's other selected claims), and
job-posting relevance. Each classifier is run CLASSIFIER_TRIALS times per
claim to measure verdict CONSISTENCY (a single sample cannot distinguish a
reliable judgment from a coin flip that happened to land the "right" way
once). Per AGENTS.md, classifier agreement is evidence for review, not
ground truth - all verdicts/reasoning are printed and persisted, not
collapsed into a single pass/fail number.

Also computes the dev plan's own Phase 3 metrics (BULLET_TAILORING_DEV_PLAN.md
Phase 3 validation gate) directly from generated claims, not just classifier
verdicts: unsupported-fact-ID rate and duplicate-primary-proof rate, pooled
across every synthetic trial and the real project.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Sequence, Tuple

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

# Repeated-trial counts, added specifically to measure RELIABILITY (verdict
# consistency across resampling), not just whether one sample looked
# plausible - a single generation/classification call cannot distinguish a
# reliable judgment from a coin flip that happened to land the "right" way
# once.
GENERATION_TRIALS = 2
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


# --- Synthetic fixture hard-constraint checks -------------------------------
#
# One generic mechanical checker covers every fixture's separable hard
# constraint by expressing it as (a) groups of fact ids that must never be
# cited together in the same claim, (b) fact ids that may be cited but only
# as a claim's SOLE supporting fact (never merged with anything else), and/or
# (c) fact ids that must never be cited at all, even alone.


def _check_group_isolation(
    claims: Sequence[CoreClaimMolecule],
    exclusive_groups: Tuple[FrozenSet[str], ...] = (),
    standalone_only_ids: FrozenSet[str] = frozenset(),
    never_cited_ids: FrozenSet[str] = frozenset(),
) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    for claim in claims:
        ids = set(claim.supporting_fact_ids)

        never_cited_hit = ids & never_cited_ids
        if never_cited_hit:
            violations.append(
                {"claim_id": claim.id, "reason": "never_cited_id_used", "fact_ids": sorted(never_cited_hit)}
            )
            continue

        touched_groups = [group for group in exclusive_groups if ids & group]
        if len(touched_groups) > 1:
            violations.append({"claim_id": claim.id, "reason": "cross_group_merge", "fact_ids": sorted(ids)})
            continue

        standalone_hit = ids & standalone_only_ids
        if standalone_hit and len(ids) > 1:
            violations.append(
                {"claim_id": claim.id, "reason": "standalone_id_merged", "fact_ids": sorted(ids)}
            )

    return {"passed": not violations, "violations": violations}


_FRONTEND = frozenset(
    {
        "demo_fullstack_project_fact_001",
        "demo_fullstack_project_fact_002",
        "demo_fullstack_project_fact_003",
    }
)
_BACKEND = frozenset(
    {
        "demo_fullstack_project_fact_004",
        "demo_fullstack_project_fact_005",
        "demo_fullstack_project_fact_006",
    }
)
_NOISE = frozenset({"demo_noise_project_fact_004", "demo_noise_project_fact_005"})
_SCATTERED_SINGLETONS = tuple(
    frozenset({f"demo_scattered_project_fact_{i:03d}"}) for i in range(1, 4)
)
_SMALL_CLEAN_OUTLIER = frozenset({"demo_small_clean_project_fact_004"})
_LARGE_SPARSE_OUTLIERS = frozenset(f"demo_large_sparse_project_fact_{i:03d}" for i in range(4, 13))
_AMBIGUOUS_OUTLIERS = frozenset(f"demo_ambiguous_project_fact_{i:03d}" for i in range(4, 13))
_TWO_GROUPS_A = frozenset(f"demo_two_groups_project_fact_{i:03d}" for i in range(1, 4))
_TWO_GROUPS_B = frozenset(f"demo_two_groups_project_fact_{i:03d}" for i in range(4, 8))
_TWO_GROUPS_OUTLIERS = frozenset(f"demo_two_groups_project_fact_{i:03d}" for i in range(8, 13))

_FIXTURE_CHECKS: Dict[str, Callable[[Sequence[CoreClaimMolecule]], Dict[str, Any]]] = {
    "frontend_backend_no_merge": partial(_check_group_isolation, exclusive_groups=(_FRONTEND, _BACKEND)),
    "noise_unused": partial(_check_group_isolation, never_cited_ids=_NOISE),
    "no_coherent_grouping": partial(_check_group_isolation, exclusive_groups=_SCATTERED_SINGLETONS),
    "small_clean_grouping": partial(_check_group_isolation, standalone_only_ids=_SMALL_CLEAN_OUTLIER),
    "large_sparse_grouping": partial(_check_group_isolation, standalone_only_ids=_LARGE_SPARSE_OUTLIERS),
    "large_ambiguous_grouping": partial(_check_group_isolation, standalone_only_ids=_AMBIGUOUS_OUTLIERS),
    "two_distinct_groups": partial(
        _check_group_isolation, exclusive_groups=(_TWO_GROUPS_A, _TWO_GROUPS_B), standalone_only_ids=_TWO_GROUPS_OUTLIERS
    ),
}


def _run_synthetic_fixtures(
    reasoning_llm,
) -> Tuple[Dict[str, Any], List[CoreClaimMolecule], List[List[CoreClaimMolecule]]]:
    print(f"=== Part 1: synthetic fixture hard-constraint checks ({GENERATION_TRIALS} generation trials each) ===")
    results: Dict[str, Any] = {}
    all_claims: List[CoreClaimMolecule] = []
    generation_batches: List[List[CoreClaimMolecule]] = []

    for case_id, check_fn in _FIXTURE_CHECKS.items():
        fact_atoms_path = FIXTURE_DIR / f"{case_id}_fact_atoms.yaml"
        data = yaml.safe_load(fact_atoms_path.read_text(encoding="utf-8"))
        fact_atoms = [
            FactAtom(id=item["id"], fact=item["fact"], skill_tags=tuple(item.get("skill_tags") or ()))
            for item in data["fact_atoms"]
        ]

        print(f"\n[{case_id}] ({len(fact_atoms)} facts)")
        trial_reports = []
        for trial in range(1, GENERATION_TRIALS + 1):
            claims = generate_core_claim_molecules(data["project_id"], fact_atoms, reasoning_llm)
            check_result = check_fn(claims)
            all_claims.extend(claims)
            generation_batches.append(claims)

            marker = "PASS" if check_result["passed"] else "FAIL"
            print(f"  trial {trial}: {len(claims)} claim(s) -- hard constraint {marker}")
            for claim in claims:
                print(f"    - {claim.id}: facts={list(claim.supporting_fact_ids)} :: {claim.claim_text}")
            if not check_result["passed"]:
                print(f"    VIOLATIONS: {check_result['violations']}")

            trial_reports.append(
                {
                    "trial": trial,
                    "claims": [
                        {"id": c.id, "supporting_fact_ids": list(c.supporting_fact_ids), "claim_text": c.claim_text}
                        for c in claims
                    ],
                    "hard_constraint_check": check_result,
                }
            )

        all_trials_passed = all(t["hard_constraint_check"]["passed"] for t in trial_reports)
        print(f"  all {GENERATION_TRIALS} trials passed: {all_trials_passed}")
        results[case_id] = {"trials": trial_reports, "all_trials_passed": all_trials_passed}

    return results, all_claims, generation_batches


# --- Single-purpose LLM classifiers ------------------------------------------
#
# Each system prompt below is anchored with 2 concrete true/false examples
# (per AGENTS.md's LLM Scoring Rubric Design guidance - originally written
# for numeric scores, but the same "anchor with concrete examples, spell out
# edge cases explicitly" logic applies to boolean classifiers) and states an
# explicit edge-case rule, rather than leaving the boundary to be inferred
# per-call. This directly followed from a real finding: the first live
# benchmark run's fact_support prompt produced a false negative (see git
# history), and the coherence classifier's fuzzy multi-tool-one-deliverable
# boundary was flagged as under-specified in review.


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
        "independently corroborate or prove the claim beyond what they already state.\n\n"
        "Example (true): claim \"Built a Redis cache that reduced latency by 65%.\" cited facts: "
        "[\"Built a Redis-backed caching layer.\", \"Reduced average API latency by 65% after introducing the "
        "cache.\"] -> true, every detail is a restatement of a cited fact.\n"
        "Example (false): claim \"Built a Redis cache that reduced latency by 65% and cut infrastructure costs "
        "by 30%.\" with the SAME two cited facts (no cost fact given) -> false, the 30% cost figure is not in "
        "any cited fact.\n\n"
        "Answer with a single boolean verdict (true = every detail already appears in the cited facts, "
        "false = the claim adds/embellishes a detail absent from the cited facts)."
    )
    prompt = f'Claim: "{claim.claim_text}"\n\nCited source facts:\n{cited}\n\nIs this claim fully supported?'
    return _classify(reasoning_llm, system, prompt)


def _classify_single_accomplishment(reasoning_llm, claim: CoreClaimMolecule) -> Dict[str, Any]:
    system = (
        "You check exactly one thing: does this resume claim describe exactly ONE coherent accomplishment?\n\n"
        "Edge-case rule: facts describing different TOOLS or STEPS that all serve the SAME single deliverable "
        "or goal (for example adopting a configuration tool AND a tracking tool to build one reliable "
        "experiment-management workflow) count as ONE accomplishment - do not flag a claim as incoherent just "
        "because it names more than one tool or step. Only answer false when the claim describes genuinely "
        "different deliverables, goals, or unrelated systems (for example a frontend UI feature and a separate "
        "backend database schema).\n\n"
        "Example (true): \"Built an experiment-management workflow using Hydra for configuration and Weights & "
        "Biases for tracking.\" -> true, both tools serve one deliverable (reliable experiment management).\n"
        "Example (false): \"Built a React component library and designed a PostgreSQL schema for the order "
        "service.\" -> false, two unrelated deliverables (frontend UI vs. backend database).\n\n"
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
        "genuinely new information rather than substantially restating one of the others?\n\n"
        "Example (true): claim describes model results/metrics; the other selected claim describes tooling/"
        "infrastructure (a different aspect of the work) -> true, new information.\n"
        "Example (false): claim restates the same accomplishment or metric already covered by another selected "
        "claim, just reworded -> false, redundant.\n\n"
        "Answer with a single boolean verdict (true = adds genuinely new information, false = largely redundant)."
    )
    prompt = f'Claim: "{claim.claim_text}"\n\nOther selected claims for this project:\n{others}\n\nIs this claim novel?'
    return _classify(reasoning_llm, system, prompt)


def _classify_job_relevance(reasoning_llm, claim: CoreClaimMolecule, target_skills: List[str]) -> Dict[str, Any]:
    skills = ", ".join(target_skills) or "(none listed)"
    system = (
        "You check exactly one thing: is this resume claim relevant to a job posting's target skills - would "
        "it plausibly help demonstrate at least one of the listed target skills to a hiring reader?\n\n"
        "Example (true): claim names a tool/technique explicitly listed among the target skills (or a close, "
        "well-known synonym of one) -> true.\n"
        "Example (false): claim describes a real accomplishment with no overlap, direct or plausible/"
        "transferable, with any listed target skill -> false.\n\n"
        "Answer with a single boolean verdict (true = relevant, false = not relevant)."
    )
    prompt = f'Claim: "{claim.claim_text}"\nClaim\'s own target_skills: {list(claim.target_skills)}\n\nJob posting target skills: {skills}\n\nIs this claim relevant to this posting?'
    return _classify(reasoning_llm, system, prompt)


def _run_classifier_trials(classify_call: Callable[[], Dict[str, Any]], trials: int = CLASSIFIER_TRIALS) -> Dict[str, Any]:
    """Calls a single-purpose classifier `trials` times and reports verdict
    CONSISTENCY (majority verdict + agreement rate), not just one sample -
    per the reliability gap identified in review: a single call cannot
    distinguish a reliable judgment from a coin flip that happened to land
    the "right" way once."""

    trial_results = [classify_call() for _ in range(trials)]
    true_count = sum(1 for result in trial_results if result["verdict"])
    majority_verdict = true_count * 2 >= trials
    agreement_rate = max(true_count, trials - true_count) / trials
    return {
        "majority_verdict": majority_verdict,
        "agreement_rate": round(agreement_rate, 3),
        "true_count": true_count,
        "trials": trial_results,
    }


# --- Dev-plan Phase 3 metrics (BULLET_TAILORING_DEV_PLAN.md validation gate) -
#
# "Validate separable fixture constraints directly; for open-ended claim
# generation, measure classifier verdict rates, duplicate-primary-proof
# rate, and unsupported-fact-ID rate rather than precision/recall against a
# supposedly complete expected claim set." Computed directly from generated
# claims - no extra LLM calls needed for these two.


def _unsupported_fact_id_rate(claims: Sequence[CoreClaimMolecule]) -> float:
    if not claims:
        return 0.0
    bad = sum(1 for c in claims if c.non_advancement_reason and "unsupported_fact_ids" in c.non_advancement_reason)
    return bad / len(claims)


def _duplicate_primary_proof_rate(claims: Sequence[CoreClaimMolecule]) -> float:
    """Duplicate rate WITHIN ONE generation batch's claim set (claims that
    genuinely competed for the same underlying grouping and still ended up
    citing the same primary proof). Callers with multiple independent
    generation batches (e.g. repeated trials of the same fixture) must use
    `_duplicate_primary_proof_rate_across_batches` instead - pooling
    primary_proof strings ACROSS separate trials would flag two trials
    independently describing the same real facts as "duplication", which
    is an artifact of resampling, not a claim-quality defect.
    """

    proofs = [c.primary_proof.strip().lower() for c in claims if c.primary_proof]
    if not proofs:
        return 0.0
    counts = Counter(proofs)
    duplicated = sum(count for count in counts.values() if count > 1)
    return duplicated / len(proofs)


def _duplicate_primary_proof_rate_across_batches(batches: Sequence[Sequence[CoreClaimMolecule]]) -> float:
    total = 0
    duplicated = 0
    for batch in batches:
        proofs = [c.primary_proof.strip().lower() for c in batch if c.primary_proof]
        counts = Counter(proofs)
        total += len(proofs)
        duplicated += sum(count for count in counts.values() if count > 1)
    return duplicated / total if total else 0.0


def _run_real_project(reasoning_llm, embedding_llm) -> Dict[str, Any]:
    print(f"\n\n=== Part 2: real project fact pool -> generation -> ranking -> 4 classifiers x {CLASSIFIER_TRIALS} trials ===")

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
        fact_support = _run_classifier_trials(lambda: _classify_fact_support(reasoning_llm, claim, fact_texts))
        coherence = _run_classifier_trials(lambda: _classify_single_accomplishment(reasoning_llm, claim))
        novelty = _run_classifier_trials(lambda: _classify_local_novelty(reasoning_llm, claim, selected))
        relevance = _run_classifier_trials(
            lambda: _classify_job_relevance(reasoning_llm, claim, list(target_skills))
        )

        print(f"\nClaim {claim.id}: {claim.claim_text}")
        for name, result in (
            ("fact_support", fact_support),
            ("single_accomplishment_coherence", coherence),
            ("local_novelty", novelty),
            ("job_relevance", relevance),
        ):
            print(
                f"  {name}: majority={result['majority_verdict']} "
                f"agreement={result['true_count']}/{CLASSIFIER_TRIALS} ({result['agreement_rate']})"
            )
            for trial in result["trials"]:
                print(f"    - {trial['verdict']}: {trial['reasoning']}")

        classifier_results.append(
            {
                "claim_id": claim.id,
                "claim_text": claim.claim_text,
                "supporting_fact_ids": list(claim.supporting_fact_ids),
                "primary_proof": claim.primary_proof,
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
        "generated_claims": claims,
        "generated_claim_count": len(claims),
        "selected_claim_count": len(selected),
        "classifier_results": classifier_results,
    }


def _classifier_pass_rate(classifier_results: List[Dict[str, Any]], classifier_name: str) -> float:
    if not classifier_results:
        return 0.0
    passed = sum(1 for entry in classifier_results if entry[classifier_name]["majority_verdict"])
    return passed / len(classifier_results)


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")  # only .embed() is used here

    start = time.monotonic()
    synthetic_report, synthetic_claims, synthetic_batches = _run_synthetic_fixtures(reasoning_llm)
    real_report = _run_real_project(reasoning_llm, embedding_llm)
    elapsed = time.monotonic() - start

    real_claims = real_report.pop("generated_claims")
    all_claims = list(synthetic_claims) + list(real_claims)
    all_batches = list(synthetic_batches) + [real_claims]

    print(f"\n\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    all_hard_constraints_passed = all(case["all_trials_passed"] for case in synthetic_report.values())
    print(f"\nAll synthetic hard constraints passed (all trials): {all_hard_constraints_passed}")

    dev_plan_metrics = {
        "unsupported_fact_id_rate": round(_unsupported_fact_id_rate(all_claims), 3),
        "duplicate_primary_proof_rate": round(_duplicate_primary_proof_rate_across_batches(all_batches), 3),
        "unsupported_fact_id_rate_synthetic_only": round(_unsupported_fact_id_rate(synthetic_claims), 3),
        "duplicate_primary_proof_rate_real_project_only": round(
            _duplicate_primary_proof_rate(real_claims), 3
        ),
    }
    classifier_verdict_rates = {
        name: _classifier_pass_rate(real_report["classifier_results"], name)
        for name in ("fact_support", "single_accomplishment_coherence", "local_novelty", "job_relevance")
    }
    print(f"\nDev-plan Phase 3 metrics: {dev_plan_metrics}")
    print(f"Classifier majority-verdict pass rates (real project): {classifier_verdict_rates}")

    report = {
        "synthetic_fixtures": synthetic_report,
        "real_project": real_report,
        "elapsed_seconds": round(elapsed, 2),
        "reasoning_model": REASONING_MODEL,
        "generation_trials": GENERATION_TRIALS,
        "classifier_trials": CLASSIFIER_TRIALS,
        "reasoning_usage_totals": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
        "all_synthetic_hard_constraints_passed": all_hard_constraints_passed,
        "dev_plan_metrics": dev_plan_metrics,
        "classifier_verdict_rates": classifier_verdict_rates,
    }

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()

