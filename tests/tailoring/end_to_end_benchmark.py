"""End-to-end integration run: the full bullet-tailoring pipeline (Phases
0-6) chained together against one real project and one real job posting.

Run via `python -m tests.tailoring.end_to_end_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed `gpt-5-mini` reasoning-tier calls plus
embedding calls for retrieval/expansion ranking.

Per AGENTS.md Testing Expectations ("Reserve end-to-end tests for
validating the integrated, completed workflow after its modules have
passed isolated tests"): every phase this script chains (slot triage,
project fact retrieval, claim generation/ranking, bounded support
expansion, proposal synthesis/verification/repair, slot competition) has
ALREADY been validated in isolation by its own dedicated benchmark script
and deterministic test suite (Phases 0-6, all merged to main). This is the
first run that chains them ALL together for one real project, producing
the dev plan's real per-stage artifacts (requirements.json,
slot_triage.json, project_fact_matches.json, core_claim_molecules.json,
expanded_claim_molecules.json, annotated_proposal_set.json,
verification_report.json, project_candidate_sets.json,
default_resume_recommendation.json) via each module's own production
writer function - not a synthetic/benchmark-only JSON blob.

Uses the real project 'benchmark_driven_llm_workflow_orchestration' (not
yet wired into data/experience/resume_manifest.yaml - loaded directly, the
same convention every other single-phase benchmark script in this package
already uses for its own target project) against the 'llm_ml_infra'
posting fixture (an already-persisted requirements.json, reused rather
than re-parsed - its Python/FastAPI/CI-CD/Agentic-AI/LLM emphasis is a
plausible real-world match for this project's own LLM-workflow-
orchestration facts).

A real triage pass runs against ALL of this project's baseline bullets -
unlike the single-phase benchmarks, which use a simplifying "treat one
bullet as protected" shortcut (appropriate for testing ONE phase in
isolation), this script's whole point is to exercise the real, un-
shortcut chain end to end. Since only one project is run (not a full
multi-project resume), Phase 6's global greedy filter only runs in its
single-project degenerate form (nothing else to compare against, so its
own top local candidate is recommended with zero overlap calls) - this
still validates the full chain does not crash and produces a sensible
`SlotCandidateSet`/recommendation, but is not a multi-project diversity-
conflict test (already covered by `tests/tailoring/competition_benchmark.py`'s
synthetic, human-approved fixture).

Runs ONCE end to end (no repeated trials) - this validates the INTEGRATED
chain and produces one concrete artifact set for human evaluation of the
generated candidate bullet text, not classifier reliability (each phase's
own benchmark script already measures that separately).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider

from tailoring.claim_discovery import discover_core_claims_for_posting
from tailoring.claims import rank_core_claim_molecules, write_core_claim_molecules_json
from tailoring.competition import (
    build_global_recommendation,
    rank_local_candidates,
    write_default_resume_recommendation_json,
    write_project_candidate_sets_json,
)
from tailoring.expansion import apply_verbosity_prefilter, build_support_pool, expand_claim_molecule, write_expanded_claim_molecules_json
from tailoring.loaders import load_fact_atoms, load_project_baseline
from tailoring.requirements import load_requirements_json, write_requirements_json
from tailoring.retrieval import target_skills_from_requirements, write_project_fact_matches_json
from tailoring.triage import triage_project_bullets, write_slot_triage_json
from tailoring.validation import derive_protection_states
from tailoring.verification import (
    repair_proposal,
    synthesize_proposal,
    verify_proposal,
    write_annotated_proposal_set_json,
    write_verification_report_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "benchmark_driven_llm_workflow_orchestration"
FACT_ATOMS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
BULLETS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_bullets.yaml"
REAL_POSTING_ID = "llm_ml_infra"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"

REASONING_MODEL = "gpt-5-mini"

_RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
RUN_DIR = REPO_ROOT / "build" / "tailoring_e2e_runs" / f"{PROJECT_ID}__{REAL_POSTING_ID}__{_RUN_TIMESTAMP}"


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)
    embedding_llm = get_llm_provider("openai")  # only .embed() is used here

    start = time.monotonic()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {RUN_DIR.relative_to(REPO_ROOT)}")

    # --- Stage 0: load source data ---
    fact_atoms = load_fact_atoms(FACT_ATOMS_PATH)
    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    fact_atoms_by_project = {PROJECT_ID: fact_atoms}
    project = load_project_baseline(BULLETS_PATH)
    print(f"Loaded {len(fact_atoms)} fact atoms and {len(project.bullets)} baseline bullets for '{PROJECT_ID}'.")

    # --- Stage 1: job requirements (reused fixture, not re-parsed) ---
    requirements = load_requirements_json(JOB_POSTINGS_DIR / f"{REAL_POSTING_ID}_requirements.json")
    write_requirements_json(requirements, RUN_DIR / "requirements.json")
    target_skills = target_skills_from_requirements(requirements)
    print(f"\n=== Stage 1: job requirements ({REAL_POSTING_ID}) ===")
    print(f"role_title={requirements.role_title!r} target_skills={target_skills}")

    # --- Stage 2: real slot triage (every bullet, no shortcuts) ---
    print(f"\n=== Stage 2: slot triage ({len(project.bullets)} bullets) ===")
    triage_results = triage_project_bullets(project, requirements, llm_provider=reasoning_llm)
    write_slot_triage_json(triage_results, RUN_DIR / "slot_triage.json")
    for result in triage_results:
        print(f"  {result.bullet_id}: {result.label} - {(result.reason or '')[:120]}")

    triage_by_bullet_id = {result.bullet_id: result.label for result in triage_results}
    protection_states = derive_protection_states(project.bullets, triage_by_bullet_id)
    protected_fact_ids = {
        fact_id for state in protection_states if state.protected for fact_id in state.reserved_fact_ids
    }
    protected_baseline_bullets = [
        bullet for bullet, state in zip(project.bullets, protection_states) if state.protected
    ]
    print(f"Protected facts (reserved by keep/idk bullets): {sorted(protected_fact_ids) or '(none)'}")
    print(f"Protected bullets: {[b.id for b in protected_baseline_bullets] or '(none)'}")

    # --- Stage 3+4: posting-sentence-seeded claim discovery (Phase 3.9) +
    # deterministic ranking. Replaces the old single whole-pool
    # retrieval+generation calls: discover_core_claims_for_posting scopes
    # retrieval and generation to each posting requirement sentence in
    # turn, tagging each resulting claim with its seeding sentence
    # (source_requirement_sentence), then runs one residual whole-pool
    # pass over any fact no sentence's own retrieval captured (tagged
    # source_requirement_sentence=None).
    print("\n=== Stage 3+4: posting-sentence-seeded claim discovery + ranking ===")
    claims, matches = discover_core_claims_for_posting(
        PROJECT_ID,
        fact_atoms,
        fact_atoms_by_project,
        protected_fact_ids,
        requirements,
        reasoning_llm_provider=reasoning_llm,
        embedding_llm_provider=embedding_llm,
    )
    write_project_fact_matches_json(matches, RUN_DIR / "project_fact_matches.json")
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]
    print(f"Job-relevant pool (union across all sentence + residual retrievals): {len(pool)}/{len(fact_atoms)} facts: {sorted(included_ids)}")

    ranked = rank_core_claim_molecules(claims)
    write_core_claim_molecules_json(ranked, RUN_DIR / "core_claim_molecules.json")
    selected = sorted((claim for claim in ranked if claim.rank is not None), key=lambda claim: claim.rank)
    print(f"Generated {len(claims)} claim(s), selected {len(selected)} for advancement:")
    for claim in selected:
        seed = claim.source_requirement_sentence or "(residual whole-pool pass)"
        print(f"  [{claim.rank}] {claim.id}: {claim.claim_text!r} (facts={list(claim.supporting_fact_ids)})")
        print(f"        seeded_by={seed!r}")

    # --- Stage 5: bounded support expansion ---
    print("\n=== Stage 5: support expansion ===")
    expansions = []
    for claim in selected:
        support_pool = build_support_pool(claim, pool, llm_provider=embedding_llm)
        expansion = expand_claim_molecule(claim, support_pool, fact_atoms_by_id, reasoning_llm)
        expansion = apply_verbosity_prefilter(claim, expansion)
        expansions.append(expansion)
        print(f"  {claim.id}: added={list(expansion.added_support_fact_ids)} stop_reason={expansion.stop_reason!r}")
    write_expanded_claim_molecules_json(expansions, RUN_DIR / "expanded_claim_molecules.json")
    expansion_by_claim_id = {expansion.core_claim_id: expansion for expansion in expansions}

    # --- Stage 6: proposal synthesis + verification + typed repair ---
    print("\n=== Stage 6: synthesis + verification + repair ===")
    proposals = []
    verification_results = []
    for claim in selected:
        expansion = expansion_by_claim_id.get(claim.id)
        proposal = synthesize_proposal(claim, expansion, fact_atoms_by_id, reasoning_llm)
        result = verify_proposal(
            proposal, fact_atoms_by_id, protected_fact_ids, protected_baseline_bullets, target_skills, reasoning_llm
        )
        print(f"  {claim.id} -> proposal_text={proposal.proposal_text!r}")
        print(f"    verify: status={result.status} failure_type={result.failure_type}")
        if result.status == "fail" and result.failure_type in ("hallucination", "bad_flow", "bad_wording"):
            proposal, result = repair_proposal(
                proposal,
                result,
                fact_atoms_by_id,
                protected_fact_ids,
                protected_baseline_bullets,
                target_skills,
                reasoning_llm,
            )
            print(f"    after repair: status={result.status} final_text={result.final_text!r}")
        proposals.append(proposal)
        verification_results.append(result)

    write_annotated_proposal_set_json(proposals, RUN_DIR / "annotated_proposal_set.json")
    write_verification_report_json(verification_results, RUN_DIR / "verification_report.json")

    result_by_proposal_id = {result.proposal_id: result for result in verification_results}
    verified_proposals = [p for p in proposals if result_by_proposal_id[p.id].status == "pass"]
    print(f"\n{len(verified_proposals)}/{len(proposals)} proposal(s) passed verification.")

    # --- Stage 7: slot competition (single project - degenerate global filter) ---
    print("\n=== Stage 7: slot competition ===")
    primary_proof_by_core_claim_id = {claim.id: claim.primary_proof for claim in selected}
    proposals_by_id = {proposal.id: proposal for proposal in verified_proposals}
    candidate_set = rank_local_candidates(
        PROJECT_ID, triage_results, verified_proposals, primary_proof_by_core_claim_id, target_skills
    )
    updated_sets, decisions, warnings = build_global_recommendation(
        [candidate_set], proposals_by_id, primary_proof_by_core_claim_id, reasoning_llm
    )
    write_project_candidate_sets_json(updated_sets, RUN_DIR / "project_candidate_sets.json")
    write_default_resume_recommendation_json(
        updated_sets, decisions, RUN_DIR / "default_resume_recommendation.json", duplicate_warnings=warnings
    )

    final_set = updated_sets[0]
    print(f"eligible_original_bullet_ids: {list(final_set.eligible_original_bullet_ids)}")
    print(f"verified_proposal_ids (ranked): {list(final_set.verified_proposal_ids)}")
    print(f"recommended_proposal_id: {final_set.recommended_proposal_id}")
    print(f"recommendation_reason: {final_set.recommendation_reason}")

    elapsed = time.monotonic() - start
    print(f"\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Reasoning-model token usage: {reasoning_llm.usage_totals}")

    # --- Human-readable "final candidate resume bullet points" summary ---
    print("\n=== Final candidate resume bullet points for this project ===")
    proposal_text_by_id = {proposal.id: proposal.proposal_text for proposal in verified_proposals}
    original_text_by_id = {bullet.id: bullet.text for bullet in project.bullets}
    for state in protection_states:
        if state.protected:
            print(f"  [original, kept/protected] {state.bullet_id}: {original_text_by_id.get(state.bullet_id, '')!r}")
    for bullet_id in final_set.eligible_original_bullet_ids:
        print(f"  [original, eligible/replaceable] {bullet_id}: {original_text_by_id.get(bullet_id, '')!r}")
    for proposal_id in final_set.verified_proposal_ids:
        marker = " <- RECOMMENDED" if proposal_id == final_set.recommended_proposal_id else ""
        print(f"  [generated alternative]{marker} {proposal_id}: {proposal_text_by_id.get(proposal_id, '')!r}")

    summary = {
        "project_id": PROJECT_ID,
        "posting_id": REAL_POSTING_ID,
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "elapsed_seconds": round(elapsed, 1),
        "reasoning_model_usage": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
        "triage": [{"bullet_id": r.bullet_id, "label": r.label} for r in triage_results],
        "selected_claim_count": len(selected),
        "proposal_count": len(proposals),
        "verified_proposal_count": len(verified_proposals),
        "final_candidate_set": {
            "eligible_original_bullet_ids": list(final_set.eligible_original_bullet_ids),
            "verified_proposal_ids": list(final_set.verified_proposal_ids),
            "recommended_proposal_id": final_set.recommended_proposal_id,
            "recommendation_reason": final_set.recommendation_reason,
        },
        "duplicate_warnings": warnings,
    }
    with (RUN_DIR / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote all stage artifacts + run_summary.json to {RUN_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
