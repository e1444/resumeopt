"""End-to-end integration run: the full bullet-tailoring pipeline (Phases
0-6) chained together against one real project and one real job posting.

Run via `python -m tests.tailoring.end_to_end_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed `gpt-5` (nucleus generation,
`reasoning_effort="high"` - `tailoring.nucleus_pipeline`'s validated tier,
see its own module docstring) and `gpt-5-mini` (everything else) calls,
plus embedding calls for retrieval ranking.

Per AGENTS.md Testing Expectations ("Reserve end-to-end tests for
validating the integrated, completed workflow after its modules have
passed isolated tests"): every phase this script chains (slot triage,
project fact retrieval, whole-posting-seeded nucleus generation + synthesis,
verification, slot competition) has ALREADY been validated in isolation by
its own dedicated benchmark script and deterministic test suite. This is
the first run that chains them ALL together for one real project,
producing the dev plan's real per-stage artifacts (requirements.json,
slot_triage.json, project_fact_matches.json, posting_nucleus_claims.json,
annotated_proposal_set.json, verification_report.json,
project_candidate_sets.json, default_resume_recommendation.json) via each
module's own production writer function - not a synthetic/benchmark-only
JSON blob.

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

from tailoring.competition import (
    build_global_recommendation,
    rank_local_candidates,
    write_default_resume_recommendation_json,
    write_project_candidate_sets_json,
)
from tailoring.loaders import load_fact_atoms, load_project_baseline
from tailoring.nucleus_pipeline import discover_and_synthesize_posting_nuclei, write_posting_nucleus_claims_json
from tailoring.requirements import load_requirements_json, write_requirements_json
from tailoring.retrieval import target_skills_from_requirements, write_project_fact_matches_json
from tailoring.triage import triage_project_bullets, write_slot_triage_json
from tailoring.validation import derive_protection_states
from tailoring.verification import (
    verify_proposal,
    write_annotated_proposal_set_json,
    write_verification_report_json,
)

# Phase 3 replacement (2026-07-22, revised): whole-posting-seeded nucleus
# generation + direct synthesis (tailoring.nucleus_pipeline) replaces
# tailoring.claim_discovery/tailoring.claims entirely for this real pipeline.
# First landed per-sentence (one nucleus call per posting sentence); revised
# after a live e2e run showed both high cost (one gpt-5/"high" call per
# relevant sentence) and heavy cross-sentence duplication (a generically-
# applicable fact independently matched many sentences' own retrieval
# queries, producing near-identical nuclei with no cross-call awareness).
# Now ONE retrieval call (whole-posting target skills) + ONE nucleus call
# (proposing 1-20 candidate themes across the whole posting, preferring
# fewer, stronger ones) replace what used to be one call pair per sentence.
# No ranking/selection cap (every generated nucleus is synthesized and
# handed to verification; repair and Phase 6 competition are the only
# remaining filters).
# code still exercised by their own benchmark scripts - see the dev plan's
# Phase 3 superseded note.
# Repair (`repair_proposal`) is also temporarily disabled for this integration: it's
# unclear whether/how its rewrite-in-place approach interacts with the new nucleus-first
# sentence structure, and validating that isn't worth the time right now. A proposal that
# fails `verify_proposal` is kept and surfaced with its typed `failure_type` as a visible
# warning instead of being repaired or silently discarded.

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "benchmark_driven_llm_workflow_orchestration"
FACT_ATOMS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
BULLETS_PATH = REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_bullets.yaml"
REAL_POSTING_ID = "llm_ml_infra"
JOB_POSTINGS_DIR = REPO_ROOT / "tests" / "evals" / "tailoring" / "job_postings"

NUCLEUS_MODEL = "gpt-5"
REASONING_MODEL = "gpt-5-mini"

_RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
RUN_DIR = REPO_ROOT / "build" / "tailoring_e2e_runs" / f"{PROJECT_ID}__{REAL_POSTING_ID}__{_RUN_TIMESTAMP}"


def main() -> None:
    nucleus_llm = get_llm_provider("openai", model=NUCLEUS_MODEL)
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

    # --- Stage 3+4+5: whole-posting-seeded nucleus generation + direct
    # synthesis (Phase 3 replacement). discover_and_synthesize_posting_nuclei
    # makes ONE retrieval call (the posting's whole flattened target-skill
    # list) and ONE nucleus-generation call (1-20 candidate themes across
    # the WHOLE posting, not per-sentence, preferring fewer/stronger ones),
    # then synthesizes each nucleus directly into an AnnotatedProposal - no
    # ranking/selection cap; every generated nucleus is synthesized and
    # handed to verification below.
    print("\n=== Stage 3+4+5: whole-posting-seeded nucleus generation + synthesis ===")
    claims, proposals, matches = discover_and_synthesize_posting_nuclei(
        PROJECT_ID,
        fact_atoms,
        fact_atoms_by_project,
        protected_fact_ids,
        requirements,
        nucleus_llm_provider=nucleus_llm,
        synthesis_llm_provider=reasoning_llm,
        embedding_llm_provider=embedding_llm,
        project_summary=project.project_summary,
    )
    write_project_fact_matches_json(matches, RUN_DIR / "project_fact_matches.json")
    included_ids = {match.fact_id for match in matches if match.included}
    pool = [atom for atom in fact_atoms if atom.id in included_ids]
    print(f"Job-relevant pool (whole-posting retrieval): {len(pool)}/{len(fact_atoms)} facts: {sorted(included_ids)}")

    write_posting_nucleus_claims_json(claims, RUN_DIR / "posting_nucleus_claims.json")
    print(f"Generated {len(claims)} claim(s) from the whole posting - all advance (no ranking/selection cap):")
    proposal_by_claim_id = {proposal.core_claim_id: proposal for proposal in proposals}
    for claim in claims:
        proposal = proposal_by_claim_id[claim.id]
        print(f"  {claim.id} (facts={list(claim.supporting_fact_ids)})")
        print(f"        why={claim.why!r} result={claim.result or '(none)'!r}")
        print(f"        proposal_text={proposal.proposal_text!r}")

    # --- Stage 6: verification (repair temporarily disabled - see the Phase 3
    # replacement note above) ---
    print("\n=== Stage 6: verification (no repair) ===")
    verification_results = []
    for proposal in proposals:
        result = verify_proposal(
            proposal, fact_atoms_by_id, protected_fact_ids, protected_baseline_bullets, target_skills, reasoning_llm
        )
        print(f"  {proposal.id} -> status={result.status} failure_type={result.failure_type}")
        verification_results.append(result)

    write_annotated_proposal_set_json(proposals, RUN_DIR / "annotated_proposal_set.json")
    write_verification_report_json(verification_results, RUN_DIR / "verification_report.json")

    result_by_proposal_id = {result.proposal_id: result for result in verification_results}
    verified_proposals = [p for p in proposals if result_by_proposal_id[p.id].status == "pass"]
    warned_proposals = [p for p in proposals if result_by_proposal_id[p.id].status == "fail"]
    print(f"\n{len(verified_proposals)}/{len(proposals)} proposal(s) passed verification.")
    if warned_proposals:
        print(f"{len(warned_proposals)} proposal(s) surfaced with a warning (verification failed, repair disabled):")
        for proposal in warned_proposals:
            failure_type = result_by_proposal_id[proposal.id].failure_type
            print(f"  [WARNING: {failure_type}] {proposal.id}: {proposal.proposal_text!r}")

    # --- Stage 7: slot competition (single project - degenerate global filter) ---
    print("\n=== Stage 7: slot competition ===")
    # PostingNucleusClaim has no primary_proof field (see its docstring) - use
    # the nucleus's own result when present (a concrete, already-evidenced
    # payoff), falling back to its first cited fact's text otherwise.
    primary_proof_by_core_claim_id = {
        claim.id: claim.result or fact_atoms_by_id[claim.supporting_fact_ids[0]].fact
        for claim in claims
        if claim.supporting_fact_ids
    }
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
    if nucleus_llm.usage_available:
        print(f"Nucleus-model (gpt-5, high) token usage: {nucleus_llm.usage_totals}")
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
    for proposal in warned_proposals:
        failure_type = result_by_proposal_id[proposal.id].failure_type
        print(f"  [generated alternative, WARNING: {failure_type}] {proposal.id}: {proposal.proposal_text!r}")

    summary = {
        "project_id": PROJECT_ID,
        "posting_id": REAL_POSTING_ID,
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "elapsed_seconds": round(elapsed, 1),
        "nucleus_model_usage": nucleus_llm.usage_totals if nucleus_llm.usage_available else None,
        "reasoning_model_usage": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
        "triage": [{"bullet_id": r.bullet_id, "label": r.label} for r in triage_results],
        "generated_claim_count": len(claims),
        "proposal_count": len(proposals),
        "verified_proposal_count": len(verified_proposals),
        "warned_proposals": [
            {"proposal_id": p.id, "failure_type": result_by_proposal_id[p.id].failure_type, "proposal_text": p.proposal_text}
            for p in warned_proposals
        ],
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
