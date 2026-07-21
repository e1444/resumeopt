"""Live validation: project-level fact retrieval on real data (Phase 2).

Run via `python -m tests.tailoring.retrieval_benchmark` from the repo root
(PYTHONPATH=src). Makes real, billed embedding API calls for the semantic
tier (exact-tier-only runs make zero calls). Reads the ALREADY-PERSISTED
`requirements.json` fixtures (see `generate_requirements_fixtures.py`)
rather than re-parsing postings.

Uses `protected_fact_ids=set()` deliberately (not a real triage pass): this
script's purpose is to validate the RETRIEVAL/matching mechanism's recall
and precision against real, messy job-derived target-skill lists and
compare exact-only vs exact+semantic tiers - the protection-exclusion
mechanism itself is already covered by deterministic fixture tests in
tests/tailoring/test_retrieval.py and doesn't need live LLM data to
re-validate. (Separately: every real triage result gathered so far for this
project's bullets against these same postings has been uniformly `keep`
for the aligned posting or uniformly `deprioritize`/`candidate_for_replacement`
for the others - there is no real posting in the current fixture set that
would leave a genuinely MIXED protected/eligible split for this one active
project, so a live protection-integrated run wouldn't exercise that
mechanism any more meaningfully than the synthetic fixture already does.)

Per AGENTS.md, this is inspected fact-by-fact (which facts were retrieved,
at what tier, for what target skill), not just aggregate counts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import yaml
from llm import get_llm_provider

from tailoring.loaders import load_fact_atoms
from tailoring.requirements import load_requirements_json
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring"
OUTCOMES_PATH = FIXTURE_DIR / "expected_first_stage_outcomes.yaml"
PROJECT_ID = "constrained_optimization_for_generative_classification"
FACT_ATOMS_PATH = (
    REPO_ROOT / "data" / "experience" / PROJECT_ID / f"{PROJECT_ID}_fact_atoms.yaml"
)
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase2_retrieval_benchmark.json"


def main() -> None:
    outcomes = yaml.safe_load(OUTCOMES_PATH.read_text(encoding="utf-8"))
    fact_atoms = load_fact_atoms(FACT_ATOMS_PATH)
    fact_texts = {atom.id: atom.fact for atom in fact_atoms}
    fact_atoms_by_project = {PROJECT_ID: fact_atoms}

    # `model` is irrelevant here (only .embed() is ever called on this
    # provider) - embedding_model defaults to "text-embedding-3-small".
    embedding_llm = get_llm_provider("openai")

    report = {}
    for posting in outcomes["postings"]:
        posting_id = posting["id"]
        requirements = load_requirements_json(FIXTURE_DIR / "job_postings" / f"{posting_id}_requirements.json")
        target_skills = target_skills_from_requirements(requirements)

        exact_only = retrieve_project_fact_pool(
            PROJECT_ID, fact_atoms_by_project, set(), target_skills, llm_provider=None
        )

        start = time.monotonic()
        with_semantic = retrieve_project_fact_pool(
            PROJECT_ID, fact_atoms_by_project, set(), target_skills, llm_provider=embedding_llm
        )
        semantic_elapsed = time.monotonic() - start

        exact_included = sorted(match.fact_id for match in exact_only if match.included)
        semantic_included = sorted(match.fact_id for match in with_semantic if match.included)
        semantic_only = sorted(set(semantic_included) - set(exact_included))

        print(f"\n=== {posting_id} ({len(target_skills)} target skills) ===")
        print(f"exact-only included:      {exact_included}")
        print(f"exact+semantic included:  {semantic_included}")
        print(f"semantic-tier-only adds:  {semantic_only}")
        for fact_id in semantic_only:
            match = next(m for m in with_semantic if m.fact_id == fact_id)
            print(f"  {fact_id} ({fact_texts[fact_id][:70]!r}) <- {match.matched_target_skill!r} (score={match.score})")
        print(f"semantic-tier elapsed: {semantic_elapsed:.2f}s")

        report[posting_id] = {
            "target_skill_count": len(target_skills),
            "exact_only_included": exact_included,
            "exact_plus_semantic_included": semantic_included,
            "semantic_tier_only_additions": semantic_only,
            "semantic_elapsed_seconds": round(semantic_elapsed, 2),
        }

    if embedding_llm.usage_available:
        print(f"\nTotal embedding usage: {embedding_llm.usage_totals}")
        report["embedding_usage_totals"] = embedding_llm.usage_totals

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
