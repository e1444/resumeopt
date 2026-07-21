"""Live benchmark: slot triage vs. the draft fixture ground truth.

Run via `python -m tests.tailoring.triage_benchmark` from the repo root
(PYTHONPATH=src). NOT a gated unittest - makes real, billed LLM calls (one
per `triage_examples` entry in `expected_first_stage_outcomes.yaml`, gpt-5-mini
reasoning-tier, `reasoning_effort="minimal"` per this project's established
default). Reads the ALREADY-PERSISTED `requirements.json` fixtures written by
`generate_requirements_fixtures.py` rather than re-parsing postings, per
AGENTS.md's Fixture-First Phase Method.

Records per-example verdict/agreement plus call count and latency, and
writes `build/benchmarks/tailoring_phase1_triage_benchmark.json`. Per
AGENTS.md, this is inspected at the term/claim level (each disagreement's
full model reasoning is printed), not just a pass-rate percentage.
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

from tailoring.loaders import load_project_baseline, load_resume_manifest
from tailoring.requirements import load_requirements_json
from tailoring.triage import triage_bullet

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring"
OUTCOMES_PATH = FIXTURE_DIR / "expected_first_stage_outcomes.yaml"
EXPERIENCE_DIR = REPO_ROOT / "data" / "experience"
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase1_triage_benchmark.json"

REASONING_MODEL = "gpt-5-mini"


def main() -> None:
    outcomes = yaml.safe_load(OUTCOMES_PATH.read_text(encoding="utf-8"))
    triage_examples = outcomes["triage_examples"]

    manifest = load_resume_manifest(EXPERIENCE_DIR)
    bullets_by_id = {bullet.id: (bullet, project) for project in manifest.projects for bullet in project.bullets}

    requirements_by_posting = {}
    for posting in outcomes["postings"]:
        req_path = FIXTURE_DIR / "job_postings" / f"{posting['id']}_requirements.json"
        requirements_by_posting[posting["id"]] = load_requirements_json(req_path)

    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)

    results: List[Dict[str, Any]] = []
    agree_count = 0
    start = time.monotonic()

    for example in triage_examples:
        bullet_id = example["bullet_id"]
        posting_id = example["posting_id"]
        expected_labels = example["expected_labels"]

        if bullet_id not in bullets_by_id:
            print(f"SKIP: bullet_id {bullet_id!r} not found in the active resume manifest (out of scope)")
            continue

        bullet, project = bullets_by_id[bullet_id]
        requirements = requirements_by_posting[posting_id]

        result = triage_bullet(bullet, project, requirements, llm_provider=reasoning_llm)
        agrees = result.label in expected_labels
        agree_count += int(agrees)

        results.append(
            {
                "posting_id": posting_id,
                "bullet_id": bullet_id,
                "expected_labels": expected_labels,
                "actual_label": result.label,
                "agrees": agrees,
                "job_relevance": result.job_relevance,
                "narrative_value": result.narrative_value,
                "replacement_opportunity": result.replacement_opportunity,
                "reason": result.reason,
                "fixture_rationale": example["rationale"].strip(),
            }
        )

        marker = "OK  " if agrees else "MISS"
        print(f"[{marker}] {posting_id:16s} {bullet_id:60s} expected={expected_labels} actual={result.label}")
        if not agrees:
            print(f"       reason: {result.reason}")

    elapsed = time.monotonic() - start
    total = len(results)
    print()
    print(f"Agreement: {agree_count}/{total} ({100 * agree_count / total:.1f}%)" if total else "No examples run.")
    print(f"Model: {REASONING_MODEL}, calls: {total}, elapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Token usage: {reasoning_llm.usage_totals}")

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": REASONING_MODEL,
                "reasoning_effort": "minimal",
                "call_count": total,
                "elapsed_seconds": round(elapsed, 2),
                "usage_totals": reasoning_llm.usage_totals if reasoning_llm.usage_available else None,
                "agreement_count": agree_count,
                "agreement_rate": (agree_count / total) if total else None,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
