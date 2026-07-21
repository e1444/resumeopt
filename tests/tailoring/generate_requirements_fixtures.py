"""One-off script: persist real `requirements.json` fixtures for the Phase 1
posting fixture set (`tests/evals/tailoring/expected_first_stage_outcomes.yaml`'s
`postings` list).

Run via `python -m tests.tailoring.generate_requirements_fixtures` from the
repo root (PYTHONPATH=src). NOT a gated unittest - makes real, billed LLM
calls (same production defaults as `main.py`: gpt-4o summary, gpt-5-mini
reasoning). Per AGENTS.md's Fixture-First Phase Method ("persist every input
a module consumes from an upstream module ... test directly from those
inputs"), this is run ONCE to produce durable fixture inputs for
`triage_benchmark.py`, which reads the persisted files rather than
re-parsing the postings on every benchmark run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import yaml
from llm import get_llm_provider

from tailoring.requirements import extract_job_requirements, write_requirements_json

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "evals" / "tailoring"
OUTCOMES_PATH = FIXTURE_DIR / "expected_first_stage_outcomes.yaml"

SUMMARY_MODEL = "gpt-4o"
REASONING_MODEL = "gpt-5-mini"


def main() -> None:
    outcomes = yaml.safe_load(OUTCOMES_PATH.read_text(encoding="utf-8"))
    postings = outcomes["postings"]

    summary_llm = get_llm_provider("openai", model=SUMMARY_MODEL)
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)

    for posting in postings:
        posting_path = REPO_ROOT / posting["path"]
        posting_text = posting_path.read_text(encoding="utf-8")
        print(f"Extracting requirements for '{posting['id']}' ({posting_path})...")

        requirements = extract_job_requirements(
            posting_text,
            summary_llm_provider=summary_llm,
            reasoning_llm_provider=reasoning_llm,
        )

        out_path = FIXTURE_DIR / "job_postings" / f"{posting['id']}_requirements.json"
        write_requirements_json(requirements, out_path)
        print(f"  -> wrote {out_path}")
        print(f"  role_title={requirements.role_title!r} core_requirements={list(requirements.core_requirements)}")


if __name__ == "__main__":
    main()
