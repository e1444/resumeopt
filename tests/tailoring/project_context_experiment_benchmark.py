"""Live experiment (NOT wired into production): does giving the
classifier ADDITIONAL, EXPLICIT project fact context (not an abstract
declared "role" label) change its verdict on the Phase 3.6/3.7 reproducer
case? Per explicit instruction, this is exploratory only - it will not be
merged into src/tailoring/expansion.py unless it demonstrably changes the
decision (reasoning-quality improvements alone are not sufficient).

Uses the EXACT REAL project fact (not an invented one) as the additional
context, since this tests whether grounded facts (which the classifier
can't as easily discount as it did with the unenforced `role` label)
change the outcome - not a hypothesis about wording style.

Reuses the production system prompts UNCHANGED from tailoring.expansion
(_SAME_UNDERLYING_DELIVERABLE_SYSTEM_PROMPT,
_MERGEABLE_INTO_ONE_CLAIM_SYSTEM_PROMPT) - only the USER prompt gains an
"Additional established project context" section listing extra facts not
currently cited by the claim. Compares baseline (no extra context) against
experimental (with the real "jointly optimizes accuracy and calibration"
fact) for both classifiers, CLASSIFIER_TRIALS times each.

Run via `python -m tests.tailoring.project_context_experiment_benchmark`
from the repo root (PYTHONPATH=src). Makes real, billed gpt-5-mini calls.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider

from tailoring.expansion import (
    _MERGEABLE_INTO_ONE_CLAIM_SYSTEM_PROMPT,
    _SAME_UNDERLYING_DELIVERABLE_SYSTEM_PROMPT,
    _VERDICT_JSON_SCHEMA,
    _format_fact_list,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_OUT = REPO_ROOT / "build" / "benchmarks" / "tailoring_phase3_7_project_context_experiment.json"

REASONING_MODEL = "gpt-5-mini"
CLASSIFIER_TRIALS = 3

# The exact claim/candidate from the Phase 3.6/3.7 reproducer FIXTURE
# (tests/evals/tailoring/expansion/generative_model_claim_fact_atoms.yaml) -
# deliberately "generative MODEL", not the real project's "generative
# CLASSIFIER" wording, since an earlier draft of this experiment
# accidentally used "classifier" in both conditions, which alone flipped
# the baseline to add_support (the word already implies classification is
# part of the system) and confounded the additional-context comparison.
CLAIM_TEXT = "Developed a flow-based generative model that achieved state-of-the-art generative quality (0.88 bits/dim)."
CORE_FACT_TEXTS = [
    "Developed a flow-based generative model.",
    "Achieved state-of-the-art generative quality (0.88 bits/dim).",
]
CANDIDATE_FACT_TEXT = "Maintained 98.5% classification accuracy on MNIST using the same model."

# The REAL project fact establishing the joint objective - NOT currently
# cited by the claim above. This is the exact real fact text (not
# invented), since the point is to test whether GROUNDED extra facts (as
# opposed to an abstract "role" label) change the verdict.
ADDITIONAL_CONTEXT_FACT = "Used constrained optimization to jointly optimize predictive accuracy and probabilistic calibration."


def _classify(reasoning_llm, system_prompt: str, prompt: str) -> Dict[str, Any]:
    response = reasoning_llm.call_json(
        prompt=prompt, system_prompt=system_prompt, json_schema=_VERDICT_JSON_SCHEMA, reasoning_effort="low"
    )
    return {"verdict": bool(response.get("verdict")), "reasoning": response.get("reasoning", "")}


def _run_trials(classify_call: Callable[[], Dict[str, Any]], trials: int = CLASSIFIER_TRIALS) -> Dict[str, Any]:
    results = [classify_call() for _ in range(trials)]
    true_count = sum(1 for r in results if r["verdict"])
    return {"majority_verdict": true_count * 2 >= trials, "true_count": true_count, "trials": results}


def _build_deliverable_prompt(additional_context: Sequence[str]) -> str:
    context_block = (
        f"\n\nAdditional established project context (not currently cited by this claim, but already "
        f"established as true for this same project):\n{_format_fact_list(additional_context)}"
        if additional_context
        else ""
    )
    return (
        f'Existing claim: "{CLAIM_TEXT}"\n\n'
        f"Facts already cited by this claim:\n{_format_fact_list(CORE_FACT_TEXTS)}"
        f"{context_block}\n\n"
        f'Candidate fact to evaluate: "{CANDIDATE_FACT_TEXT}"\n\n'
        "Is this candidate fact a result of the same underlying system/method/deliverable as this claim?"
    )


def _build_mergeability_prompt(additional_context: Sequence[str]) -> str:
    context_block = (
        f"\n\nAdditional established project context (not currently cited by this claim, but already "
        f"established as true for this same project):\n{_format_fact_list(additional_context)}"
        if additional_context
        else ""
    )
    return (
        f'Existing claim: "{CLAIM_TEXT}"\n\n'
        f"Facts already cited by this claim:\n{_format_fact_list(CORE_FACT_TEXTS)}"
        f"{context_block}\n\n"
        f"Facts already added as extra support this round:\n(none)\n\n"
        f'Candidate fact to evaluate: "{CANDIDATE_FACT_TEXT}"\n\n'
        "If this claim were broadened to naturally incorporate this candidate fact, would the result still read "
        "as one coherent accomplishment?"
    )


def _run_condition(reasoning_llm, label: str, additional_context: Sequence[str]) -> Dict[str, Any]:
    print(f"\n=== {label} ===")
    deliverable = _run_trials(
        lambda: _classify(
            reasoning_llm, _SAME_UNDERLYING_DELIVERABLE_SYSTEM_PROMPT, _build_deliverable_prompt(additional_context)
        )
    )
    print(f"same_underlying_deliverable: majority={deliverable['majority_verdict']} ({deliverable['true_count']}/{CLASSIFIER_TRIALS})")
    for t in deliverable["trials"]:
        print(f"  {t['verdict']}: {t['reasoning']}")

    mergeable = _run_trials(
        lambda: _classify(
            reasoning_llm, _MERGEABLE_INTO_ONE_CLAIM_SYSTEM_PROMPT, _build_mergeability_prompt(additional_context)
        )
    )
    print(f"mergeable_into_one_claim: majority={mergeable['majority_verdict']} ({mergeable['true_count']}/{CLASSIFIER_TRIALS})")
    for t in mergeable["trials"]:
        print(f"  {t['verdict']}: {t['reasoning']}")

    would_add = deliverable["majority_verdict"] and mergeable["majority_verdict"]
    print(f"Judge (AND of both majorities): {'add_support' if would_add else 'keep_out'}")

    return {
        "label": label,
        "additional_context": list(additional_context),
        "same_underlying_deliverable": deliverable,
        "mergeable_into_one_claim": mergeable,
        "judge_would_add": would_add,
    }


def main() -> None:
    reasoning_llm = get_llm_provider("openai", model=REASONING_MODEL)

    start = time.monotonic()
    baseline = _run_condition(reasoning_llm, "baseline (no additional context)", additional_context=())
    experimental = _run_condition(
        reasoning_llm,
        "experimental (+ real 'jointly optimizes accuracy and calibration' fact)",
        additional_context=(ADDITIONAL_CONTEXT_FACT,),
    )
    elapsed = time.monotonic() - start

    print(f"\nElapsed: {elapsed:.1f}s")
    if reasoning_llm.usage_available:
        print(f"Token usage: {reasoning_llm.usage_totals}")

    changed = baseline["judge_would_add"] != experimental["judge_would_add"]
    print(f"\nDecision changed by additional context: {changed}")

    report = {
        "claim_text": CLAIM_TEXT,
        "candidate_fact_text": CANDIDATE_FACT_TEXT,
        "additional_context_fact": ADDITIONAL_CONTEXT_FACT,
        "baseline": baseline,
        "experimental": experimental,
        "decision_changed": changed,
        "elapsed_seconds": round(elapsed, 2),
    }
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
