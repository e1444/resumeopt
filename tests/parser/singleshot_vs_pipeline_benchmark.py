"""Benchmark (standalone, run via `-m`, not a gated unittest, same convention
as this project's other benchmark scripts): how does this project's staged
extraction pipeline (chunking -> Stage 0 summary -> Stage 1 extraction ->
Stage 2 categorization -> Stage 3a/3b atomicity/redundancy, see
`parser.run_parser_pipeline`) compare - in F1 MEAN and VARIANCE across
repeated trials, not just a single-run snapshot - against a naive
single-shot architecture (one LLM call over the whole posting, no chunking
or staging at all)?

Sweeps across multiple models for the reasoning-tier role BOTH
architectures use (`--models`, comma-separated) - including both
reasoning-tier models (`gpt-5*`, which get `reasoning_effort`/
`max_completion_tokens` handling via `llm.openai._is_reasoning_model`) and
ordinary non-reasoning models (`gpt-4o`/`gpt-4o-mini`, where `reasoning_
effort` is simply ignored) - so the comparison isolates architecture
EFFECT from model choice, across both with- and without-reasoning models,
rather than assuming one fixed model. The staged pipeline's Stage 0 summary
(`gpt-4o`) and Stage 0.5 screening (`gpt-4o-mini`) roles are held fixed
across the sweep - only the reasoning-tier role (chunking/extraction/
categorization/atomicity/redundancy for the pipeline; the single call for
the naive baseline) varies, so each model comparison isolates that one role.

Ground truth: `sample_job_posting_big4_expected_skill_contexts.json`
(`sample_job_posting_big4.txt`, a real posting for a Python/Agentic-AI/MCP
backend engineering role). NOTE: this ground truth is a first-pass draft
annotation authored by an agent, not yet human-confirmed - per this
project's Human Review Gates policy (see `AGENTS.md`), treat scores against
it as provisional until a human reviews/edits the expected-terms file.

Repeats each architecture `trials` times per model (default 3, since every
trial is a real, billed set of API calls) to capture LLM sampling variance -
reports per-trial precision/recall/F1, plus mean and sample variance/stdev
across trials for each architecture x model combination. Scoring uses the
same matcher-based fuzzy F1 convention as `tests.parser.batching_big2_
benchmark` (resolve each observed term against the expected list via
exact/alias match, then semantic-similarity fallback, so phrasing variance
between architectures isn't penalized as a miss).

Run: `python -m tests.parser.singleshot_vs_pipeline_benchmark --trials 5
--models gpt-4o-mini,gpt-4o,gpt-5-nano,gpt-5-mini` from repo root (needs
OPENAI_API_KEY). `--posting`/`--expected` override the default fixture.
Writes `build/benchmarks/singleshot_vs_pipeline_big4_benchmark.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import normalize_whitespace  # noqa: E402
from llm import DEFAULT_REASONING_EFFORT, LLMProvider, get_llm_provider  # noqa: E402
from matcher import ExactAliasMatcher, SemanticMatcher, SkillRecord  # noqa: E402
from parser import run_parser_pipeline  # noqa: E402
from parser.summary import format_summary_block, generate_posting_summary  # noqa: E402

_DEFAULT_POSTING_PATH = "tests/evals/sample_job_posting_big4.txt"
_DEFAULT_EXPECTED_PATH = "tests/evals/sample_job_posting_big4_expected_skill_contexts.json"
_SUMMARY_MODEL = "gpt-4o"
_SCREENING_MODEL = "gpt-4o-mini"
_DEFAULT_TRIALS = 3
_DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini"]

# Same reasoning-tier model detection this project's `llm.openai` module
# uses internally (`_is_reasoning_model`, not re-imported here since it's a
# private helper) - used only to LABEL each model in the report as
# reasoning/non-reasoning, not to change any actual call behavior (that
# handling already lives generically in `llm.openai.OpenAIProvider`).
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_MODEL_PREFIXES)

_SINGLE_SHOT_JSON_SCHEMA = {
    "name": "single_shot_skill_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["skills"],
        "additionalProperties": False,
    },
}

_SINGLE_SHOT_PROMPT_TEMPLATE = """Read the ENTIRE job posting below in one pass and list every genuine, \
resume-worthy TECHNICAL skill it mentions or implies - programming languages, frameworks, tools, platforms, \
protocols, architectural patterns, and named technical disciplines/practices a candidate would legitimately \
put in a resume "Skills" section.

Do NOT include:
- soft skills (e.g. "communication", "problem-solving", "teamwork")
- degrees/qualifications (e.g. "Bachelor's degree", "Computer Science")
- generic non-technical business/domain terms (e.g. "banking experience", "risk management")

Return each skill as a short, specific keyword or phrase (split compound mentions into individual skills, \
e.g. "Python, Java" -> ["Python", "Java"]). One entry per distinct skill, no duplicates.

Job posting:
\"\"\"
{posting_text}
\"\"\"
"""


def _score(expected_terms: List[str], included_terms: List[str], llm_provider: LLMProvider) -> Dict[str, Any]:
    """Matcher-based fuzzy F1 - identical convention to
    `tests.parser.batching_big2_benchmark._score`: resolve each observed
    term against the expected-terms list via exact/alias match, then
    semantic-similarity fallback, rather than requiring literal string
    equality (phrasing variance between architectures is expected and
    shouldn't be penalized as a miss)."""

    expected_sorted = sorted({t.lower().strip() for t in expected_terms})
    observed_set = {t.lower().strip() for t in included_terms if t.strip()}

    matched_expected: set = set()
    matched_observed: set = set()
    if expected_sorted:
        expected_records = [SkillRecord(name=term, aliases=()) for term in expected_sorted]
        exact_matcher = ExactAliasMatcher(expected_records)
        try:
            semantic_matcher: Any = SemanticMatcher(expected_records, llm_provider)
        except NotImplementedError:
            semantic_matcher = None

        for observed_term in observed_set:
            candidates = exact_matcher.match(observed_term)
            if not candidates and semantic_matcher is not None:
                candidates = semantic_matcher.match(observed_term, context="")
            if candidates:
                matched_observed.add(observed_term)
                matched_expected.update(c.canonical_name for c in candidates)

    precision = len(matched_observed) / len(observed_set) if observed_set else (1.0 if not expected_sorted else 0.0)
    recall = len(matched_expected) / len(expected_sorted) if expected_sorted else 1.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "unmatched_expected": sorted(set(expected_sorted) - matched_expected),
        "included_count": len(observed_set),
    }


async def _run_pipeline_trial(posting_text: str, model: str) -> List[str]:
    """One trial of the production, staged pipeline (same architecture as
    `main.py`/`parser.factory.parse_posting`, default batch sizes). `model`
    is the reasoning-tier model under test (chunking/extraction/
    categorization/atomicity/redundancy) - Stage 0 summary (`gpt-4o`) and
    Stage 0.5 screening (`gpt-4o-mini`) stay fixed across the sweep."""

    summary_llm = get_llm_provider("openai", model=_SUMMARY_MODEL)
    reasoning_llm = get_llm_provider("openai", model=model)
    screening_llm = get_llm_provider("openai", model=_SCREENING_MODEL)

    posting_summary = await generate_posting_summary(summary_llm, posting_text)
    summary_block = format_summary_block(posting_summary)

    verdicts = await run_parser_pipeline(
        reasoning_llm,
        posting_text,
        summary_block=summary_block,
        screening_llm_provider=screening_llm,
    )
    return [v.raw_term for v in verdicts.values() if v.included]


async def _run_single_shot_trial(posting_text: str, model: str) -> List[str]:
    """One trial of the naive baseline: a single LLM call over the whole
    posting, no chunking/summary/staged categorization/redundancy at all -
    the simplest architecture a naive implementation would reach for.
    `reasoning_effort` is passed unconditionally; non-reasoning models
    (`gpt-4o`/`gpt-4o-mini`) simply ignore it (see `llm.openai.
    _is_reasoning_model`)."""

    reasoning_llm = get_llm_provider("openai", model=model)
    response = await asyncio.to_thread(
        reasoning_llm.call_json,
        prompt=_SINGLE_SHOT_PROMPT_TEMPLATE.format(posting_text=posting_text),
        json_schema=_SINGLE_SHOT_JSON_SCHEMA,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_tokens=2048,
    )
    return [str(term).strip() for term in (response or {}).get("skills", []) if str(term).strip()]


def _summarize(trials_scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    f1_scores = [t["f1"] for t in trials_scored]
    precision_scores = [t["precision"] for t in trials_scored]
    recall_scores = [t["recall"] for t in trials_scored]
    return {
        "trials": trials_scored,
        "f1_mean": round(statistics.mean(f1_scores), 4),
        "f1_variance": round(statistics.variance(f1_scores), 6) if len(f1_scores) > 1 else 0.0,
        "f1_stdev": round(statistics.stdev(f1_scores), 4) if len(f1_scores) > 1 else 0.0,
        "precision_mean": round(statistics.mean(precision_scores), 4),
        "recall_mean": round(statistics.mean(recall_scores), 4),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=_DEFAULT_TRIALS, help="Trials per architecture per model.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(_DEFAULT_MODELS),
        help="Comma-separated reasoning-tier models to sweep (both architectures use each model in turn).",
    )
    parser.add_argument("--posting", type=str, default=_DEFAULT_POSTING_PATH)
    parser.add_argument("--expected", type=str, default=_DEFAULT_EXPECTED_PATH)
    return parser.parse_args()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for the live benchmark")

    args = _parse_args()
    trials = args.trials
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    repo_root = Path(__file__).resolve().parents[2]
    posting_text = normalize_whitespace((repo_root / args.posting).read_text(encoding="utf-8"))
    expected_terms = list(json.loads((repo_root / args.expected).read_text(encoding="utf-8")).keys())
    embedding_llm = get_llm_provider("openai", model="gpt-4o")

    per_model_results: Dict[str, Dict[str, Any]] = {}

    for model in models:
        reasoning_label = "reasoning" if _is_reasoning_model(model) else "non-reasoning"
        print(f"\n### model={model} ({reasoning_label}) ###")

        pipeline_trials: List[Dict[str, Any]] = []
        singleshot_trials: List[Dict[str, Any]] = []

        for trial_num in range(1, trials + 1):
            print(f"[pipeline]    {model} trial {trial_num}/{trials} running...")
            included = asyncio.run(_run_pipeline_trial(posting_text, model))
            score = _score(expected_terms, included, embedding_llm)
            score["included_terms"] = included
            pipeline_trials.append(score)
            print(f"  precision={score['precision']:.2%} recall={score['recall']:.2%} f1={score['f1']:.2%}")

        for trial_num in range(1, trials + 1):
            print(f"[single-shot] {model} trial {trial_num}/{trials} running...")
            included = asyncio.run(_run_single_shot_trial(posting_text, model))
            score = _score(expected_terms, included, embedding_llm)
            score["included_terms"] = included
            singleshot_trials.append(score)
            print(f"  precision={score['precision']:.2%} recall={score['recall']:.2%} f1={score['f1']:.2%}")

        per_model_results[model] = {
            "reasoning_model": _is_reasoning_model(model),
            "pipeline": _summarize(pipeline_trials),
            "single_shot": _summarize(singleshot_trials),
        }

    report = {
        "benchmark": "singleshot_vs_pipeline_big4",
        "generated_at": datetime.now(UTC).isoformat(),
        "posting": args.posting,
        "expected_terms_source": args.expected,
        "expected_terms_note": "Draft ground truth, not yet human-reviewed - see AGENTS.md Human Review Gates.",
        "expected_term_count": len(expected_terms),
        "trials_per_architecture": trials,
        "models": models,
        "results_by_model": per_model_results,
    }

    artifact_path = repo_root / "build" / "benchmarks" / "singleshot_vs_pipeline_big4_benchmark.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n=== Summary ===")
    for model, result in per_model_results.items():
        reasoning_label = "reasoning" if result["reasoning_model"] else "non-reasoning"
        print(f"\n{model} ({reasoning_label}):")
        print(
            f"  pipeline:    f1_mean={result['pipeline']['f1_mean']:.2%} "
            f"f1_variance={result['pipeline']['f1_variance']:.6f} "
            f"f1_stdev={result['pipeline']['f1_stdev']:.4f}"
        )
        print(
            f"  single-shot: f1_mean={result['single_shot']['f1_mean']:.2%} "
            f"f1_variance={result['single_shot']['f1_variance']:.6f} "
            f"f1_stdev={result['single_shot']['f1_stdev']:.4f}"
        )
    print(f"\nWrote {artifact_path}")


if __name__ == "__main__":
    main()
