"""Benchmark (standalone, run via `-m`, not a gated unittest, same convention
as this project's other benchmark scripts): does batching multiple chunks
per Stage 1/2 call (see `parser.extraction.EXTRACTION_BATCH_SIZE`/
`parser.categorization.CATEGORIZATION_BATCH_SIZE`) hold up against the
`sample_job_posting_big2.txt` whole-posting expected-skills fixture
(`sample_job_posting_big2_expected_skill_contexts.json`, 14 expected terms)?

Runs the full production pipeline (`parser.run_parser_pipeline`, same
chunking/screening/atomicity/redundancy stages as `main.py`) TWICE against
the same posting - once at `batch_size=1` (the production default) and once
at a batched size - and scores each against the expected-terms fixture using
this project's established matcher-based fuzzy F1 convention (build an ad
hoc `SkillRecord` list from the expected terms, resolve observed included
terms via `ExactAliasMatcher` then `SemanticMatcher` fallback).

Run: `python -m tests.parser.batching_big2_benchmark [batch_size]` from repo
root (needs OPENAI_API_KEY). `batch_size` defaults to 6. Writes
`build/benchmarks/batching_big2_benchmark.json`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import normalize_whitespace  # noqa: E402
from llm import LLMProvider, get_llm_provider  # noqa: E402
from matcher import ExactAliasMatcher, SemanticMatcher, SkillRecord  # noqa: E402
from parser import run_parser_pipeline  # noqa: E402
from parser.summary import format_summary_block, generate_posting_summary  # noqa: E402

_POSTING_PATH = "tests/evals/sample_job_posting_big2.txt"
_EXPECTED_PATH = "tests/evals/sample_job_posting_big2_expected_skill_contexts.json"
_REASONING_MODEL = "gpt-5-mini"
_SUMMARY_MODEL = "gpt-4o"
_SCREENING_MODEL = "gpt-4o-mini"


def _score(expected_terms: List[str], included_terms: List[str], llm_provider: LLMProvider) -> Dict[str, Any]:
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


async def _run_once(posting_text: str, extraction_batch_size: int, categorization_batch_size: int) -> Dict[str, Any]:
    summary_llm = get_llm_provider("openai", model=_SUMMARY_MODEL)
    reasoning_llm = get_llm_provider("openai", model=_REASONING_MODEL)
    screening_llm = get_llm_provider("openai", model=_SCREENING_MODEL)

    posting_summary = await generate_posting_summary(summary_llm, posting_text)
    summary_block = format_summary_block(posting_summary)

    verdicts = await run_parser_pipeline(
        reasoning_llm,
        posting_text,
        summary_block=summary_block,
        screening_llm_provider=screening_llm,
        extraction_batch_size=extraction_batch_size,
        categorization_batch_size=categorization_batch_size,
    )
    included = [v.raw_term for v in verdicts.values() if v.included]
    return {
        "included_terms": included,
        "reasoning_usage": reasoning_llm.usage_totals,
        "screening_usage": screening_llm.usage_totals,
        "summary_usage": summary_llm.usage_totals,
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for the live benchmark")

    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    repo_root = Path(__file__).resolve().parents[2]
    posting_text = normalize_whitespace((repo_root / _POSTING_PATH).read_text(encoding="utf-8"))
    expected_terms = list(json.loads((repo_root / _EXPECTED_PATH).read_text(encoding="utf-8")).keys())

    embedding_llm = get_llm_provider("openai", model="gpt-4o")

    print(f"[batch_size=1] running baseline (one call per chunk) over {_POSTING_PATH}...")
    baseline = asyncio.run(_run_once(posting_text, 1, 1))
    baseline_score = _score(expected_terms, baseline["included_terms"], embedding_llm)

    print(f"[batch_size={batch_size}] running batched extraction/categorization...")
    batched = asyncio.run(_run_once(posting_text, batch_size, batch_size))
    batched_score = _score(expected_terms, batched["included_terms"], embedding_llm)

    def _combined_tokens(run: Dict[str, Any]) -> int:
        return (
            run["reasoning_usage"]["total_tokens"]
            + run["screening_usage"]["total_tokens"]
            + run["summary_usage"]["total_tokens"]
        )

    def _combined_calls(run: Dict[str, Any]) -> int:
        return (
            run["reasoning_usage"]["call_count"]
            + run["screening_usage"]["call_count"]
            + run["summary_usage"]["call_count"]
        )

    report = {
        "benchmark": "batching_big2",
        "generated_at": datetime.now(UTC).isoformat(),
        "posting": _POSTING_PATH,
        "expected_term_count": len(expected_terms),
        "batch_size_tested": batch_size,
        "baseline_batch_size_1": {
            **baseline_score,
            "total_tokens": _combined_tokens(baseline),
            "call_count": _combined_calls(baseline),
            "included_terms": baseline["included_terms"],
        },
        "batched": {
            **batched_score,
            "total_tokens": _combined_tokens(batched),
            "call_count": _combined_calls(batched),
            "included_terms": batched["included_terms"],
        },
    }
    artifact_path = repo_root / "build" / "benchmarks" / "batching_big2_benchmark.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        f"\nbaseline (batch_size=1): precision={baseline_score['precision']:.2%} "
        f"recall={baseline_score['recall']:.2%} f1={baseline_score['f1']:.2%} "
        f"tokens={report['baseline_batch_size_1']['total_tokens']} calls={report['baseline_batch_size_1']['call_count']}"
    )
    print(
        f"batched (batch_size={batch_size}): precision={batched_score['precision']:.2%} "
        f"recall={batched_score['recall']:.2%} f1={batched_score['f1']:.2%} "
        f"tokens={report['batched']['total_tokens']} calls={report['batched']['call_count']}"
    )
    print(f"\nWrote {artifact_path}")


if __name__ == "__main__":
    main()
