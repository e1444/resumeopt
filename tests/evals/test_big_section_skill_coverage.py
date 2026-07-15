import json
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider
from parser import OrchestraSingleShotParser, SingleShotPostingParser


# Benchmark score is mean per-case term-set F1 (not exact-set-match), since minor
# over/under-splitting of a compound skill phrase is a partial-credit error, not
# a total miss. This threshold is fixed once chosen; do not lower it to make a
# particular run pass.
MIN_ORCHESTRA_F1_SCORE = 0.90


class BigSectionSkillBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cases_path = self.repo_root / "tests" / "evals" / "sample_big_section_sentence_cases.yaml"
        self.artifact_path = self.repo_root / "build" / "benchmarks" / "big_section_parser_benchmark.json"

    def _score_parser(self, parser: OrchestraSingleShotParser, cases: list[dict[str, object]]) -> dict[str, object]:
        def score_case(case: dict[str, object]) -> dict[str, object]:
            chunk = str(case["chunk"]).strip()
            expected_terms = {str(term).lower().strip() for term in case.get("expected_terms", [])}

            records = parser.parse(chunk)
            observed_terms: set[str] = set()
            for record in records:
                for match in record.get("matched_skills", []):
                    observed_terms.add(str(match.get("canonical_name", "")).lower().strip())
                for term in record.get("missing_skills", []):
                    observed_terms.add(str(term).lower().strip())

            intersection = observed_terms & expected_terms
            precision = len(intersection) / len(observed_terms) if observed_terms else (1.0 if not expected_terms else 0.0)
            recall = len(intersection) / len(expected_terms) if expected_terms else 1.0
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            exact_match = observed_terms == expected_terms

            return {
                "chunk": chunk,
                "expected_terms": sorted(expected_terms),
                "observed_terms": sorted(observed_terms),
                "exact_match": exact_match,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Each case is independent, so score them concurrently instead of
        # waiting on one round trip at a time.
        with ThreadPoolExecutor(max_workers=min(8, len(cases)) or 1) as executor:
            case_results = list(executor.map(score_case, cases))

        exact_matches = sum(1 for result in case_results if result["exact_match"])
        precision_total = sum(result["precision"] for result in case_results)
        recall_total = sum(result["recall"] for result in case_results)
        f1_total = sum(result["f1"] for result in case_results)

        case_count = len(cases)
        return {
            "case_count": case_count,
            "exact_match_rate": round(exact_matches / case_count, 4) if case_count else 0.0,
            "mean_precision": round(precision_total / case_count, 4) if case_count else 0.0,
            "mean_recall": round(recall_total / case_count, 4) if case_count else 0.0,
            "mean_f1": round(f1_total / case_count, 4) if case_count else 0.0,
            "cases": case_results,
        }

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY required for the gpt-4o benchmark")
    def test_sentence_chunks_benchmark_orchestra_vs_single_shot(self) -> None:
        cases = yaml.safe_load(self.cases_path.read_text(encoding="utf-8"))

        llm = get_llm_provider("openai", model="gpt-4o")
        parsers = {
            "orchestra_single_shot": OrchestraSingleShotParser(llm_provider=llm),
            "single_shot": SingleShotPostingParser(llm_provider=llm),
        }

        # The parsers are independent of each other, so score them concurrently.
        with ThreadPoolExecutor(max_workers=len(parsers)) as executor:
            scored = dict(
                zip(
                    parsers.keys(),
                    executor.map(lambda name: self._score_parser(parsers[name], cases), parsers.keys()),
                )
            )

        report = {
            "benchmark": "big_section_sentence_cases",
            "generated_at": datetime.now(UTC).isoformat(),
            "score_definition": "mean per-case term-set F1 across the pre-split sentence cases",
            "pass_threshold": MIN_ORCHESTRA_F1_SCORE,
            "parsers": scored,
        }

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

        self.assertEqual(report["parsers"]["orchestra_single_shot"]["case_count"], len(cases))
        self.assertEqual(report["parsers"]["single_shot"]["case_count"], len(cases))
        self.assertGreaterEqual(
            report["parsers"]["orchestra_single_shot"]["mean_f1"],
            MIN_ORCHESTRA_F1_SCORE,
            msg=(
                "orchestra_single_shot parser mean F1 "
                f"{report['parsers']['orchestra_single_shot']['mean_f1']:.2%} "
                f"must stay at or above {MIN_ORCHESTRA_F1_SCORE:.2%}."
            ),
        )


if __name__ == "__main__":
    unittest.main()
