import json
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider, LLMProvider
from matcher import ExactAliasMatcher, SemanticMatcher, SkillRecord
from parser import OrchestraSingleShotParser, SingleShotPostingParser


# Benchmark score is mean per-case term-set F1 (not exact-set-match), since minor
# over/under-splitting of a compound skill phrase is a partial-credit error, not
# a total miss.
# sample_big_section_sentence_cases.yaml's expected_terms are treated as totalic
# (exhaustive) ground truth (2026-07-15): exactly these terms, no more/no less.
# An observed term counts toward a given expected term if the SAME tiered
# matcher the production pipeline uses (ExactAliasMatcher, then SemanticMatcher
# as a fallback) resolves it to that expected term - not if the raw/canonical
# string happens to be byte-identical. Requiring exact string equality was
# brittle by construction (e.g. a cache canonical_name or a raw missing-skill
# phrase rarely matches the fixture's exact wording even when it's clearly the
# same skill), which is exactly the wording-variance problem the matcher
# classes already exist to solve - so this test reuses them instead of
# duplicating a second, weaker string-equality heuristic.
# Still NOT enforced as a hard gate: no threshold has been calibrated yet
# against this matcher-based scoring method.
MIN_ORCHESTRA_F1_SCORE = 0.90


class BigSectionSkillBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cases_path = self.repo_root / "tests" / "evals" / "sample_big_section_sentence_cases.yaml"
        self.artifact_path = self.repo_root / "build" / "benchmarks" / "big_section_parser_benchmark.json"

    def _score_parser(
        self,
        parser: OrchestraSingleShotParser,
        cases: list[dict[str, object]],
        llm_provider: LLMProvider,
    ) -> dict[str, object]:
        def score_case(case: dict[str, object]) -> dict[str, object]:
            chunk = str(case["chunk"]).strip()
            expected_terms = sorted({str(term).lower().strip() for term in case.get("expected_terms", [])})

            records = parser.parse(chunk)
            observed_terms: set[str] = set()
            for record in records:
                for match in record.get("matched_skills", []):
                    observed_terms.add(str(match.get("canonical_name", "")).lower().strip())
                for term in record.get("missing_skills", []):
                    observed_terms.add(str(term).lower().strip())
            observed_terms.discard("")

            # Resolve observed <-> expected via the same tiered matcher the
            # production pipeline uses, treating this case's own expected_terms
            # as a tiny ad hoc skill cache (one SkillRecord per expected term,
            # no aliases). Exact/alias lookup first, embedding similarity as a
            # fallback - identical tiering to the real cache-matching path.
            matched_expected: set[str] = set()
            matched_observed: set[str] = set()
            if expected_terms:
                expected_records = [SkillRecord(name=term, aliases=(), related=()) for term in expected_terms]
                exact_matcher = ExactAliasMatcher(expected_records)
                try:
                    semantic_matcher: SemanticMatcher | None = SemanticMatcher(expected_records, llm_provider)
                except NotImplementedError:
                    semantic_matcher = None

                for observed_term in observed_terms:
                    candidates = exact_matcher.match(observed_term)
                    if not candidates and semantic_matcher is not None:
                        candidates = semantic_matcher.match(observed_term, context=chunk)
                    if candidates:
                        matched_observed.add(observed_term)
                        matched_expected.update(candidate.canonical_name for candidate in candidates)

            precision = (
                len(matched_observed) / len(observed_terms)
                if observed_terms
                else (1.0 if not expected_terms else 0.0)
            )
            recall = len(matched_expected) / len(expected_terms) if expected_terms else 1.0
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            perfect_match = precision == 1.0 and recall == 1.0

            return {
                "chunk": chunk,
                "expected_terms": expected_terms,
                "observed_terms": sorted(observed_terms),
                "unmatched_expected_terms": sorted(set(expected_terms) - matched_expected),
                "unmatched_observed_terms": sorted(observed_terms - matched_observed),
                "perfect_match": perfect_match,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Each case is independent, so score them concurrently instead of
        # waiting on one round trip at a time.
        with ThreadPoolExecutor(max_workers=min(8, len(cases)) or 1) as executor:
            case_results = list(executor.map(score_case, cases))

        perfect_matches = sum(1 for result in case_results if result["perfect_match"])
        precision_total = sum(result["precision"] for result in case_results)
        recall_total = sum(result["recall"] for result in case_results)
        f1_total = sum(result["f1"] for result in case_results)

        case_count = len(cases)
        return {
            "case_count": case_count,
            "perfect_match_rate": round(perfect_matches / case_count, 4) if case_count else 0.0,
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
                    executor.map(lambda name: self._score_parser(parsers[name], cases, llm), parsers.keys()),
                )
            )

        report = {
            "benchmark": "big_section_sentence_cases",
            "generated_at": datetime.now(UTC).isoformat(),
            "score_definition": (
                "mean per-case term-set F1 across the pre-split sentence cases; a term counts as "
                "matched if the tiered ExactAliasMatcher/SemanticMatcher resolves it to the expected "
                "term, not only on byte-identical strings"
            ),
            "pass_threshold": MIN_ORCHESTRA_F1_SCORE,
            "parsers": scored,
        }

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

        self.assertEqual(report["parsers"]["orchestra_single_shot"]["case_count"], len(cases))
        self.assertEqual(report["parsers"]["single_shot"]["case_count"], len(cases))
        # NOT a hard gate yet: sample_big_section_sentence_cases.yaml's expected_terms
        # are totalic ground truth again (2026-07-15), scored via the same tiered
        # matcher the production pipeline uses rather than exact string equality,
        # but no pass threshold has been calibrated against this scoring method yet.
        print(
            f"\norchestra_single_shot mean F1 vs. sample_big_section_sentence_cases.yaml "
            f"(matcher-based scoring, informational only, not a gate): "
            f"{report['parsers']['orchestra_single_shot']['mean_f1']:.2%} "
            f"(previous gate was {MIN_ORCHESTRA_F1_SCORE:.2%})"
        )



if __name__ == "__main__":
    unittest.main()
