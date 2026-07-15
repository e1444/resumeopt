"""Shared scoring helpers for parser benchmark tests.

Kept separate from any single benchmark test file so multiple experiments can
reuse identical precision/recall/F1 accounting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence


def score_term_sets(observed_terms: set[str], expected_terms: set[str]) -> Dict[str, Any]:
    intersection = observed_terms & expected_terms
    precision = len(intersection) / len(observed_terms) if observed_terms else (1.0 if not expected_terms else 0.0)
    recall = len(intersection) / len(expected_terms) if expected_terms else 1.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "exact_match": observed_terms == expected_terms,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def observed_terms_from_records(records: Sequence[Dict[str, Any]]) -> set[str]:
    observed: set[str] = set()
    for record in records:
        for match in record.get("matched_skills", []):
            observed.add(str(match.get("canonical_name", "")).lower().strip())
        for term in record.get("missing_skills", []):
            observed.add(str(term).lower().strip())
    return observed


def score_parser_on_cases(parser: Any, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score a parser against pre-split, atomic per-case sentence fixtures.

    Each case's chunk is parsed independently (parallelized since cases are
    independent), matching the primary benchmark's methodology.
    """

    def score_case(case: Dict[str, Any]) -> Dict[str, Any]:
        chunk = str(case["chunk"]).strip()
        expected_terms = {str(term).lower().strip() for term in case.get("expected_terms", [])}
        records = parser.parse(chunk)
        observed_terms = observed_terms_from_records(records)
        result = score_term_sets(observed_terms, expected_terms)
        result.update(
            {
                "chunk": chunk,
                "expected_terms": sorted(expected_terms),
                "observed_terms": sorted(observed_terms),
            }
        )
        return result

    with ThreadPoolExecutor(max_workers=min(8, len(cases)) or 1) as executor:
        case_results = list(executor.map(score_case, cases))

    return _aggregate(case_results)


def score_parser_on_combined_posting(parser: Any, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score a parser on all case chunks joined into one multi-line posting.

    This exercises chunk-splitting behavior (deterministic line splitting,
    optional LLM-based re-chunking) the way a real multi-bullet posting
    section would, unlike the atomic per-case benchmark where each chunk is
    already a single unit.
    """

    combined_text = "\n".join(str(case["chunk"]).strip() for case in cases)
    expected_terms = {
        str(term).lower().strip() for case in cases for term in case.get("expected_terms", [])
    }

    records = parser.parse(combined_text)
    observed_terms = observed_terms_from_records(records)
    result = score_term_sets(observed_terms, expected_terms)
    result.update(
        {
            "expected_terms": sorted(expected_terms),
            "observed_terms": sorted(observed_terms),
            "record_count": len(records),
        }
    )
    return result


def _aggregate(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    case_count = len(case_results)
    exact_matches = sum(1 for result in case_results if result["exact_match"])
    precision_total = sum(result["precision"] for result in case_results)
    recall_total = sum(result["recall"] for result in case_results)
    f1_total = sum(result["f1"] for result in case_results)

    return {
        "case_count": case_count,
        "exact_match_rate": round(exact_matches / case_count, 4) if case_count else 0.0,
        "mean_precision": round(precision_total / case_count, 4) if case_count else 0.0,
        "mean_recall": round(recall_total / case_count, 4) if case_count else 0.0,
        "mean_f1": round(f1_total / case_count, 4) if case_count else 0.0,
        "cases": case_results,
    }
