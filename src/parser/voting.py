"""Self-consistency voting for LLM-based skill extraction.

Runs the same extraction call multiple times per chunk and keeps only
candidates that a majority of the independent samples agree on, reducing the
effect of per-call sampling variance (observed empirically: repeated
benchmark runs of the same parser scored anywhere from ~0.80 to ~0.90 F1 on
identical input).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def normalize_for_vote(value: str) -> str:
    return " ".join(value.lower().strip().split())


def majority_threshold(num_samples: int) -> int:
    """Smallest vote count that represents a strict majority of num_samples."""

    return (num_samples // 2) + 1


def vote_candidates(
    candidate_samples: Sequence[List[Dict[str, Any]]],
    min_votes: int,
) -> List[Dict[str, Any]]:
    """Keep candidates whose normalized raw_term appears in at least min_votes samples.

    Each sample is deduplicated by normalized raw_term before counting, so a
    single sample cannot inflate its own term's vote count.
    """

    vote_counts: Dict[str, int] = {}
    representative: Dict[str, Dict[str, Any]] = {}

    for sample in candidate_samples:
        seen_in_sample: set[str] = set()
        for candidate in sample:
            raw_term = str(candidate.get("raw_term", "")).strip()
            key = normalize_for_vote(raw_term)
            if not key or key in seen_in_sample:
                continue
            seen_in_sample.add(key)
            vote_counts[key] = vote_counts.get(key, 0) + 1
            representative.setdefault(key, candidate)

    return [representative[key] for key, count in vote_counts.items() if count >= min_votes]
