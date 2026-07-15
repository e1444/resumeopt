"""Mechanical helpers for shaping and deduplicating LLM-extracted skill candidates.

These helpers are intentionally free of any skill-specific or posting-specific
knowledge (no hardcoded skill names, phrase tables, or noise lists). Parsers
remain responsible for their own extraction and matching judgment; this module
only validates shape and removes exact duplicates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def normalize_extraction_candidates(raw_candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    """Keep only well-shaped candidate dicts and drop duplicates by lowercased raw_term."""

    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue

        raw_term = str(raw_candidate.get("raw_term", "")).strip()
        if not raw_term:
            continue

        dedup_key = " ".join(raw_term.lower().split())
        if not dedup_key or dedup_key in seen:
            continue
        seen.add(dedup_key)

        normalized.append(dict(raw_candidate, raw_term=raw_term))

    return normalized
