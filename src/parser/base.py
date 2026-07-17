"""Deterministic parser base: cache loading and cache-backed keyword matching.

This module contains only the parts of parsing that are truly deterministic
and skill-agnostic: loading the YAML cache, splitting text into line-based
chunks, and scoring matches. Term-lookup construction and matching itself
live in the `matcher` package (`ExactAliasMatcher`, `SemanticMatcher`), so
each matching strategy is independently testable and decoupled from parsing.
LLM extraction, cache-matching judgment, and any posting-specific heuristics
live in the individual parser implementations under src/parser/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from matcher import BASE_CONFIDENCE, BASE_RELEVANCE, MATCH_PRIORITY, ExactAliasMatcher, SkillRecord, normalize_term


def _clean_string_sequence(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"'{field}' must be a list when present")

    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"'{field}' entries must be strings")
        token = item.strip()
        if token:
            cleaned.append(token)
    return cleaned


def load_skill_cache(skills_cache_path: Path) -> List[SkillRecord]:
    """Load and validate the YAML skill cache into `SkillRecord`s.

    Shared, module-level entry point (not just a private parser method) so
    other callers - e.g. the webapp backend's skills-cache CRUD endpoints -
    can reuse the exact same validation (unique canonical names, `aliases`
    must be a list of strings) instead of reimplementing it.
    """

    with Path(skills_cache_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or []

    if not isinstance(payload, list):
        raise ValueError("Skill cache must be a YAML list of skill records")

    skills: List[SkillRecord] = []
    seen_names: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each skill record must be a mapping")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each skill record must include a non-empty 'name'")

        canonical_name = name.strip()
        lowered_name = canonical_name.lower()
        if lowered_name in seen_names:
            raise ValueError(f"Duplicate canonical skill name detected: {canonical_name}")
        seen_names.add(lowered_name)

        aliases = _clean_string_sequence(item.get("aliases", []), field="aliases")

        always_include = bool(item.get("always_include", False))

        skills.append(
            SkillRecord(
                name=canonical_name,
                aliases=tuple(aliases),
                always_include=always_include,
            )
        )

    return skills


class PostingParser(ABC):
    """Shared parser interface for deterministic and LLM-backed parsing."""

    @abstractmethod
    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        """Parse a job posting into schema-shaped line or chunk records."""


class DeterministicPostingParser(PostingParser):
    """Deterministic parser using cache-backed keyword matching."""

    def __init__(self, skills_cache_path: Path = Path("data/skills.yaml")):
        self.skills_cache_path = Path(skills_cache_path)
        self._skills = self._load_skill_cache(self.skills_cache_path)
        self._exact_alias_matcher = ExactAliasMatcher(self._skills)
        self._term_lookup = self._exact_alias_matcher.term_lookup
        self._ordered_terms = self._exact_alias_matcher.ordered_terms

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        chunks = self._split_chunks(posting_text)
        records: List[Dict[str, Any]] = []

        for chunk in chunks:
            extracted_raw_terms, matched_skills = self._extract_matches_for_chunk(chunk)
            if not matched_skills:
                continue

            records.append(
                {
                    "posting_line": chunk,
                    "extracted_raw_terms": extracted_raw_terms,
                    "matched_skills": matched_skills,
                    "validation": self._build_validation(matched_skills),
                }
            )

        return records

    def _load_skill_cache(self, skills_cache_path: Path) -> List[SkillRecord]:
        return load_skill_cache(skills_cache_path)

    def _clean_string_sequence(self, value: Any, field: str) -> List[str]:
        return _clean_string_sequence(value, field)

    def _build_term_lookup(self, skills: Sequence[SkillRecord]) -> Dict[str, List[Tuple[str, str]]]:
        """Deprecated: term-lookup construction now lives in `ExactAliasMatcher`.

        Kept only as a thin compatibility wrapper in case any external code
        still calls it directly.
        """

        return ExactAliasMatcher(skills).term_lookup

    def _split_chunks(self, posting_text: str) -> List[str]:
        chunks: List[str] = []
        for raw_line in posting_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[\-*\u2022\s]+", "", stripped).strip()
            if stripped:
                chunks.append(stripped)
        return chunks

    def _extract_matches_for_chunk(self, chunk: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        lowered_chunk = chunk.lower()
        grouped: Dict[str, Dict[str, Any]] = {}
        extracted_order: List[str] = []

        for term in self._ordered_terms:
            if not self._contains_term(lowered_chunk, term):
                continue

            raw_occurrence = self._extract_raw_occurrence(chunk, term)
            if raw_occurrence and raw_occurrence not in extracted_order:
                extracted_order.append(raw_occurrence)

            for canonical_name, match_type in self._term_lookup[term]:
                existing = grouped.get(canonical_name)
                if existing and MATCH_PRIORITY[existing["match_type"]] >= MATCH_PRIORITY[match_type]:
                    existing["frequency"] += 1
                    continue

                grouped[canonical_name] = {
                    "raw_term": raw_occurrence or term,
                    "canonical_name": canonical_name,
                    "match_type": match_type,
                    "frequency": 1,
                    "evidence": chunk,
                }

        matches = [self._finalize_match(match) for match in grouped.values()]
        matches.sort(key=lambda item: (-item["relevance_score"], item["canonical_name"]))
        return extracted_order, matches

    def _contains_term(self, lowered_chunk: str, term: str) -> bool:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, lowered_chunk) is not None

    def _extract_raw_occurrence(self, chunk: str, term: str) -> str:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.IGNORECASE)
        match = pattern.search(chunk)
        if not match:
            return ""
        return chunk[match.start() : match.end()]

    def _finalize_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        frequency = int(match.pop("frequency", 1))
        match_type = match["match_type"]

        # Semantic matches carry their own similarity-derived confidence
        # (set as `base_confidence` by the caller) instead of the fixed
        # per-match_type constant, since two semantic matches of the same
        # type can have very different similarity scores.
        base_confidence = match.get("base_confidence", BASE_CONFIDENCE.get(match_type, 0.5))
        confidence = min(1.0, base_confidence + 0.02 * max(0, frequency - 1))
        relevance_score = BASE_RELEVANCE.get(match_type, 3) + min(2, max(0, frequency - 1))

        return {
            "raw_term": match["raw_term"],
            "canonical_name": match["canonical_name"],
            "match_type": match_type,
            "confidence": round(confidence, 2),
            "relevance_score": relevance_score,
            "evidence": match["evidence"],
        }

    def _build_validation(self, matched_skills: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        canonical_names = [item["canonical_name"] for item in matched_skills]
        duplicates = sorted({name for name in canonical_names if canonical_names.count(name) > 1})

        if duplicates:
            return {
                "status": "fail",
                "notes": ["Duplicate canonical matches detected"],
                "issues": [
                    {
                        "type": "duplicate_skill",
                        "canonical_name": name,
                    }
                    for name in duplicates
                ],
            }

        return {
            "status": "pass",
            "notes": [
                "Matched skills are grounded in the posting chunk",
                "No duplicate canonical skills detected",
            ],
        }

    def _normalize(self, value: str) -> str:
        return normalize_term(value)
