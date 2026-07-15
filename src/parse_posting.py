"""Job posting parsing and cache-backed skill matching.

This module provides a class-based parser surface with two implementations:
- DeterministicPostingParser (default)
- LLMPostingParser (optional, reuses src/llm provider abstractions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from llm import LLMProvider


_MATCH_PRIORITY = {"exact": 3, "alias": 2, "related": 1}
_BASE_CONFIDENCE = {"exact": 0.98, "alias": 0.90, "related": 0.75}
_BASE_RELEVANCE = {"exact": 5, "alias": 4, "related": 3}


@dataclass(frozen=True)
class SkillRecord:
    """Canonical skill definition from the YAML cache."""

    name: str
    aliases: Tuple[str, ...]
    related: Tuple[str, ...]


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
        self._term_lookup = self._build_term_lookup(self._skills)
        self._ordered_terms = sorted(self._term_lookup.keys(), key=len, reverse=True)

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
        with skills_cache_path.open("r", encoding="utf-8") as handle:
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

            aliases = self._clean_string_sequence(item.get("aliases", []), field="aliases")
            related = self._clean_string_sequence(item.get("related", []), field="related")

            skills.append(
                SkillRecord(
                    name=canonical_name,
                    aliases=tuple(aliases),
                    related=tuple(related),
                )
            )

        return skills

    def _clean_string_sequence(self, value: Any, field: str) -> List[str]:
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

    def _build_term_lookup(self, skills: Sequence[SkillRecord]) -> Dict[str, List[Tuple[str, str]]]:
        lookup: Dict[str, List[Tuple[str, str]]] = {}

        for record in skills:
            self._append_lookup_term(lookup, record.name, record.name, "exact")
            for alias in record.aliases:
                self._append_lookup_term(lookup, alias, record.name, "alias")
            for related in record.related:
                self._append_lookup_term(lookup, related, record.name, "related")

        return lookup

    def _append_lookup_term(
        self,
        lookup: Dict[str, List[Tuple[str, str]]],
        raw_term: str,
        canonical_name: str,
        match_type: str,
    ) -> None:
        key = self._normalize(raw_term)
        if not key:
            return
        lookup.setdefault(key, []).append((canonical_name, match_type))

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
                if existing and _MATCH_PRIORITY[existing["match_type"]] >= _MATCH_PRIORITY[match_type]:
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

        confidence = min(1.0, _BASE_CONFIDENCE[match_type] + 0.02 * max(0, frequency - 1))
        relevance_score = _BASE_RELEVANCE[match_type] + min(2, max(0, frequency - 1))

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
        return " ".join(value.lower().strip().split())


class LLMPostingParser(DeterministicPostingParser):
    """LLM-backed parser that preserves deterministic post-processing."""

    def __init__(self, llm_provider: LLMProvider, skills_cache_path: Path = Path("data/skills.yaml")):
        super().__init__(skills_cache_path=skills_cache_path)
        self.llm = llm_provider

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        chunks = self._split_chunks_llm(posting_text)
        filtered_chunks = self._filter_chunks_llm(chunks)

        records: List[Dict[str, Any]] = []
        for chunk in filtered_chunks:
            matched_skills = self._extract_matched_skills_llm(chunk)
            extracted_terms = [match["raw_term"] for match in matched_skills]

            if not matched_skills:
                extracted_terms, matched_skills = self._extract_matches_for_chunk(chunk)

            if not matched_skills:
                continue

            records.append(
                {
                    "posting_line": chunk,
                    "extracted_raw_terms": extracted_terms,
                    "matched_skills": matched_skills,
                    "validation": self._build_validation(matched_skills),
                }
            )

        return records

    def _split_chunks_llm(self, posting_text: str) -> List[str]:
        prompt = (
            "Split the job posting into meaningful chunks for skill extraction. "
            "Return JSON with exactly this shape: {\"chunks\": [\"...\"]}."
            f"\n\nPosting:\n{posting_text}"
        )
        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt="You are a strict parser. Return valid JSON only.",
                temperature=0.1,
                max_tokens=1200,
            )
            chunks = payload.get("chunks", [])
            if isinstance(chunks, list):
                cleaned = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
                if cleaned:
                    return cleaned
        except Exception:
            pass

        return self._split_chunks(posting_text)

    def _filter_chunks_llm(self, chunks: Sequence[str]) -> List[str]:
        prompt = (
            "Keep only chunks likely to contain technical or professional skills. "
            "Return JSON with exactly this shape: {\"kept_chunks\": [\"...\"]}."
            f"\n\nChunks:\n{list(chunks)}"
        )
        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt="You are a strict parser. Return valid JSON only.",
                temperature=0.1,
                max_tokens=1000,
            )
            kept = payload.get("kept_chunks", [])
            if isinstance(kept, list):
                cleaned = [str(chunk).strip() for chunk in kept if str(chunk).strip()]
                if cleaned:
                    return cleaned
        except Exception:
            pass

        return [chunk for chunk in chunks if any(token in chunk.lower() for token in self._ordered_terms)]

    def _extract_matched_skills_llm(self, chunk: str) -> List[Dict[str, Any]]:
        cache_records = self._cache_prompt_records()
        prompt = (
            "Given a job posting chunk and a canonical skills cache, return only matched skills "
            "that exist in the cache. "
            "Return JSON with exactly this shape: "
            "{\"matched_skills\": [{\"raw_term\":\"...\",\"canonical_name\":\"...\","
            "\"match_type\":\"exact|alias|related\",\"confidence\":0.0,"
            "\"relevance_score\":0,\"evidence\":\"...\"}]}."
            f"\n\nChunk:\n{chunk}\n\nSkills Cache:\n{cache_records}"
        )
        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt="You are a strict parser. Return valid JSON only.",
                temperature=0.1,
                max_tokens=900,
            )
            return self._sanitize_llm_matches(chunk, payload.get("matched_skills", []))
        except Exception:
            pass

        return []

    def _cache_prompt_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "aliases": list(skill.aliases),
                "related": list(skill.related),
            }
            for skill in self._skills
        ]

    def _sanitize_llm_matches(self, chunk: str, raw_matches: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_matches, list):
            return []

        canonical_by_normalized = {self._normalize(skill.name): skill.name for skill in self._skills}
        grouped: Dict[str, Dict[str, Any]] = {}

        for raw_match in raw_matches:
            if not isinstance(raw_match, dict):
                continue

            canonical_name = raw_match.get("canonical_name")
            if not isinstance(canonical_name, str):
                continue
            canonical_key = self._normalize(canonical_name)
            if canonical_key not in canonical_by_normalized:
                continue

            resolved_canonical = canonical_by_normalized[canonical_key]
            match_type = str(raw_match.get("match_type", "related")).strip().lower()
            if match_type not in _MATCH_PRIORITY:
                match_type = "related"

            raw_term = str(raw_match.get("raw_term", resolved_canonical)).strip() or resolved_canonical
            evidence = str(raw_match.get("evidence", chunk)).strip() or chunk

            existing = grouped.get(resolved_canonical)
            if existing and _MATCH_PRIORITY[existing["match_type"]] >= _MATCH_PRIORITY[match_type]:
                existing["frequency"] += 1
                continue

            grouped[resolved_canonical] = {
                "raw_term": raw_term,
                "canonical_name": resolved_canonical,
                "match_type": match_type,
                "frequency": 1,
                "evidence": evidence,
            }

        finalized = [self._finalize_match(match) for match in grouped.values()]
        finalized.sort(key=lambda item: (-item["relevance_score"], item["canonical_name"]))
        return finalized


def select_skills(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one strongest match per canonical skill from parsed chunk records."""

    strongest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for match in record.get("matched_skills", []):
            canonical_name = match.get("canonical_name")
            if not isinstance(canonical_name, str) or not canonical_name.strip():
                continue

            existing = strongest.get(canonical_name)
            if existing is None:
                strongest[canonical_name] = match
                continue

            existing_rank = (
                int(existing.get("relevance_score", 0)),
                float(existing.get("confidence", 0.0)),
                _MATCH_PRIORITY.get(str(existing.get("match_type", "related")), 0),
            )
            candidate_rank = (
                int(match.get("relevance_score", 0)),
                float(match.get("confidence", 0.0)),
                _MATCH_PRIORITY.get(str(match.get("match_type", "related")), 0),
            )
            if candidate_rank > existing_rank:
                strongest[canonical_name] = match

    selected = list(strongest.values())
    selected.sort(
        key=lambda item: (
            -int(item.get("relevance_score", 0)),
            -float(item.get("confidence", 0.0)),
            item.get("canonical_name", ""),
        )
    )
    return selected


def validate_selected_skills(
    records: Sequence[Dict[str, Any]],
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    min_confidence: float = 0.7,
    max_unique_skills: int = 12,
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Validate final selected skills against cache, confidence, and grounding constraints."""

    selected_skills = select_skills(records)
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)
    canonical_cache = {parser._normalize(skill.name) for skill in parser._skills}
    cache_by_canonical = {parser._normalize(skill.name): skill for skill in parser._skills}
    posting_lower = posting_text.lower()

    if not selected_skills:
        issues.append({"type": "empty_selection", "message": "No skills selected"})

    if len(selected_skills) > max_unique_skills:
        issues.append(
            {
                "type": "too_many_skills",
                "count": len(selected_skills),
                "max_allowed": max_unique_skills,
            }
        )

    for match in selected_skills:
        canonical_name = str(match.get("canonical_name", "")).strip()
        canonical_key = parser._normalize(canonical_name)
        if canonical_key not in canonical_cache:
            issues.append(
                {
                    "type": "unsupported_skill",
                    "canonical_name": canonical_name,
                }
            )
            continue

        confidence = float(match.get("confidence", 0.0))
        if confidence < min_confidence:
            issues.append(
                {
                    "type": "weak_match",
                    "canonical_name": canonical_name,
                    "confidence": confidence,
                    "minimum_confidence": min_confidence,
                }
            )

        evidence = str(match.get("evidence", "")).strip().lower()
        raw_term = str(match.get("raw_term", "")).strip().lower()
        canonical_lower = canonical_name.lower()

        grounded_deterministic = (
            (bool(evidence) and evidence in posting_lower)
            or (bool(raw_term) and raw_term in posting_lower)
            or (bool(canonical_lower) and canonical_lower in posting_lower)
        )

        if grounded_deterministic:
            continue

        llm_grounded = False
        if llm_provider is not None:
            skill_record = cache_by_canonical.get(canonical_key)
            llm_grounded = _llm_validate_skill_grounding(
                llm_provider=llm_provider,
                posting_text=posting_text,
                canonical_name=canonical_name,
                aliases=list(skill_record.aliases) if skill_record else [],
                related=list(skill_record.related) if skill_record else [],
                raw_term=raw_term,
                evidence=evidence,
            )

        if llm_grounded:
            notes.append(
                f"LLM grounding accepted edge-case match for '{canonical_name}'"
            )
            continue

        if not evidence and not raw_term:
            issues.append(
                {
                    "type": "missing_grounding",
                    "canonical_name": canonical_name,
                    "message": "Match lacks evidence and raw term",
                }
            )
        elif (
            evidence
            and evidence not in posting_lower
            and raw_term not in posting_lower
            and canonical_lower not in posting_lower
        ):
            issues.append(
                {
                    "type": "missing_grounding",
                    "canonical_name": canonical_name,
                    "message": "Evidence or term not found in posting text and LLM grounding did not confirm",
                }
            )

    if issues:
        notes.append("Validation failed with one or more issues")
        return {
            "status": "fail",
            "notes": notes,
            "issues": issues,
            "selected_skills": selected_skills,
        }

    notes.append("Selected skills pass cache, confidence, and grounding checks")
    return {
        "status": "pass",
        "notes": notes,
        "issues": [],
        "selected_skills": selected_skills,
    }


def _llm_validate_skill_grounding(
    llm_provider: LLMProvider,
    posting_text: str,
    canonical_name: str,
    aliases: Sequence[str],
    related: Sequence[str],
    raw_term: str,
    evidence: str,
) -> bool:
    """Use an LLM to validate semantic grounding for borderline matches."""

    prompt = (
        "Determine whether the skill is actually supported by the posting text. "
        "Return JSON only with format: "
        '{"is_grounded": true|false, "reason": "..."}.\n\n'
        f"Posting Text:\n{posting_text}\n\n"
        f"Skill Canonical Name: {canonical_name}\n"
        f"Skill Aliases: {list(aliases)}\n"
        f"Skill Related Terms: {list(related)}\n"
        f"Parser Raw Term: {raw_term}\n"
        f"Parser Evidence: {evidence}\n"
        "Consider alias and related-term edge cases such as ipynb indicating Jupyter."
    )

    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt=(
                "You validate grounding for resume skills. "
                "Return valid JSON only and do not infer unsupported skills."
            ),
            temperature=0.1,
            max_tokens=300,
        )
    except Exception:
        return False

    return bool(payload.get("is_grounded", False))


def parse_posting(
    posting_text: str,
    skills_cache_path: Path = Path("data/skills.yaml"),
    llm_provider: Optional[LLMProvider] = None,
    use_llm: bool = False,
) -> List[Dict[str, Any]]:
    """Parse a job posting with deterministic default behavior."""

    parser: PostingParser
    if use_llm and llm_provider is not None:
        parser = LLMPostingParser(llm_provider=llm_provider, skills_cache_path=skills_cache_path)
    else:
        parser = DeterministicPostingParser(skills_cache_path=skills_cache_path)

    return parser.parse(posting_text)
