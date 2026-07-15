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
_DISCARD_TERMS = {
    "data science solutions",
    "production-quality code",
    "technical tasks",
    "success metrics",
    "data-driven decisions",
    "learning culture",
    "data science tools and platforms",
    "technical challenges",
    "tools, packages, and libraries",
    "collaborating",
}
_HEURISTIC_HINTS = (
    "insurance",
    "segmentation",
    "governance",
    "optimization",
    "optimization",
    "processing",
    "retrieval",
    "telematics",
    "calibration",
    "forecasting",
    "feature engineering",
    "driver scoring",
    "monitoring",
    "seasonality",
    "underwriting",
    "document",
    "classification",
    "summarization",
    "workflow",
    "decision making",
    "queueing",
    "routing",
)


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

    def _normalize_candidate_term(self, raw_term: str) -> str:
        normalized = self._normalize(raw_term)
        normalized = re.sub(r"\bskills?\b$", "", normalized).strip()
        return self._normalize(normalized)

    def _discard_candidate_reason(self, raw_term: str) -> str:
        normalized = self._normalize_candidate_term(raw_term)
        if not normalized:
            return "empty_candidate"
        if normalized in _DISCARD_TERMS:
            return "generic_or_noise_term"
        if re.search(r"\b(data scientist|software engineer|machine learning engineer|product manager)\b", normalized):
            return "job_title_not_resume_skill"
        if re.search(r"\b(technical tasks|success metrics|learning culture|data-driven decisions|production-quality code)\b", normalized):
            return "generic_or_noise_term"
        return ""


class LLMPostingParser(DeterministicPostingParser):
    """LLM-backed parser that preserves deterministic post-processing."""

    def __init__(self, llm_provider: LLMProvider, skills_cache_path: Path = Path("data/skills.yaml")):
        super().__init__(skills_cache_path=skills_cache_path)
        self.llm = llm_provider

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        llm_chunks = self._split_chunks_llm(posting_text)
        deterministic_chunks = self._split_chunks(posting_text)
        chunks = self._merge_chunk_lists(llm_chunks, deterministic_chunks)
        if not chunks:
            chunks = [posting_text.strip() or posting_text]

        records: List[Dict[str, Any]] = []
        for chunk in chunks:
            extraction_candidates = self._extract_terms_llm_batch(chunk)
            heuristic_candidates = self._extract_terms_from_skill_list_chunk(chunk)
            if heuristic_candidates:
                extraction_candidates = self._normalize_extraction_candidates(
                    list(extraction_candidates) + heuristic_candidates
                )
            matched_skills, missing_skills, discarded_terms = self._match_extracted_terms_to_cache(
                chunk,
                extraction_candidates,
            )
            extracted_terms = [candidate["raw_term"] for candidate in extraction_candidates]

            if not matched_skills and not extraction_candidates:
                extracted_terms, matched_skills = self._extract_matches_for_chunk(chunk)
                missing_skills = []
                discarded_terms = []

            if not matched_skills and not extracted_terms and not missing_skills:
                continue

            records.append(
                {
                    "posting_line": chunk,
                    "extracted_raw_terms": extracted_terms,
                    "extraction_candidates": extraction_candidates,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "missing_skills_discarded": discarded_terms,
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

    def _merge_chunk_lists(self, primary_chunks: Sequence[str], fallback_chunks: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()

        for chunk in list(primary_chunks) + list(fallback_chunks):
            normalized = self._normalize(str(chunk))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(str(chunk).strip())

        return merged

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

    def _extract_terms_llm_batch(self, posting_text: str) -> List[Dict[str, Any]]:
        prompt = (
            "Extract resume-suitable skill terms from the full job posting in one batch. "
            "For each candidate, decide whether it should be included as a resume skill and whether it is a good cache-candidate term. "
            "Include concrete professional capabilities such as domain knowledge, tools, methods, certifications, "
            "technical skills, operational skills, and role-relevant soft skills that belong in a resume skills section. "
            "Exclude job titles, company values/culture language, compensation details, responsibilities phrased as tasks, "
            "and generic filler terms like efficiency/reliability/repeatability unless explicitly a required competency. "
            "Return JSON with exactly this shape: "
            "{\"candidates\": [{\"raw_term\":\"...\",\"category\":\"tool|language|framework|method|domain|certification|soft_skill|responsibility|quality|title|generic\","
            "\"include_for_resume_skills\":true,\"include_for_cache_candidate\":true,\"reason\":\"...\",\"evidence_quote\":\"...\"}]}"
            f"\n\nChunk:\n{posting_text}"
        )
        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt=(
                    "You extract resume-suitable skills across industries. "
                    "Return valid JSON only, do not invent terms, and keep entries concise."
                ),
                temperature=0.1,
                max_tokens=900,
            )
            candidates = payload.get("candidates", None)
            if isinstance(candidates, list):
                normalized = self._normalize_extraction_candidates(candidates)
                if normalized:
                    return normalized

            # Backward compatibility with older extraction shape.
            terms = payload.get("extracted_raw_terms", [])
            if isinstance(terms, list):
                return self._classify_legacy_terms_llm(
                    posting_text=posting_text,
                    terms=terms,
                    fallback_reason="legacy_extracted_raw_terms_shape",
                )
        except Exception:
            pass

        # Retry with a simpler legacy-compatible prompt if structured output is empty.
        fallback_prompt = (
            "Extract resume-suitable skill terms from the full job posting in one batch. "
            "Return JSON with exactly this shape: {\"extracted_raw_terms\": [\"...\"]}."
            f"\n\nChunk:\n{posting_text}"
        )
        try:
            payload = self.llm.call_json(
                prompt=fallback_prompt,
                system_prompt="You extract resume-suitable skills across industries. Return valid JSON only.",
                temperature=0.1,
                max_tokens=900,
            )
            terms = payload.get("extracted_raw_terms", [])
            if isinstance(terms, list):
                return self._classify_legacy_terms_llm(
                    posting_text=posting_text,
                    terms=terms,
                    fallback_reason="legacy_retry_extracted_raw_terms_shape",
                )
        except Exception:
            pass

        return []

    def _classify_legacy_terms_llm(
        self,
        posting_text: str,
        terms: Sequence[Any],
        fallback_reason: str,
    ) -> List[Dict[str, Any]]:
        cleaned_terms = [str(term).strip() for term in terms if str(term).strip()]
        if not cleaned_terms:
            return []

        prompt = (
            "Classify each extracted term for resume-skill relevance and cache-candidate usefulness. "
            "Mark generic qualities, role titles, and task-only phrases as excluded. "
            "Return JSON with exactly this shape: "
            "{\"candidates\": [{\"raw_term\":\"...\",\"category\":\"tool|language|framework|method|domain|certification|soft_skill|responsibility|quality|title|generic\","
            "\"include_for_resume_skills\":true,\"include_for_cache_candidate\":true,\"reason\":\"...\",\"evidence_quote\":\"...\"}]}"
            f"\n\nTerms:\n{cleaned_terms}"
            f"\n\nChunk:\n{posting_text}"
        )
        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt=(
                    "You classify extracted job-posting terms for resume skill inclusion. "
                    "Return valid JSON only and keep decisions concise."
                ),
                temperature=0.1,
                max_tokens=1200,
            )
            candidates = payload.get("candidates", None)
            if isinstance(candidates, list):
                normalized = self._normalize_extraction_candidates(candidates)
                if normalized:
                    return normalized
        except Exception:
            pass

        return self._normalize_extraction_candidates(
            [
                {
                    "raw_term": term,
                    "category": "unknown",
                    "include_for_resume_skills": True,
                    "include_for_cache_candidate": True,
                    "reason": fallback_reason,
                    "evidence_quote": "",
                }
                for term in cleaned_terms
            ]
        )

    def _extract_terms_from_skill_list_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        lowered = chunk.lower()
        if not ("/" in chunk or "e.g." in lowered or "nice to have" in lowered or "assets" in lowered):
            return []
        if not any(hint in lowered for hint in _HEURISTIC_HINTS):
            return []

        candidate_terms: List[str] = []
        normalized_chunk = re.sub(r"\b(?:e\.g\.|for example|including)\b", ",", chunk, flags=re.IGNORECASE)
        normalized_chunk = normalized_chunk.replace("(", ",").replace(")", ",")
        normalized_chunk = normalized_chunk.replace("/", ",")

        for segment in re.split(r"[,;]", normalized_chunk):
            term = segment.strip(" .:-")
            if not term:
                continue
            term = re.sub(r"\bexperience\b$", "", term, flags=re.IGNORECASE).strip()
            term = re.sub(r"\bfor underwriting.*$", "", term, flags=re.IGNORECASE).strip()
            term = re.sub(r"\bconcepts\b$", "", term, flags=re.IGNORECASE).strip()
            if not term:
                continue
            if len(term.split()) > 7:
                continue
            candidate_terms.append(term)

        if not candidate_terms:
            return []

        return self._normalize_extraction_candidates(
            [
                {
                    "raw_term": term,
                    "category": "heuristic",
                    "include_for_resume_skills": True,
                    "include_for_cache_candidate": True,
                    "reason": "heuristic_skill_list_extraction",
                    "evidence_quote": chunk,
                }
                for term in candidate_terms
            ]
        )

    def _normalize_extraction_candidates(self, raw_candidates: Sequence[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                continue

            raw_term = str(raw_candidate.get("raw_term", "")).strip()
            if not raw_term:
                continue

            normalized_key = self._normalize(raw_term)
            if not normalized_key or normalized_key in seen:
                continue
            seen.add(normalized_key)

            normalized.append(
                {
                    "raw_term": raw_term,
                    "category": str(raw_candidate.get("category", "unknown")).strip() or "unknown",
                    "include_for_resume_skills": bool(raw_candidate.get("include_for_resume_skills", False)),
                    "include_for_cache_candidate": bool(raw_candidate.get("include_for_cache_candidate", False)),
                    "reason": str(raw_candidate.get("reason", "")).strip(),
                    "evidence_quote": str(raw_candidate.get("evidence_quote", "")).strip(),
                }
            )

        return normalized

    def _match_extracted_terms_to_cache(
        self,
        posting_text: str,
        extraction_candidates: Iterable[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        seen_missing: set[str] = set()
        discarded: List[Dict[str, Any]] = []

        for candidate in extraction_candidates:
            raw_term = str(candidate.get("raw_term", "")).strip()
            if not raw_term:
                continue

            discard_reason = self._discard_candidate_reason(raw_term)
            if discard_reason:
                discarded.append(
                    {
                        "raw_term": raw_term,
                        "category": str(candidate.get("category", "unknown")),
                        "reason": discard_reason,
                        "include_for_resume_skills": False,
                        "include_for_cache_candidate": False,
                        "evidence_quote": str(candidate.get("evidence_quote", "")),
                    }
                )
                continue

            include_for_resume = bool(candidate.get("include_for_resume_skills", False))
            include_for_cache = bool(candidate.get("include_for_cache_candidate", False))
            if not include_for_resume or not include_for_cache:
                discarded.append(
                    {
                        "raw_term": raw_term,
                        "category": str(candidate.get("category", "unknown")),
                        "reason": str(candidate.get("reason", "excluded_by_extraction")),
                        "include_for_resume_skills": include_for_resume,
                        "include_for_cache_candidate": include_for_cache,
                        "evidence_quote": str(candidate.get("evidence_quote", "")),
                    }
                )
                continue

            normalized = self._normalize_candidate_term(raw_term)
            matches = self._term_lookup.get(normalized, [])

            if not matches:
                if normalized and normalized not in seen_missing:
                    missing.append(raw_term)
                    seen_missing.add(normalized)
                continue

            for canonical_name, match_type in matches:
                existing = grouped.get(canonical_name)
                if existing and _MATCH_PRIORITY[existing["match_type"]] >= _MATCH_PRIORITY[match_type]:
                    existing["frequency"] += 1
                    continue

                grouped[canonical_name] = {
                    "raw_term": raw_term,
                    "canonical_name": canonical_name,
                    "match_type": match_type,
                    "frequency": 1,
                    "evidence": posting_text,
                }

        finalized = [self._finalize_match(match) for match in grouped.values()]
        finalized.sort(key=lambda item: (-item["relevance_score"], item["canonical_name"]))
        return finalized, missing, discarded


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
