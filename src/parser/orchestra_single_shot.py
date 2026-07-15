"""Orchestra single-shot parser: the primary parsing strategy.

Deterministic-only chunking (line splitting; no LLM-based re-chunking), with
each chunk getting its own independent, self-contained extraction+cache-match
call, run concurrently. Benchmarked to match-or-beat the retired chunk-by-chunk
"multishot" parser while being simpler and cheaper (no extra chunk-splitting
LLM call), and to avoid the fragmentation bug that came from re-splitting
already-atomic chunks with an LLM call (see DEV_PLAN.md history).

This class owns its cache-matching normalization and noise-filtering rules
directly (kept from the retired multishot baseline): `_normalize_candidate_term`,
`_DEGREE_FIELD_NOISE`, `_DISCARD_TERMS`, `_discard_candidate_reason`. These are
deliberate, hand-tuned heuristics, not accidental complexity; keep them unless
a benchmark shows a specific improvement.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from llm import LLMProvider
from llm.schemas import EXTRACTION_CANDIDATES_JSON_SCHEMA

from .base import DeterministicPostingParser
from .candidate_utils import normalize_extraction_candidates
from .models import MATCH_PRIORITY
from .selection import SOFT_SKILL_TERM_HINTS
from .voting import majority_threshold, vote_candidates

_DISCARD_TERMS = {
    "data science",
    "data science solutions",
    "mathematics",
    "engineering",
    "operations research",
    "geomatics",
    "ai",
    "analytics",
    "data insights",
    "data-driven decision making",
    "data-driven decisions",
    "collaborating on technical challenges",
    "sharing best practices",
    "risk management",
    "scope alignment",
    "technical direction",
    "platform evolution",
    "data science tools",
    "telemedicine",
    "assets",
    "nice to have",
    "e.g",
    "apply sound statistical",
    "scientific practices",
    "strong scientific practices",
    "developing and validating models",
    "ambiguity to implementation",
    "from ambiguity to implementation",
    "code reviews",
    "documentation",
    "tools",
    "packages",
    "libraries",
    "problem solving skills",
    "problem solving",
    "framing problems",
    "communication skills",
    "organizational skills",
    "time management skills",
    "time management",
    "multi-project environment",
    "impact measurement",
    "text analytics",
    "usage-based insurance (ubi)",
    "project management",
    "agile methodology",
    "data analysis",
    "team leadership",
    "agile",
    "communication",
    "knowledge sharing",
    "continuous improvement",
    "production-quality code",
    "technical tasks",
    "success metrics",
    "learning culture",
    "technical challenges",
    "tools, packages, and libraries",
    "collaborating",
}

_DEGREE_FIELD_NOISE = {
    "mathematics",
    "engineering",
    "operations research",
    "geomatics",
    "ai",
}

_LEGACY_TERMS_JSON_SCHEMA = {
    "name": "legacy_extracted_terms",
    "schema": {
        "type": "object",
        "properties": {"extracted_raw_terms": {"type": "array", "items": {"type": "string"}}},
        "required": ["extracted_raw_terms"],
        "additionalProperties": False,
    },
}


class OrchestraSingleShotParser(DeterministicPostingParser):
    """Deterministic-only chunking; each chunk gets its own single-shot-style call."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        skills_cache_path: Path = Path("data/skills.yaml"),
        max_workers: int = 8,
        num_votes: int = 3,
    ):
        super().__init__(skills_cache_path=skills_cache_path)
        self.llm = llm_provider
        self.max_workers = max(1, max_workers)
        self.num_votes = max(1, num_votes)

    def parse(self, posting_text: str) -> List[Dict[str, Any]]:
        chunks = self._split_chunks(posting_text)
        if not chunks:
            chunks = [posting_text.strip() or posting_text]

        # Flatten (chunk, vote-attempt) into one task list so all extraction
        # calls across all chunks and all self-consistency votes run
        # concurrently together, not chunk-by-chunk or vote-by-vote.
        tasks = [chunk for chunk in chunks for _ in range(self.num_votes)]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks)) or 1) as executor:
            flat_results = list(executor.map(self._extract_terms_llm_batch, tasks))

        min_votes = majority_threshold(self.num_votes)
        records: List[Dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            start = chunk_index * self.num_votes
            samples = flat_results[start : start + self.num_votes]
            extraction_candidates = (
                vote_candidates(samples, min_votes=min_votes) if self.num_votes > 1 else samples[0]
            )
            record = self._build_record_from_candidates(chunk, extraction_candidates)
            if record is not None:
                records.append(record)

        return records

    def _build_record_from_candidates(
        self,
        chunk: str,
        extraction_candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
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
            return None

        return {
            "posting_line": chunk,
            "extracted_raw_terms": extracted_terms,
            "extraction_candidates": list(extraction_candidates),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "missing_skills_discarded": discarded_terms,
            "validation": self._build_validation(matched_skills),
        }

    def _extract_terms_llm_batch(self, posting_text: str) -> List[Dict[str, Any]]:
        prompt = (
            "Extract resume-suitable skill terms from the full job posting in one batch. "
            "Return only concrete, directly demonstrable skill terms that would belong in a resume skills section. "
            "Do not extract soft skills, general responsibilities, abstract process language, job titles, company values, compensation details, "
            "or generic filler phrases like efficiency/reliability/repeatability. "
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
                json_schema=EXTRACTION_CANDIDATES_JSON_SCHEMA,
            )
            candidates = payload.get("candidates", None)
            if isinstance(candidates, list):
                normalized = self._normalize_extraction_candidates(candidates)
                if normalized:
                    refined = self._refine_extraction_candidates_llm(posting_text=posting_text, candidates=normalized)
                    return refined or normalized

            # Backward compatibility with older extraction shape.
            terms = payload.get("extracted_raw_terms", [])
            if isinstance(terms, list) and terms:
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
            f"\n\nChunk:\n{posting_text}"
        )
        try:
            payload = self.llm.call_json(
                prompt=fallback_prompt,
                system_prompt="You extract resume-suitable skills across industries. Return valid JSON only.",
                temperature=0.1,
                max_tokens=900,
                json_schema=_LEGACY_TERMS_JSON_SCHEMA,
            )
            candidates = payload.get("candidates", None)
            if isinstance(candidates, list):
                normalized = self._normalize_extraction_candidates(candidates)
                if normalized:
                    return normalized

            terms = payload.get("extracted_raw_terms", [])
            if isinstance(terms, list) and terms:
                return self._classify_legacy_terms_llm(
                    posting_text=posting_text,
                    terms=terms,
                    fallback_reason="legacy_retry_extracted_raw_terms_shape",
                )
        except Exception:
            pass

        return []

    def _refine_extraction_candidates_llm(
        self,
        posting_text: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt = (
            "Refine the extracted skill candidates so they are the smallest complete skill phrases supported by the chunk. "
            "Keep multiword skill phrases intact when they are a single concept. "
            "Split only when the text clearly lists distinct skills. "
            "Do not output fragments of a larger skill phrase. "
            "If the chunk includes an acronym or alias in parentheses, include it when explicitly present. "
            f"\n\nChunk:\n{posting_text}"
            f"\n\nInitial candidates:\n{[candidate['raw_term'] for candidate in candidates]}"
        )

        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt=(
                    "You refine extracted skill terms. "
                    "Return valid JSON only and keep entries concise, complete, and grounded in the provided chunk."
                ),
                temperature=0.1,
                max_tokens=900,
                json_schema=EXTRACTION_CANDIDATES_JSON_SCHEMA,
            )
            refined_candidates = payload.get("candidates", None)
            if isinstance(refined_candidates, list):
                normalized = self._normalize_extraction_candidates(refined_candidates)
                if normalized:
                    return normalized
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
                json_schema=EXTRACTION_CANDIDATES_JSON_SCHEMA,
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

    def _normalize_extraction_candidates(self, raw_candidates: Sequence[Any]) -> List[Dict[str, Any]]:
        deduped = normalize_extraction_candidates(raw_candidates)
        normalized: List[Dict[str, Any]] = []
        for candidate in deduped:
            normalized.append(
                {
                    "raw_term": candidate["raw_term"],
                    "category": str(candidate.get("category", "unknown")).strip() or "unknown",
                    "include_for_resume_skills": bool(candidate.get("include_for_resume_skills", False)),
                    "include_for_cache_candidate": bool(candidate.get("include_for_cache_candidate", False)),
                    "reason": str(candidate.get("reason", "")).strip(),
                    "evidence_quote": str(candidate.get("evidence_quote", "")).strip(),
                }
            )
        return normalized

    def _normalize_candidate_term(self, raw_term: str) -> str:
        normalized = self._normalize(raw_term)
        normalized = re.sub(r"^strong\s+", "", normalized).strip()
        normalized = re.sub(r"^solid understanding of\s+", "", normalized).strip()
        normalized = re.sub(r"^experience\s+", "", normalized).strip()
        normalized = re.sub(r"^producing clear\s+", "", normalized).strip()
        normalized = re.sub(r"^comfortable explaining results and limitations.*$", "", normalized).strip()
        normalized = re.sub(r"\bskills?\b$", "", normalized).strip()
        normalized = re.sub(r"\bpractices?\b$", "", normalized).strip()
        normalized = re.sub(r"\bmethods?\b$", "", normalized).strip()
        normalized = re.sub(r"\bmethodology\b$", "", normalized).strip()
        normalized = re.sub(r"\band when to use them\b$", "", normalized).strip()
        normalized = re.sub(r"\bexperience\b$", "", normalized).strip()
        return self._normalize(normalized)

    def _candidate_display_terms(self, posting_text: str, raw_term: str) -> List[str]:
        normalized_term = self._normalize_candidate_term(raw_term)
        if not normalized_term:
            return []

        normalized_posting = self._normalize(posting_text)
        if normalized_term in normalized_posting:
            return [normalized_term]

        raw_lower = normalized_term.lower()
        if "/" in raw_lower:
            slash_parts = [part.strip() for part in raw_lower.split("/") if part.strip()]
            if len(slash_parts) == 2:
                left_part, right_part = slash_parts
                abbreviation_like = lambda value: value.isalpha() and (value.isupper() or len(value) <= 4)
                if abbreviation_like(left_part) and abbreviation_like(right_part):
                    return [left_part, right_part]

        if " and " in raw_lower:
            left_part, right_part = [part.strip() for part in raw_lower.split(" and ", 1)]
            left_tokens = left_part.split()
            right_tokens = right_part.split()
            if len(right_tokens) > 1 and len(left_tokens) <= 2:
                shared_head = right_tokens[-1]
                left_modifier_like = "-" in left_part or len(left_tokens) == 1 or left_part.isupper()
                if left_modifier_like:
                    return [f"{left_part} {shared_head}", right_part]
            return [raw_lower]

        match = re.search(re.escape(normalized_term), posting_text, flags=re.IGNORECASE)
        if match:
            tail = posting_text[match.end() :]
            tail_match = re.search(r"[\,;\)\.]", tail)
            if tail_match:
                expanded = posting_text[match.start() : match.end() + tail_match.start()].strip(" .:-")
                expanded_normalized = self._normalize(expanded)
                if expanded_normalized:
                    return [expanded_normalized]

        return [normalized_term]

    def _is_grounded_in_posting(self, posting_text: str, raw_term: str) -> bool:
        normalized_posting = self._normalize(posting_text)
        normalized_term = self._normalize_candidate_term(raw_term)
        if not normalized_term:
            return False
        if normalized_term in normalized_posting:
            return True

        compact_term = re.sub(r"[()]+", " ", normalized_term)
        compact_term = re.sub(r"\s+", " ", compact_term).strip()
        return compact_term in normalized_posting

    def _discard_candidate_reason(self, raw_term: str, posting_text: str = "") -> str:
        normalized = self._normalize_candidate_term(raw_term)
        if not normalized:
            return "empty_candidate"
        if "degree in a relevant discipline" in posting_text.lower() and normalized in _DEGREE_FIELD_NOISE:
            return "degree_field_not_skill"
        if normalized in SOFT_SKILL_TERM_HINTS or any(hint in normalized for hint in SOFT_SKILL_TERM_HINTS):
            return "soft_skill_not_section_skill"
        if normalized in _DISCARD_TERMS:
            return "generic_or_noise_term"
        if re.search(r"\b(data scientist|software engineer|machine learning engineer|product manager)\b", normalized):
            return "job_title_not_resume_skill"
        if re.search(r"\b(technical tasks|success metrics|learning culture|data-driven decisions|production-quality code)\b", normalized):
            return "generic_or_noise_term"
        return ""

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

            candidate_category = str(candidate.get("category", "unknown")).strip().lower()
            if candidate_category != "heuristic" and not self._is_grounded_in_posting(posting_text, raw_term):
                discarded.append(
                    {
                        "raw_term": raw_term,
                        "category": str(candidate.get("category", "unknown")),
                        "reason": "ungrounded_candidate",
                        "include_for_resume_skills": False,
                        "include_for_cache_candidate": False,
                        "evidence_quote": str(candidate.get("evidence_quote", "")),
                    }
                )
                continue

            category = str(candidate.get("category", "unknown")).strip().lower()
            if category == "soft_skill":
                discarded.append(
                    {
                        "raw_term": raw_term,
                        "category": str(candidate.get("category", "unknown")),
                        "reason": "soft_skill_not_section_skill",
                        "include_for_resume_skills": False,
                        "include_for_cache_candidate": False,
                        "evidence_quote": str(candidate.get("evidence_quote", "")),
                    }
                )
                continue

            discard_reason = self._discard_candidate_reason(raw_term, posting_text=posting_text)
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
                display_terms = self._candidate_display_terms(posting_text, raw_term)
                for display_term in display_terms:
                    if display_term and display_term not in seen_missing:
                        missing.append(display_term)
                        seen_missing.add(display_term)
                continue

            for canonical_name, match_type in matches:
                existing = grouped.get(canonical_name)
                if existing and MATCH_PRIORITY[existing["match_type"]] >= MATCH_PRIORITY[match_type]:
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
