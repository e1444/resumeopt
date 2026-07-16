"""Orchestra single-shot parser: the primary parsing strategy.

Deterministic-only chunking (line splitting; no LLM-based re-chunking), with
each chunk getting its own independent, self-contained extraction+cache-match
call, run concurrently. Benchmarked to match-or-beat the retired chunk-by-chunk
"multishot" parser while being simpler and cheaper (no extra chunk-splitting
LLM call), and to avoid the fragmentation bug that came from re-splitting
already-atomic chunks with an LLM call (see DEV_PLAN.md history).

Extraction itself is delegated to `parallel_extraction.extract_with_parallel_classifiers`
(promoted to production 2026-07-15): a decomposition stage (broad, recall-only,
no classification judgment) followed by three independent, concurrently-run
classifiers (degree_context, domain_vs_technical, soft_skill), each judging
the full candidate list without seeing each other's verdicts. This replaced a
single monolithic extraction prompt after gap analysis against the LLM-judge
evaluation framework showed real instruction interference between the
prompt's competing exclusion rules; splitting into isolated stages measurably
reduced errors (21 -> 14 total invalid+missed on the same frozen judge/fixture,
see build/benchmarks/parallel_classifier_n1.json). `classifier_votes` controls
optional self-consistency voting per classifier path; benchmarked n=1 vs n=3
and found no measurable difference (see parallel_extraction.py's docstring),
so n=1 is the default.

`num_votes` (default 3, changed from 1 on 2026-07-15 - see below) is a
separate, outer self-consistency layer that repeats the WHOLE per-chunk
extraction attempt (decomposition + all classifiers) `num_votes` times and
majority-votes across samples. A live cost/latency measurement initially
showed extraction dominating pipeline wall-clock time (97% of a real run)
and scaling linearly with `num_votes x classifier_votes` LLM calls per
chunk when `max_workers` (the outer thread pool size) was too small to
absorb the extra tasks - `num_votes` was defaulted to 1 as a result.
Further testing showed this was a worker-pool sizing artifact, not an
inherent cost of voting: since every LLM call here is I/O-bound (waiting on
an API response, not CPU work), replicated votes run genuinely concurrently
as long as `max_workers` is large enough to hold them all in flight at
once. Empirically, num_votes=3 with max_workers=32 completed a real 22-chunk
posting in 16.0s - FASTER than num_votes=1 with max_workers=8 (23.1s) - while
num_votes=3 with the old max_workers=8 took 47.9s (throughput-bound by the
pool, not by voting itself). `num_votes` was therefore restored to 3 and
`max_workers` raised to 24, restoring the self-consistency benefit without
the latency regression. Real API rate limits (not just local thread-pool
size) still bound how far this scales; raise `max_workers` further only
with attention to the provider's actual rate-limit tier.

This class owns its cache-matching normalization logic directly (kept from
the retired multishot baseline): `_normalize_candidate_term`. Generic/noise-term
filtering is intentionally NOT a hardcoded keyword list (no `_DISCARD_TERMS`,
`_DEGREE_FIELD_NOISE`, `SOFT_SKILL_TERM_HINTS`, or job-title-regex style checks
here): a fixed list of words or a narrow regex tends to overfit to whatever
posting it was tuned against and doesn't generalize across careers. Instead,
the extraction/classification prompts themselves instruct the model to
exclude soft skills, abstract process language, job titles, and generic filler,
and each candidate's own `include_for_resume_skills`/`include_for_cache_candidate`
flags (plus `category`) are the mechanism for excluding that kind of noise.
The only remaining deterministic candidate-level check is grounding (is the
tern actually present in the source text), which is a factual check, not a
topical judgment call.

Cache matching itself is now tiered across two independently-testable
matchers from the `matcher` package: `ExactAliasMatcher` (free, instant, exact/alias/
related string lookup) runs first; when it finds nothing, `SemanticMatcher`
(embedding cosine-similarity) gets a chance before a candidate is given up on
as a `missing_skills` entry. This avoids needing to hand-enumerate every
phrasing variant (ipynb/jupyter, GLM/GBM-style abbreviations, degree-name
variants) as a cache alias just to be matched.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm import LLMProvider
from matcher import MATCH_PRIORITY, EmbeddingCache, MatchCandidate, SemanticMatcher

from .base import DeterministicPostingParser
from .candidate_utils import normalize_extraction_candidates
from .parallel_extraction import extract_with_parallel_classifiers
from .voting import majority_threshold, vote_candidates

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
        max_workers: int = 24,
        num_votes: int = 3,
        use_semantic_matching: bool = True,
        embedding_cache_path: Optional[Path] = Path("build/cache/skill_embeddings_cache.json"),
        classifier_votes: int = 1,
    ):
        super().__init__(skills_cache_path=skills_cache_path)
        self.llm = llm_provider
        self.max_workers = max(1, max_workers)
        self.num_votes = max(1, num_votes)
        self.classifier_votes = max(1, classifier_votes)
        self._semantic_matcher: Optional[SemanticMatcher] = None
        if use_semantic_matching:
            try:
                embedding_cache = EmbeddingCache(embedding_cache_path) if embedding_cache_path is not None else None
                self._semantic_matcher = SemanticMatcher(self._skills, llm_provider, embedding_cache=embedding_cache)
            except NotImplementedError:
                # Provider doesn't support embeddings (e.g. Anthropic, Ollama
                # today) - fall back to exact/alias-only matching.
                self._semantic_matcher = None

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
        flat_candidates = [result[0] for result in flat_results]
        flat_debug = [result[1] for result in flat_results]

        min_votes = majority_threshold(self.num_votes)
        records: List[Dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            start = chunk_index * self.num_votes
            samples = flat_candidates[start : start + self.num_votes]
            debug_samples = flat_debug[start : start + self.num_votes]
            extraction_candidates = (
                vote_candidates(samples, min_votes=min_votes) if self.num_votes > 1 else samples[0]
            )
            record = self._build_record_from_candidates(chunk, extraction_candidates, debug_samples)
            if record is not None:
                records.append(record)

        return records

    def _build_record_from_candidates(
        self,
        chunk: str,
        extraction_candidates: Sequence[Dict[str, Any]],
        extraction_debug_samples: Sequence[Dict[str, Any]] = (),
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
            "extraction_debug_samples": list(extraction_debug_samples),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "missing_skills_discarded": discarded_terms,
            "validation": self._build_validation(matched_skills),
        }

    def _extract_terms_llm_batch(self, posting_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidates, debug = extract_with_parallel_classifiers(self.llm, posting_text, n=self.classifier_votes)
        return self._normalize_extraction_candidates(candidates), debug

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
            matches: List[MatchCandidate] = self._exact_alias_matcher.match(normalized)
            if not matches and self._semantic_matcher is not None:
                matches = self._semantic_matcher.match(raw_term, context=posting_text)

            if not matches:
                display_terms = self._candidate_display_terms(posting_text, raw_term)
                for display_term in display_terms:
                    if display_term and display_term not in seen_missing:
                        missing.append(display_term)
                        seen_missing.add(display_term)
                continue

            for match_candidate in matches:
                canonical_name = match_candidate.canonical_name
                match_type = match_candidate.match_type
                existing = grouped.get(canonical_name)
                if existing and MATCH_PRIORITY[existing["match_type"]] >= MATCH_PRIORITY[match_type]:
                    existing["frequency"] += 1
                    continue

                grouped[canonical_name] = {
                    "raw_term": raw_term,
                    "canonical_name": canonical_name,
                    "match_type": match_type,
                    "base_confidence": match_candidate.confidence,
                    "frequency": 1,
                    "evidence": posting_text,
                }

        finalized = [self._finalize_match(match) for match in grouped.values()]
        finalized.sort(key=lambda item: (-item["relevance_score"], item["canonical_name"]))
        return finalized, missing, discarded
