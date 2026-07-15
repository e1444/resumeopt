"""Parallel multi-classifier extraction pipeline (promoted to production 2026-07-15).

Replaces the previous single monolithic extraction prompt. Gap analysis
(originally against an LLM-judge evaluation framework, since deprecated in
favor of the static fixture tests/evals/sample_big_section_sentence_cases.yaml
- see docs/agent/DEV_PLAN.md) showed a single prompt trying to simultaneously
decompose compound phrases and apply 3+ independent, sometimes-competing
exclusion rules (degree/qualification context, business-domain-vs-technical-
field, soft-skill framing) suffered real instruction interference - e.g.
front-loading one rule to fix it caused a new regression in unrelated
extraction within the same prompt. Splitting into isolated stages removed
that interference and measurably reduced errors.

Architecture:
- Stage 1 (`decompose_candidates`): broad, recall-only decomposition into
  atomic candidate phrases. No classification judgment at all, so it can't
  be interfered with by exclusion rules.
- Stage 2 (`run_classifier_voted`, x3 run concurrently): three independent,
  narrowly-scoped classifiers - `degree_context`, `domain_vs_technical`,
  `soft_skill` - each judging the SAME full candidate list without seeing
  each other's verdicts. A candidate is excluded if ANY classifier flags it.

n controls optional self-consistency voting per classifier path (majority
vote across n samples, reusing `voting.majority_threshold`). Benchmarked
n=1 vs n=3: extracted_terms were IDENTICAL across every test chunk, with
only 2 disagreements out of ~150+ individual votes - voting provided no
measurable benefit here (each classifier is already stable at temperature
0.1), so n=1 is the default, avoiding a 3x classifier-call cost for no
measured gain.

Prompts here use only abstract placeholder examples, not verbatim text from
any eval fixture, so measured improvements reflect genuine generalization
rather than fixture memorization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

from llm import LLMProvider

from .voting import majority_threshold

_DECOMPOSITION_JSON_SCHEMA = {
    "name": "decomposition_candidates",
    "schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw_term": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": ["raw_term", "evidence_quote"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
}

_CLASSIFIER_JSON_SCHEMA = {
    "name": "classifier_flags",
    "schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw_term": {"type": "string"},
                        "excluded": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["raw_term", "excluded", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["flags"],
        "additionalProperties": False,
    },
}

CLASSIFIER_NAMES: Tuple[str, ...] = ("degree_context", "domain_vs_technical", "soft_skill")

_CLASSIFIER_PROMPTS: Dict[str, str] = {
    "degree_context": (
        "For each candidate term below, decide whether it is EXCLUDED because it is merely one of several "
        "acceptable academic majors, fields of study, or credentials listed as a qualification requirement "
        "(e.g. 'a degree in X, Y, or Z, or equivalent experience') rather than a demonstrated skill. This "
        "applies only when the candidate appears ONLY in such a qualification-requirement enumeration - not "
        "when it is otherwise a genuine skill mention elsewhere in the chunk."
    ),
    "domain_vs_technical": (
        "For each candidate term below, decide whether it is EXCLUDED because it names a business/industry "
        "domain, product category, or general area of business focus rather than a technical skill - even if "
        "it sounds specific. Do NOT exclude a recognized technical/scientific field or discipline (e.g. "
        "machine learning, distributed systems, cybersecurity), even though it sounds broad - only exclude "
        "labels that describe a business/industry area, not a technique or established technical field."
    ),
    "soft_skill": (
        "For each candidate term below, decide whether it is EXCLUDED because it is a soft skill, abstract "
        "responsibility, job title, company value, or generic filler quality (efficiency, reliability, etc.) "
        "rather than a concrete, demonstrable technical capability."
    ),
}


def decompose_candidates(llm_provider: LLMProvider, posting_text: str) -> List[Dict[str, str]]:
    """Stage 1: broad, recall-only decomposition into atomic candidate phrases."""

    prompt = (
        "Decompose the following job-posting chunk into every atomic, resume-suitable skill-mention "
        "candidate. Extract the smallest complete, meaningful phrase for each distinct mention - split "
        "compound descriptive phrases into their separate constituent mentions (e.g. a phrase describing "
        "'X-based practices including Y and Z' should be decomposed into 'X', 'Y', and 'Z' as separate "
        "candidates, not kept as one combined phrase). Use the exact wording and tokens found in the chunk "
        "for each candidate - do not paraphrase, reword, or invent phrasing not drawn directly from the "
        "chunk's own text. Cast a wide net: include anything that could plausibly be a skill mention, even "
        "if uncertain - a later step filters out invalid ones.\n\n"
        f"Chunk:\n{posting_text}"
    )
    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt="You decompose job-posting text into atomic candidate phrases. Return valid JSON only.",
            temperature=0.1,
            max_tokens=900,
            json_schema=_DECOMPOSITION_JSON_SCHEMA,
        )
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return []

        cleaned: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            raw_term = str(item.get("raw_term", "")).strip()
            if not raw_term:
                continue
            key = raw_term.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({"raw_term": raw_term, "evidence_quote": str(item.get("evidence_quote", ""))})
        return cleaned
    except Exception:
        return []


def _run_classifier_once(
    llm_provider: LLMProvider,
    classifier_name: str,
    posting_text: str,
    candidate_terms: Sequence[str],
) -> Dict[str, bool]:
    """Run one classifier sample; returns {raw_term: excluded_bool}. Missing terms default to False (not excluded)."""

    if not candidate_terms:
        return {}

    prompt = (
        f"{_CLASSIFIER_PROMPTS[classifier_name]}\n\n"
        f"Chunk:\n{posting_text}\n\nCandidates:\n{list(candidate_terms)}"
    )
    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt=(
                "You classify candidate resume-skill terms against one specific rule. "
                "Return valid JSON only."
            ),
            temperature=0.1,
            max_tokens=600,
            json_schema=_CLASSIFIER_JSON_SCHEMA,
        )
        flags = payload.get("flags", [])
        result: Dict[str, bool] = {}
        if isinstance(flags, list):
            for item in flags:
                if not isinstance(item, dict):
                    continue
                raw_term = str(item.get("raw_term", "")).strip()
                if raw_term:
                    result[raw_term] = bool(item.get("excluded", False))
        return result
    except Exception:
        return {}


def run_classifier_voted(
    llm_provider: LLMProvider,
    classifier_name: str,
    posting_text: str,
    candidate_terms: Sequence[str],
    n: int = 1,
    max_workers: int = 8,
) -> Tuple[Dict[str, bool], Dict[str, List[bool]]]:
    """Run one classifier path `n` times (self-consistency) and majority-vote per candidate.

    Returns (final_verdict: {raw_term: excluded}, raw_votes: {raw_term: [bool, ...]}).
    """

    n = max(1, n)
    with ThreadPoolExecutor(max_workers=min(max_workers, n) or 1) as executor:
        samples = list(
            executor.map(
                lambda _: _run_classifier_once(llm_provider, classifier_name, posting_text, candidate_terms),
                range(n),
            )
        )

    raw_votes: Dict[str, List[bool]] = {term: [] for term in candidate_terms}
    for sample in samples:
        for term in candidate_terms:
            raw_votes[term].append(bool(sample.get(term, False)))

    min_votes = majority_threshold(n)
    final_verdict = {term: sum(votes) >= min_votes for term, votes in raw_votes.items()}
    return final_verdict, raw_votes


def extract_with_parallel_classifiers(
    llm_provider: LLMProvider,
    posting_text: str,
    n: int = 1,
) -> List[Dict[str, Any]]:
    """Full production extraction pipeline: decompose once, classify concurrently.

    Returns candidate dicts in the standard extraction-candidate shape
    (raw_term/category/include_for_resume_skills/include_for_cache_candidate/
    reason/evidence_quote), so downstream code (_normalize_extraction_candidates,
    _match_extracted_terms_to_cache) needs no changes. Excluded candidates are
    still returned (with include flags False) so they surface in
    missing_skills_discarded for auditability, same as before.
    """

    decomposed = decompose_candidates(llm_provider, posting_text)
    candidate_terms = [item["raw_term"] for item in decomposed]
    evidence_by_term = {item["raw_term"]: item["evidence_quote"] for item in decomposed}

    if not candidate_terms:
        return []

    with ThreadPoolExecutor(max_workers=len(CLASSIFIER_NAMES)) as executor:
        results = list(
            executor.map(
                lambda name: run_classifier_voted(llm_provider, name, posting_text, candidate_terms, n=n),
                CLASSIFIER_NAMES,
            )
        )

    verdicts_by_classifier = dict(zip(CLASSIFIER_NAMES, [result[0] for result in results]))

    exclusion_reasons: Dict[str, List[str]] = {}
    for classifier_name, verdict in verdicts_by_classifier.items():
        for term, excluded in verdict.items():
            if excluded:
                exclusion_reasons.setdefault(term, []).append(classifier_name)

    candidates: List[Dict[str, Any]] = []
    for term in candidate_terms:
        excluded_by = exclusion_reasons.get(term)
        include = excluded_by is None
        candidates.append(
            {
                "raw_term": term,
                "category": "unknown",
                "include_for_resume_skills": include,
                "include_for_cache_candidate": include,
                "reason": (
                    f"excluded_by:{','.join(excluded_by)}" if excluded_by else "passed_parallel_classifiers"
                ),
                "evidence_quote": evidence_by_term.get(term, ""),
            }
        )

    return candidates
