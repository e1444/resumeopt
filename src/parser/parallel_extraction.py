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
- Stage 2 (`run_classifier_voted`, x4 run concurrently): four independent,
  narrowly-scoped classifiers - `degree_context`, `domain_vs_technical`,
  `soft_skill`, `genericity` - each judging the SAME full candidate list
  without seeing each other's verdicts. A candidate is excluded if ANY
  classifier flags it.

`genericity` was added 2026-07-15 after benchmark analysis found a real,
reproducible blind spot: phrases that reference an unnamed category rather
than a specific tool/technique (e.g. "new frameworks", "technical
methodologies") were flagged by NONE of the original 3 classifiers - each
one's rule was narrowly about a different axis (qualification-enumeration,
business-domain-vs-technical-field, soft-skill-vs-capability) and none of
them checked "does this even name a specific thing." Added as its own
classifier (not folded into `domain_vs_technical`) to keep each classifier's
rule singular and independently testable/tunable, consistent with why the
original single monolithic prompt was split up in the first place.

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

CLASSIFIER_NAMES: Tuple[str, ...] = ("degree_context", "domain_vs_technical", "soft_skill", "genericity")

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
        "labels that describe a business/industry area, not a technique or established technical field.\n"
        "This also includes generic administrative/operational nouns that only name the unspecified OBJECT of "
        "a reporting, documentation, or maintenance activity - e.g. a bare reference to 'setups', "
        "'procedures', or 'changes' being documented or updated, without naming any specific system, tool, or "
        "technique involved - exclude these the same way, since they describe what a routine business/"
        "operational activity is about rather than a technical skill."
    ),
    "soft_skill": (
        "For each candidate term below, decide whether it is EXCLUDED because it is a soft skill, abstract "
        "responsibility, job title, company value, or generic filler quality (efficiency, reliability, etc.) "
        "rather than a concrete, demonstrable technical capability."
    ),
    "genericity": (
        "For each candidate term below, decide whether it is EXCLUDED because it names an UNSPECIFIED "
        "instance of a category rather than a specific tool, technology, technique, or field. Apply this "
        "narrowly: only exclude a term that pairs an indefinite/vague modifier (new, various, other, "
        "additional, modern, relevant, similar, related, different, certain) with a bare category noun "
        "(frameworks, tools, technologies, methodologies, platforms, systems, software, languages, "
        "solutions) WITHOUT naming which specific one - e.g. 'new frameworks', 'various tools', 'relevant "
        "technologies', 'technical methodologies'. These are placeholders: they mention that a category "
        "exists without saying which member of it is meant, so a reader cannot point to any one concrete "
        "thing.\n"
        "Do NOT exclude a term merely because it sounds abstract, uses a generic-sounding word (concepts, "
        "theory, calibration, segmentation, measurement, pricing, analysis, optimization, etc.), or is a "
        "multi-word technical phrase - a specific, well-defined technique, field, or named process (e.g. "
        "model calibration, queueing concepts, machine learning, distributed systems) is NOT generic just "
        "because it is not a single short proper noun; it still names one identifiable thing. Only exclude "
        "the narrow 'indefinite modifier + bare category noun, no specific instance named' pattern described "
        "above - if in doubt, do NOT exclude."
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
) -> Dict[str, Dict[str, Any]]:
    """Run one classifier sample; returns {raw_term: {"excluded": bool, "reason": str}}.

    Missing terms default to not-excluded when consumed by callers.
    """

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
        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(flags, list):
            for item in flags:
                if not isinstance(item, dict):
                    continue
                raw_term = str(item.get("raw_term", "")).strip()
                if raw_term:
                    result[raw_term] = {
                        "excluded": bool(item.get("excluded", False)),
                        "reason": str(item.get("reason", "")),
                    }
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
) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
    """Run one classifier path `n` times (self-consistency) and majority-vote per candidate.

    Returns (final_verdict: {raw_term: excluded}, raw_votes: {raw_term: [{"excluded": bool,
    "reason": str}, ...]}) - raw_votes keeps every individual sample's verdict AND its
    textual reason, so callers can inspect exactly why a classifier did or didn't
    exclude a candidate, not just the final boolean.
    """

    n = max(1, n)
    with ThreadPoolExecutor(max_workers=min(max_workers, n) or 1) as executor:
        samples = list(
            executor.map(
                lambda _: _run_classifier_once(llm_provider, classifier_name, posting_text, candidate_terms),
                range(n),
            )
        )

    raw_votes: Dict[str, List[Dict[str, Any]]] = {term: [] for term in candidate_terms}
    for sample in samples:
        for term in candidate_terms:
            entry = sample.get(term, {"excluded": False, "reason": ""})
            raw_votes[term].append(entry)

    min_votes = majority_threshold(n)
    final_verdict = {
        term: sum(1 for vote in votes if vote["excluded"]) >= min_votes for term, votes in raw_votes.items()
    }
    return final_verdict, raw_votes


def extract_with_parallel_classifiers(
    llm_provider: LLMProvider,
    posting_text: str,
    n: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Full production extraction pipeline: decompose once, classify concurrently.

    Returns (candidates, debug):
    - `candidates`: dicts in the standard extraction-candidate shape
      (raw_term/category/include_for_resume_skills/include_for_cache_candidate/
      reason/evidence_quote), so downstream code (_normalize_extraction_candidates,
      _match_extracted_terms_to_cache) needs no changes. Excluded candidates are
      still returned (with include flags False) so they surface in
      missing_skills_discarded for auditability, same as before.
    - `debug`: the full intermediate representation - Stage 1's raw decomposition
      output and each of the 3 classifiers' per-term verdict+reason - so it's
      possible to inspect exactly why a candidate was included or excluded
      (which classifier(s) flagged it and what they said), not just the final
      merged decision. Callers that don't need this can ignore the second
      return value.
    """

    decomposed = decompose_candidates(llm_provider, posting_text)
    candidate_terms = [item["raw_term"] for item in decomposed]
    evidence_by_term = {item["raw_term"]: item["evidence_quote"] for item in decomposed}

    if not candidate_terms:
        return [], {"decomposition_candidates": decomposed, "classifier_verdicts": {}}

    with ThreadPoolExecutor(max_workers=len(CLASSIFIER_NAMES)) as executor:
        results = list(
            executor.map(
                lambda name: run_classifier_voted(llm_provider, name, posting_text, candidate_terms, n=n),
                CLASSIFIER_NAMES,
            )
        )

    verdicts_by_classifier = dict(zip(CLASSIFIER_NAMES, [result[0] for result in results]))
    raw_votes_by_classifier = dict(zip(CLASSIFIER_NAMES, [result[1] for result in results]))

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

    debug = {
        "decomposition_candidates": decomposed,
        "classifier_verdicts": {
            classifier_name: {
                term: {
                    "excluded": verdicts_by_classifier[classifier_name][term],
                    "votes": raw_votes_by_classifier[classifier_name][term],
                }
                for term in candidate_terms
            }
            for classifier_name in CLASSIFIER_NAMES
        },
    }

    return candidates, debug
