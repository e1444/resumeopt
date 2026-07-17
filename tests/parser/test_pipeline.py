"""Deterministic tests for src/parser/pipeline.py (no API key needed).

Uses a FakeLLMProvider stub (no real API calls) to validate the async
orchestration/dedup logic: Stage 1 extraction -> Stage 2 categorization ->
Stage 3a keyword-atomicity gate -> Stage 3b within-chunk redundancy check
(non-atomic terms only) -> final included/excluded verdicts, and cross-chunk
term deduplication. Model-quality questions (does the model actually
extract/categorize/atomicity/redundancy-check correctly) are out of scope
here - those need live-API benchmarks, consistent with this project's
convention of keeping provider/quality tests separate from orchestration
tests.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import unittest
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from chunker import split_into_sentence_chunks
from parser import run_parser_pipeline
from parser.categorization import CATEGORIES


class FakeLLMProvider(LLMProvider):
    """Stub provider - no real API calls.

    `extraction_by_chunk`: {chunk_text: [term, ...]} - Stage 1's canned output
    for a given chunk. Chunks not present default to no candidates.

    `category_by_term`: {term: category} - Stage 2's canned category for a
    given term (matched by its raw wording). Terms NOT present in this dict
    are simply OMITTED from the fake JSON response entirely (simulating a
    real missing-batch-result scenario), so the pipeline's own fail-safe
    ("uncategorized" -> excluded) can be exercised directly - not defaulted
    client-side in this stub.

    `atomicity_by_term`: {term: bool} - Stage 3a's canned atomic_keyword
    verdict for a given term. Terms NOT present are OMITTED from the fake
    response (same missing-batch-result convention), so the pipeline's own
    fail-safe (missing -> atomic_keyword=True, bypassing redundancy
    entirely) can be exercised directly.

    `redundancy_by_term`: {term: {"keep": bool, "redundant_with": [...]}} -
    Stage 3b's canned verdict for a given term. Terms NOT present are OMITTED
    from the fake response (same missing-batch-result convention), so the
    pipeline's own fail-safe ("keep=True") can be exercised directly.

    `chunks_for_text`: {posting_text: [chunk, ...]} - the fake LLM chunker's
    canned output for a given whole posting text. Texts NOT present default
    to the deterministic regex splitter's own output
    (`chunker.split_into_sentence_chunks`) - since every existing test's
    input text is already a single atomic sentence, this default preserves
    all pre-LLM-chunking test behavior unchanged; only tests specifically
    exercising LLM-chunking behavior need to override it.
    """

    def __init__(
        self,
        extraction_by_chunk: Optional[Dict[str, List[str]]] = None,
        category_by_term: Optional[Dict[str, str]] = None,
        atomicity_by_term: Optional[Dict[str, bool]] = None,
        redundancy_by_term: Optional[Dict[str, Dict[str, Any]]] = None,
        chunks_for_text: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__()
        self.extraction_by_chunk = extraction_by_chunk or {}
        self.category_by_term = category_by_term or {}
        self.atomicity_by_term = atomicity_by_term or {}
        self.redundancy_by_term = redundancy_by_term or {}
        self.chunks_for_text = chunks_for_text or {}
        self.calls: List[Dict[str, Any]] = []

    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def call_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
        few_shot_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        schema_name = (json_schema or {}).get("name")
        self.calls.append({"json_schema_name": schema_name, "prompt": prompt})

        if schema_name == "posting_chunks":
            match = re.search(r"Job posting:\n(.*)", prompt, re.DOTALL)
            posting_text = match.group(1).strip() if match else ""
            chunks = self.chunks_for_text.get(posting_text, split_into_sentence_chunks(posting_text))
            return {"chunks": chunks}
        if schema_name == "chunk_skill_extraction":
            match = re.search(r"Excerpt:\n(.*)", prompt, re.DOTALL)
            chunk = match.group(1).strip() if match else ""
            terms = self.extraction_by_chunk.get(chunk, [])
            return {
                "reasoning": "fake reasoning",
                "skills": [{"term": term, "reason": "fake reason"} for term in terms],
            }
        if schema_name == "skill_category_flags":
            terms = re.findall(r"term: '([^']*)'", prompt)
            return {
                "verdicts": [
                    {
                        "term": term,
                        "category": self.category_by_term[term],
                        "reason": "fake category reason",
                    }
                    for term in terms
                    if term in self.category_by_term
                ]
            }
        if schema_name == "keyword_atomicity_flags":
            terms = re.findall(r"term: '([^']*)'", prompt)
            return {
                "verdicts": [
                    {
                        "term": term,
                        "atomic_keyword": self.atomicity_by_term[term],
                        "reason": "fake atomicity reason",
                    }
                    for term in terms
                    if term in self.atomicity_by_term
                ]
            }
        if schema_name == "redundancy_flags":
            terms = re.findall(r"term: '([^']*)'", prompt)
            return {
                "verdicts": [
                    {
                        "term": term,
                        "keep": self.redundancy_by_term[term]["keep"],
                        "redundant_with": self.redundancy_by_term[term].get("redundant_with", []),
                        "reason": "fake redundancy reason",
                    }
                    for term in terms
                    if term in self.redundancy_by_term
                ]
            }
        raise AssertionError(f"unexpected schema in test: {schema_name}")


def _run(coro):
    return asyncio.run(coro)


class RunParserPipelineTest(unittest.TestCase):
    def test_included_term_survives_stage1_and_stage2(self) -> None:
        provider = FakeLLMProvider(
            extraction_by_chunk={"We use Python daily.": ["Python"]},
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        verdicts = _run(run_parser_pipeline(provider, "We use Python daily."))

        self.assertIn("python", verdicts)
        verdict = verdicts["python"]
        self.assertTrue(verdict.included)
        self.assertEqual(verdict.category, "resume_technical_skill")
        self.assertEqual(verdict.chunk, "We use Python daily.")

    def test_degree_or_qualification_category_is_excluded(self) -> None:
        chunk = "Degree in mathematics or equivalent experience."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Mathematics"]},
            category_by_term={"Mathematics": "degree_or_qualification"},
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        verdict = verdicts["mathematics"]
        self.assertFalse(verdict.included)
        self.assertEqual(verdict.category, "degree_or_qualification")

    def test_soft_skill_and_non_skill_categories_are_excluded(self) -> None:
        chunk = "Strong communication skills and a passion for excellence."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["communication skills", "passion for excellence"]},
            category_by_term={"communication skills": "soft_skill", "passion for excellence": "non_skill"},
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        self.assertFalse(verdicts["communication skills"].included)
        self.assertFalse(verdicts["passion for excellence"].included)

    def test_term_missing_from_stage2_result_defaults_to_excluded(self) -> None:
        # Fail-safe: a term Stage 1 extracted but Stage 2 never returned a
        # verdict for (e.g. a failed batch) must default to excluded, not
        # silently included.
        chunk = "We use Rust daily."
        provider = FakeLLMProvider(extraction_by_chunk={chunk: ["Rust"]}, category_by_term={})
        verdicts = _run(run_parser_pipeline(provider, chunk))

        verdict = verdicts["rust"]
        self.assertFalse(verdict.included)
        self.assertEqual(verdict.category, "uncategorized")

    def test_same_term_from_multiple_chunks_is_deduped_first_occurrence_wins(self) -> None:
        text = "We use Python daily. Our team loves Python too."
        provider = FakeLLMProvider(
            extraction_by_chunk={
                "We use Python daily.": ["Python"],
                "Our team loves Python too.": ["Python"],
            },
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        verdicts = _run(run_parser_pipeline(provider, text))

        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts["python"].chunk, "We use Python daily.")

    def test_empty_posting_returns_no_verdicts(self) -> None:
        provider = FakeLLMProvider()
        verdicts = _run(run_parser_pipeline(provider, ""))
        self.assertEqual(verdicts, {})

    def test_llm_chunker_splits_a_header_and_bullet_the_regex_splitter_would_merge(self) -> None:
        # The exact real-world defect this feature fixes: a section header
        # with no terminal punctuation, followed by a bullet - the regex
        # splitter merges these into one chunk; the (fake) LLM chunker here
        # is told to keep them separate.
        text = "What You Bring To The Table Strong Python skills."
        provider = FakeLLMProvider(
            chunks_for_text={text: ["What You Bring To The Table", "Strong Python skills."]},
            extraction_by_chunk={"Strong Python skills.": ["Python"]},
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        verdicts = _run(run_parser_pipeline(provider, text))

        self.assertTrue(verdicts["python"].included)
        self.assertEqual(verdicts["python"].chunk, "Strong Python skills.")
        extraction_calls = [c for c in provider.calls if c["json_schema_name"] == "chunk_skill_extraction"]
        self.assertEqual(len(extraction_calls), 2, "each LLM-split chunk should get its own extraction call")

    def test_ungrounded_llm_chunks_fall_back_to_the_regex_splitter(self) -> None:
        # If the LLM chunker hallucinates/paraphrases text that isn't a real
        # substring of the posting, the whole result is discarded in favor of
        # the deterministic regex splitter - never silently propagate an
        # ungrounded chunk.
        text = "We use Python daily."
        provider = FakeLLMProvider(
            chunks_for_text={text: ["We use some other language entirely."]},  # not grounded
            extraction_by_chunk={"We use Python daily.": ["Python"]},
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        verdicts = _run(run_parser_pipeline(provider, text))

        self.assertIn("python", verdicts)
        self.assertTrue(verdicts["python"].included)

    def test_llm_chunking_disabled_via_flag_never_calls_the_chunker(self) -> None:
        text = "We use Python daily."
        provider = FakeLLMProvider(
            extraction_by_chunk={text: ["Python"]},
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        verdicts = _run(run_parser_pipeline(provider, text, use_llm_chunking=False))

        self.assertTrue(verdicts["python"].included)
        chunking_calls = [c for c in provider.calls if c["json_schema_name"] == "posting_chunks"]
        self.assertEqual(len(chunking_calls), 0)

    def test_atomic_term_bypasses_redundancy_check_entirely(self) -> None:
        # Even though a redundancy verdict for this term says keep=False, an
        # atomic_keyword=True verdict must win - Stage 3b is never consulted.
        chunk = "Strong Machine Learning skills including regression and classification."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Machine Learning", "regression", "classification"]},
            category_by_term={
                "Machine Learning": "resume_technical_skill",
                "regression": "resume_technical_skill",
                "classification": "resume_technical_skill",
            },
            atomicity_by_term={"Machine Learning": True, "regression": False, "classification": False},
            redundancy_by_term={
                "Machine Learning": {"keep": False, "redundant_with": ["regression"]},
                "regression": {"keep": True, "redundant_with": []},
                "classification": {"keep": True, "redundant_with": []},
            },
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        verdict = verdicts["machine learning"]
        self.assertTrue(verdict.included)
        self.assertTrue(verdict.atomic_keyword)
        self.assertIsNone(verdict.redundant_with)

        redundancy_calls = [c for c in provider.calls if c["json_schema_name"] == "redundancy_flags"]
        self.assertEqual(len(redundancy_calls), 1)
        self.assertNotIn("term: 'Machine Learning'", redundancy_calls[0]["prompt"])

    def test_non_atomic_redundant_term_is_dropped_by_stage3b(self) -> None:
        # Both the general term AND its disambiguating sibling must be
        # non-atomic to even appear TOGETHER in Stage 3b's comparison batch -
        # if the sibling were atomic (bypassing Stage 3b entirely), it would
        # no longer be visible as a candidate for the redundancy judgment at
        # all (a real, benchmarked trade-off of the 2-step design: a general
        # term whose only specific sibling happens to be atomic now survives,
        # trading a little precision for higher recall).
        chunk = "We use Grafana for monitoring and dashboards."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["monitoring", "Grafana"]},
            category_by_term={"monitoring": "resume_technical_skill", "Grafana": "resume_technical_skill"},
            atomicity_by_term={"monitoring": False, "Grafana": False},
            redundancy_by_term={
                "monitoring": {"keep": False, "redundant_with": ["Grafana"]},
                "Grafana": {"keep": True, "redundant_with": []},
            },
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        self.assertTrue(verdicts["grafana"].included)
        verdict = verdicts["monitoring"]
        self.assertFalse(verdict.included)
        self.assertFalse(verdict.atomic_keyword)
        self.assertEqual(verdict.redundant_with, ["Grafana"])

    def test_non_atomic_non_redundant_terms_all_survive_stage3b(self) -> None:
        chunk = "We use Grafana and Prometheus for observability."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Grafana", "Prometheus"]},
            category_by_term={"Grafana": "resume_technical_skill", "Prometheus": "resume_technical_skill"},
            atomicity_by_term={"Grafana": False, "Prometheus": False},
            redundancy_by_term={
                "Grafana": {"keep": True, "redundant_with": []},
                "Prometheus": {"keep": True, "redundant_with": []},
            },
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        self.assertTrue(verdicts["grafana"].included)
        self.assertTrue(verdicts["prometheus"].included)

    def test_single_non_atomic_survivor_in_chunk_skips_the_redundancy_call_entirely(self) -> None:
        # A lone non-atomic survivor has no sibling to be redundant with -
        # redundancy.py's own <2-candidate fail-safe means no LLM call
        # should even be made.
        chunk = "We use Rust daily."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Rust"]},
            category_by_term={"Rust": "resume_technical_skill"},
            atomicity_by_term={"Rust": False},
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        self.assertTrue(verdicts["rust"].included)
        redundancy_calls = [c for c in provider.calls if c["json_schema_name"] == "redundancy_flags"]
        self.assertEqual(len(redundancy_calls), 0)

    def test_term_missing_from_stage3a_result_defaults_to_atomic_and_bypasses_redundancy(self) -> None:
        # Fail-safe: a term Stage 2 included but Stage 3a never returned a
        # verdict for (e.g. a failed batch) must default to
        # atomic_keyword=True (bypass redundancy, always keep) - not
        # silently excluded or sent through redundancy anyway.
        chunk = "We use Python daily."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Python"]},
            category_by_term={"Python": "resume_technical_skill"},
            atomicity_by_term={},  # Stage 3a batch "failed" - nothing returned
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        verdict = verdicts["python"]
        self.assertTrue(verdict.included)
        self.assertTrue(verdict.atomic_keyword)

    def test_term_missing_from_stage3b_result_defaults_to_keep(self) -> None:
        # Fail-safe: high recall / medium precision is explicitly acceptable
        # here - a non-atomic term Stage 3b never returned a verdict for must
        # default to keep=True, not silently excluded. Needs >=2 non-atomic
        # survivors in the chunk so the redundancy call is actually attempted
        # (a lone non-atomic survivor short-circuits to keep=True before any
        # call is made - see the single-survivor test above).
        chunk = "We use Grafana for monitoring and dashboards."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["monitoring", "Grafana"]},
            category_by_term={"monitoring": "resume_technical_skill", "Grafana": "resume_technical_skill"},
            atomicity_by_term={"monitoring": False, "Grafana": False},
            redundancy_by_term={},  # Stage 3b batch "failed" - nothing returned
        )
        verdicts = _run(run_parser_pipeline(provider, chunk))

        self.assertTrue(verdicts["monitoring"].included)
        self.assertEqual(verdicts["monitoring"].redundant_with, [])
        redundancy_calls = [c for c in provider.calls if c["json_schema_name"] == "redundancy_flags"]
        self.assertEqual(len(redundancy_calls), 1)

    def test_redundancy_check_disabled_via_flag_skips_both_stage3_calls(self) -> None:
        chunk = "Strong Python skills and Git-based development practices, e.g. version control."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["Python", "Git", "version control"]},
            category_by_term={
                "Python": "resume_technical_skill",
                "Git": "resume_technical_skill",
                "version control": "resume_technical_skill",
            },
            atomicity_by_term={"Python": True, "Git": True, "version control": False},
            redundancy_by_term={"version control": {"keep": False, "redundant_with": ["Git"]}},
        )
        verdicts = _run(run_parser_pipeline(provider, chunk, enable_redundancy_check=False))

        self.assertTrue(verdicts["version control"].included)
        atomicity_calls = [c for c in provider.calls if c["json_schema_name"] == "keyword_atomicity_flags"]
        redundancy_calls = [c for c in provider.calls if c["json_schema_name"] == "redundancy_flags"]
        self.assertEqual(len(atomicity_calls), 0)
        self.assertEqual(len(redundancy_calls), 0)

    def test_stage2_excluded_terms_never_reach_stage3(self) -> None:
        chunk = "We value strong communication skills and use Python daily."
        provider = FakeLLMProvider(
            extraction_by_chunk={chunk: ["communication skills", "Python"]},
            category_by_term={"communication skills": "soft_skill", "Python": "resume_technical_skill"},
            atomicity_by_term={"Python": True},
        )
        _run(run_parser_pipeline(provider, chunk))

        for schema_name in ("keyword_atomicity_flags", "redundancy_flags"):
            for call in [c for c in provider.calls if c["json_schema_name"] == schema_name]:
                self.assertNotIn("communication skills", call["prompt"])

    def test_multiple_chunks_each_get_their_own_extraction_and_category_calls(self) -> None:
        provider = FakeLLMProvider(
            extraction_by_chunk={
                "We use Python daily.": ["Python"],
                "We also use Rust.": ["Rust"],
            },
            category_by_term={"Python": "resume_technical_skill", "Rust": "resume_technical_skill"},
            atomicity_by_term={"Python": True, "Rust": True},
        )
        text = "We use Python daily. We also use Rust."
        verdicts = _run(run_parser_pipeline(provider, text))

        self.assertEqual(len(verdicts), 2)
        extraction_calls = [c for c in provider.calls if c["json_schema_name"] == "chunk_skill_extraction"]
        category_calls = [c for c in provider.calls if c["json_schema_name"] == "skill_category_flags"]
        self.assertEqual(len(extraction_calls), 2)
        self.assertEqual(len(category_calls), 2)

    def test_all_four_categories_are_recognized(self) -> None:
        self.assertEqual(
            set(CATEGORIES),
            {"resume_technical_skill", "degree_or_qualification", "soft_skill", "non_skill"},
        )


if __name__ == "__main__":
    unittest.main()
