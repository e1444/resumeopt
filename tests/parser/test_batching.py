"""Deterministic tests for the multi-chunk batching capability in
`parser.extraction`/`parser.categorization` (no API key needed).

Batching (multiple chunks per call, via a `batch_size` argument) is the
PRODUCTION DEFAULT (`batch_size=6`) as of 2026-07-17 - see `extraction.py`'s
module docstring for the full cost/quality analysis behind this decision
(large, consistent token/call savings; a term-level inspection found most
of an initial apparent recall drop was phrasing/granularity variance and
over-fragmentation cleanup rather than genuine loss; a separate benchmark
found batching IMPROVED F1 on a simpler posting). These tests cover the
mechanics of the batching capability itself (grouping, index-based response
mapping, fail-safes), independent of that quality finding.
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
from parser.categorization import categorize_candidates_for_chunks
from parser.extraction import extract_candidates_for_chunks


class BatchingFakeLLMProvider(LLMProvider):
    def __init__(
        self,
        extraction_by_chunk: Optional[Dict[str, List[str]]] = None,
        category_by_term: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.extraction_by_chunk = extraction_by_chunk or {}
        self.category_by_term = category_by_term or {}
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
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        schema_name = (json_schema or {}).get("name")
        self.calls.append({"json_schema_name": schema_name, "prompt": prompt})

        if schema_name == "chunk_skill_extraction":
            excerpts = re.findall(r"\d+\. excerpt: '((?:[^'\\]|\\.)*)'", prompt)
            return {
                "excerpts": [
                    {
                        "index": i,
                        "reasoning": "fake reasoning",
                        "skills": [
                            {"term": term, "reason": "fake reason"}
                            for term in self.extraction_by_chunk.get(chunk, [])
                        ],
                    }
                    for i, chunk in enumerate(excerpts, start=1)
                ]
            }
        if schema_name == "skill_category_flags":
            excerpt_blocks = re.split(r"\d+\. excerpt \(local context\): ", prompt)[1:]
            return {
                "excerpts": [
                    {
                        "index": i,
                        "verdicts": [
                            {
                                "term": term,
                                "category": self.category_by_term[term],
                                "reason": "fake category reason",
                            }
                            for term in re.findall(r"- term: '((?:[^'\\]|\\.)*)'", block)
                            if term in self.category_by_term
                        ],
                    }
                    for i, block in enumerate(excerpt_blocks, start=1)
                ]
            }
        raise AssertionError(f"unexpected schema in test: {schema_name}")


def _run(coro):
    return asyncio.run(coro)


class ExtractionBatchingTest(unittest.TestCase):
    def test_default_batch_size_groups_chunks_into_one_call(self) -> None:
        chunks = ["We use Python daily.", "We also use Rust."]
        provider = BatchingFakeLLMProvider(
            extraction_by_chunk={"We use Python daily.": ["Python"], "We also use Rust.": ["Rust"]}
        )
        results = _run(extract_candidates_for_chunks(provider, chunks))

        self.assertEqual(len(provider.calls), 1, "batch_size defaults to 6 - both chunks fit in one call")
        self.assertEqual([r["terms"] for r in results], [["Python"], ["Rust"]])

    def test_explicit_batch_size_of_one_gives_one_call_per_chunk(self) -> None:
        chunks = ["We use Python daily.", "We also use Rust.", "We also use Go."]
        provider = BatchingFakeLLMProvider(
            extraction_by_chunk={
                "We use Python daily.": ["Python"],
                "We also use Rust.": ["Rust"],
                "We also use Go.": ["Go"],
            }
        )
        results = _run(extract_candidates_for_chunks(provider, chunks, batch_size=1))

        self.assertEqual(len(provider.calls), 3)
        self.assertEqual([r["terms"] for r in results], [["Python"], ["Rust"], ["Go"]])

    def test_chunk_missing_from_batch_response_defaults_to_no_terms(self) -> None:
        chunks = ["We use Python daily.", "We also use Rust."]
        # Only Python's terms are canned - Rust's index will be absent from
        # the fake response entirely, simulating a partial-response gap.
        provider = BatchingFakeLLMProvider(extraction_by_chunk={"We use Python daily.": ["Python"]})
        results = _run(extract_candidates_for_chunks(provider, chunks, batch_size=2))

        self.assertEqual(results[0]["terms"], ["Python"])
        self.assertEqual(results[1]["terms"], [])


class CategorizationBatchingTest(unittest.TestCase):
    def test_default_batch_size_groups_chunks_into_one_call(self) -> None:
        chunk_terms = [
            {"chunk": "We use Python daily.", "terms": ["Python"]},
            {"chunk": "We also use Rust.", "terms": ["Rust"]},
        ]
        provider = BatchingFakeLLMProvider(
            category_by_term={"Python": "resume_technical_skill", "Rust": "resume_technical_skill"}
        )
        results = _run(categorize_candidates_for_chunks(provider, chunk_terms))

        self.assertEqual(len(provider.calls), 1, "batch_size defaults to 6 - both chunks fit in one call")
        self.assertEqual(results[0]["Python"]["category"], "resume_technical_skill")
        self.assertEqual(results[1]["Rust"]["category"], "resume_technical_skill")

    def test_explicit_batch_size_groups_multiple_chunks_per_call(self) -> None:
        chunk_terms = [
            {"chunk": "We use Python daily.", "terms": ["Python"]},
            {"chunk": "We also use Rust.", "terms": ["Rust"]},
        ]
        provider = BatchingFakeLLMProvider(
            category_by_term={"Python": "resume_technical_skill", "Rust": "soft_skill"}
        )
        results = _run(categorize_candidates_for_chunks(provider, chunk_terms, batch_size=2))

        self.assertEqual(len(provider.calls), 1, "both chunks fit in one explicit batch")
        self.assertEqual(results[0]["Python"]["category"], "resume_technical_skill")
        self.assertEqual(results[1]["Rust"]["category"], "soft_skill")

    def test_chunks_with_no_terms_never_occupy_a_batch_slot(self) -> None:
        chunk_terms = [
            {"chunk": "Boilerplate.", "terms": []},
            {"chunk": "We use Python daily.", "terms": ["Python"]},
        ]
        provider = BatchingFakeLLMProvider(category_by_term={"Python": "resume_technical_skill"})
        results = _run(categorize_candidates_for_chunks(provider, chunk_terms, batch_size=2))

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(results[0], {})
        self.assertEqual(results[1]["Python"]["category"], "resume_technical_skill")


if __name__ == "__main__":
    unittest.main()
