"""Deterministic regression test for a real bug found via a live posting
(2026-07-20): `parser.factory.parse_posting`'s matching loop only keeps the
HIGHEST-confidence candidate per canonical skill bucket when two different
raw terms both resolve to the same cache entry - the LOSING candidate used
to be silently dropped entirely (not in `matched_skills`, not `missing_
skills`, not even `missing_skills_discarded`), directly contradicting this
project's own documented guarantee (README: "Anything extracted from a
posting that isn't in the cache shows up in missing_skills for review,
rather than being silently invented or silently dropped").

Concretely: a posting mentioning both "Microsoft Azure" and "NLP"/"NLG" -
none of which have a dedicated cache entry - had those terms semantically
match weak, unrelated canonical skills (Azure -> GitHub at 0.46, NLP/NLG ->
Machine Learning at ~0.46-0.48) that were ALSO independently claimed by a
different, higher-confidence raw term (the literal "Machine learning"
mention at 0.98 exact, "Version control" at 0.46 semantic) - Azure/NLP/NLG
vanished with zero trace anywhere in the run's logs.

This test reproduces the same shape deterministically using only exact/
alias matching (no semantic matcher/embeddings needed): a cache with ONE
skill (`GitHub`) that has an alias (`version control`) - if a posting
mentions BOTH "GitHub" (exact match, confidence 0.98) and "version control"
(alias match, confidence 0.90), only the higher-confidence "GitHub" should
occupy the canonical bucket - "version control" must now fall back into
`missing_skills` instead of disappearing.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from parser import parse_posting


class FakeLLMProvider(LLMProvider):
    """Minimal stub covering every schema `parse_posting(use_llm=True, ...)`
    calls when `use_llm_chunking=False`/`enable_chunk_screening=False`
    (deterministic regex chunking, no screening call) - `posting_summary`
    (Stage 0), `chunk_skill_extraction` (Stage 1), `skill_category_flags`
    (Stage 2), `keyword_atomicity_flags` (Stage 3a). Stage 3b redundancy is
    never invoked here since every term is marked atomic (bypasses it
    entirely) - same convention as `tests/parser/test_pipeline.py`'s fake.
    """

    def __init__(
        self,
        extraction_by_chunk: Dict[str, List[str]],
        category_by_term: Dict[str, str],
        atomicity_by_term: Dict[str, bool],
    ) -> None:
        super().__init__()
        self.extraction_by_chunk = extraction_by_chunk
        self.category_by_term = category_by_term
        self.atomicity_by_term = atomicity_by_term

    def call(self, *args, **kwargs):  # pragma: no cover - not used in this test
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

        if schema_name == "posting_summary":
            return {
                "role_title": "",
                "seniority": "",
                "industry_domain": "",
                "core_requirements": [],
                "nice_to_have": [],
                "summary_paragraph": "",
            }
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
        raise AssertionError(f"unexpected schema in test: {schema_name}")


class LosingCanonicalBucketCandidateTest(unittest.TestCase):
    def test_lower_confidence_candidate_falls_back_to_missing_not_dropped(self) -> None:
        posting_text = "We use GitHub daily. We rely on version control for collaboration."
        chunk_a = "We use GitHub daily."
        chunk_b = "We rely on version control for collaboration."

        provider = FakeLLMProvider(
            extraction_by_chunk={chunk_a: ["GitHub"], chunk_b: ["version control"]},
            category_by_term={"GitHub": "resume_technical_skill", "version control": "resume_technical_skill"},
            atomicity_by_term={"GitHub": True, "version control": True},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "skills.yaml"
            cache_path.write_text(
                "- name: GitHub\n  aliases:\n  - version control\n",
                encoding="utf-8",
            )

            records = parse_posting(
                posting_text=posting_text,
                skills_cache_path=cache_path,
                use_llm=True,
                summary_llm_provider=provider,
                reasoning_llm_provider=provider,
                use_semantic_matching=False,
                embedding_cache_path=None,
                use_llm_chunking=False,
                enable_chunk_screening=False,
            )

        self.assertEqual(len(records), 1)
        record = records[0]

        # "GitHub" (exact, 0.98) should win the canonical "GitHub" bucket.
        matched_raw_terms = {m["raw_term"] for m in record["matched_skills"]}
        self.assertIn("GitHub", matched_raw_terms)
        self.assertEqual(len(record["matched_skills"]), 1)

        # "version control" (alias, 0.90) lost the same bucket - it must now
        # fall back into missing_skills, NOT vanish without a trace.
        self.assertIn("version control", record["missing_skills"])
        self.assertNotIn(
            "version control",
            [d.get("raw_term") for d in record["missing_skills_discarded"]],
        )

    def test_candidate_that_wins_then_gets_evicted_later_falls_back_to_missing(self) -> None:
        """Reproduces the DEEPER version of the same bug (found via a real
        posting where "Microsoft Azure" still vanished even after the first
        fix above): a raw_term can WIN its canonical bucket at the moment
        it's processed (the bucket is still empty), only to be silently
        EVICTED later in the same run when a different, higher-confidence
        raw_term (processed afterward, e.g. because it appears later in the
        posting) claims that same bucket. A single streaming pass can't
        catch this - the earlier term already "succeeded" at its own turn,
        so it was never routed to `missing` even though, by the end, it
        holds no bucket at all. `parse_posting` must only finalize a term as
        "matched" once every verdict has had a chance to compete.
        """

        posting_text = "We use ml stuff for our storytelling engine. Solid understanding of Machine Learning is required."
        chunk_a = "We use ml stuff for our storytelling engine."
        chunk_b = "Solid understanding of Machine Learning is required."

        provider = FakeLLMProvider(
            extraction_by_chunk={chunk_a: ["ml stuff"], chunk_b: ["Machine Learning"]},
            category_by_term={"ml stuff": "resume_technical_skill", "Machine Learning": "resume_technical_skill"},
            atomicity_by_term={"ml stuff": True, "Machine Learning": True},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "skills.yaml"
            cache_path.write_text(
                "- name: Machine Learning\n  aliases:\n  - ml stuff\n",
                encoding="utf-8",
            )

            records = parse_posting(
                posting_text=posting_text,
                skills_cache_path=cache_path,
                use_llm=True,
                summary_llm_provider=provider,
                reasoning_llm_provider=provider,
                use_semantic_matching=False,
                embedding_cache_path=None,
                use_llm_chunking=False,
                enable_chunk_screening=False,
            )

        self.assertEqual(len(records), 1)
        record = records[0]

        # "Machine Learning" (exact, 0.98) is the final winner of its own
        # canonical bucket, even though "ml stuff" (alias, 0.90) claimed it
        # first (chunk_a is processed before chunk_b).
        matched_raw_terms = {m["raw_term"] for m in record["matched_skills"]}
        self.assertIn("Machine Learning", matched_raw_terms)
        self.assertEqual(len(record["matched_skills"]), 1)

        # "ml stuff" won initially but was evicted - it must now fall back
        # into missing_skills instead of vanishing with zero trace.
        self.assertIn("ml stuff", record["missing_skills"])
        self.assertNotIn(
            "ml stuff",
            [d.get("raw_term") for d in record["missing_skills_discarded"]],
        )


if __name__ == "__main__":
    unittest.main()
