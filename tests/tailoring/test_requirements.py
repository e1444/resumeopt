"""Deterministic tests for `tailoring.requirements` (Phase 1).

No real LLM/network calls - `extract_job_requirements`'s `parse_fn` is
injected with a fake that returns a canned parser-record shape, so these
tests exercise only the RESHAPING logic (parser record -> `JobRequirements`
-> `requirements.json`), not the parser pipeline itself (already covered by
`tests/parser/`).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tailoring.requirements import (
    _requirement_sentences_from_chunk_verdicts,
    extract_job_requirements,
    job_requirements_from_dict,
    job_requirements_to_dict,
    load_requirements_json,
    write_requirements_json,
)


class FakeSummaryProvider:
    model = "fake-summary-model"


class FakeReasoningProvider:
    model = "fake-reasoning-model"


def _fake_parse_fn(posting_text, **kwargs):
    return [
        {
            "posting_line": posting_text,
            "matched_skills": [
                {
                    "raw_term": "python",
                    "canonical_name": "python",
                    "match_type": "exact",
                    "confidence": 0.98,
                    "relevance_score": 5,
                    "evidence": "Experience with Python required.",
                }
            ],
            "missing_skills": ["hydra"],
            "missing_skills_evidence": {"hydra": "Familiarity with Hydra configs."},
            "missing_skills_discarded": [],
            "extraction_debug_samples": [
                {
                    "chunks": [],
                    "ungrounded_discarded": [],
                    "chunk_verdicts": {
                        "python": {
                            "chunk": "Experience with Python required.",
                            "category": "core",
                            "extraction_reason": "explicitly required",
                            "category_reason": "core requirement",
                            "atomic_keyword": "python",
                            "atomicity_reason": "single skill",
                            "redundant_with": None,
                            "redundancy_reason": None,
                            "included": True,
                        },
                        "hydra": {
                            "chunk": "Familiarity with Hydra configs.",
                            "category": "nice_to_have",
                            "extraction_reason": "explicitly mentioned",
                            "category_reason": "nice to have",
                            "atomic_keyword": "hydra",
                            "atomicity_reason": "single skill",
                            "redundant_with": None,
                            "redundancy_reason": None,
                            "included": True,
                        },
                        "wandb": {
                            "chunk": "Familiarity with Hydra configs.",
                            "category": "nice_to_have",
                            "extraction_reason": "explicitly mentioned",
                            "category_reason": "nice to have",
                            "atomic_keyword": "wandb",
                            "atomicity_reason": "single skill",
                            "redundant_with": None,
                            "redundancy_reason": None,
                            "included": True,
                        },
                        "some_discarded_term": {
                            "chunk": "Familiarity with Hydra configs.",
                            "category": "nice_to_have",
                            "extraction_reason": "explicitly mentioned",
                            "category_reason": "nice to have",
                            "atomic_keyword": "some_discarded_term",
                            "atomicity_reason": "redundant",
                            "redundant_with": "hydra",
                            "redundancy_reason": "same skill, different phrasing",
                            "included": False,
                        },
                    },
                    "posting_summary": {
                        "role_title": "ML Research Engineer",
                        "seniority": "mid",
                        "industry_domain": "machine learning research",
                        "core_requirements": ["generative modeling", "Python"],
                        "nice_to_have": ["Hydra", "Weights & Biases"],
                        "summary_paragraph": "Builds generative models for structured prediction.",
                    },
                }
            ],
        }
    ]


def _empty_parse_fn(posting_text, **kwargs):
    return []


class ExtractJobRequirementsTest(unittest.TestCase):
    def test_reshapes_parser_record_into_job_requirements(self) -> None:
        requirements = extract_job_requirements(
            "posting text",
            summary_llm_provider=FakeSummaryProvider(),
            reasoning_llm_provider=FakeReasoningProvider(),
            parse_fn=_fake_parse_fn,
        )

        self.assertEqual(requirements.role_title, "ML Research Engineer")
        self.assertEqual(requirements.seniority, "mid")
        self.assertEqual(requirements.industry_domain, "machine learning research")
        self.assertEqual(requirements.core_requirements, ("generative modeling", "Python"))
        self.assertEqual(requirements.nice_to_have, ("Hydra", "Weights & Biases"))
        self.assertEqual(len(requirements.matched_skills), 1)
        self.assertEqual(requirements.matched_skills[0]["canonical_name"], "python")
        self.assertEqual(requirements.missing_skills, ("hydra",))
        self.assertEqual(requirements.parser_provenance["summary_model"], "fake-summary-model")
        self.assertEqual(requirements.parser_provenance["reasoning_model"], "fake-reasoning-model")

        # Phase 3.9: requirement_sentences grouped from chunk_verdicts,
        # excluding the one included=False term (some_discarded_term).
        self.assertEqual(len(requirements.requirement_sentences), 2)
        by_sentence = {match.sentence: match.skill_terms for match in requirements.requirement_sentences}
        self.assertEqual(by_sentence["Experience with Python required."], ("python",))
        self.assertEqual(by_sentence["Familiarity with Hydra configs."], ("hydra", "wandb"))

    def test_raises_on_empty_parser_output(self) -> None:
        with self.assertRaises(ValueError):
            extract_job_requirements(
                "",
                summary_llm_provider=FakeSummaryProvider(),
                reasoning_llm_provider=FakeReasoningProvider(),
                parse_fn=_empty_parse_fn,
            )


class RequirementSentencesFromChunkVerdictsTest(unittest.TestCase):
    def test_groups_by_sentence_and_excludes_not_included_terms(self) -> None:
        chunk_verdicts = {
            "react": {"chunk": "Experience with React required.", "included": True},
            "redux": {"chunk": "Experience with React required.", "included": True},
            "jquery": {"chunk": "Experience with React required.", "included": False},
            "python": {"chunk": "Python backend experience a plus.", "included": True},
        }

        matches = _requirement_sentences_from_chunk_verdicts(chunk_verdicts)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].sentence, "Experience with React required.")
        self.assertEqual(matches[0].skill_terms, ("react", "redux"))
        self.assertEqual(matches[1].sentence, "Python backend experience a plus.")
        self.assertEqual(matches[1].skill_terms, ("python",))

    def test_empty_chunk_verdicts_yields_no_sentences(self) -> None:
        self.assertEqual(_requirement_sentences_from_chunk_verdicts({}), [])


class RequirementsJsonRoundTripTest(unittest.TestCase):
    def test_to_dict_and_from_dict_round_trip(self) -> None:
        requirements = extract_job_requirements(
            "posting text",
            summary_llm_provider=FakeSummaryProvider(),
            reasoning_llm_provider=FakeReasoningProvider(),
            parse_fn=_fake_parse_fn,
        )

        round_tripped = job_requirements_from_dict(job_requirements_to_dict(requirements))

        self.assertEqual(round_tripped, requirements)

    def test_write_and_load_requirements_json(self) -> None:
        requirements = extract_job_requirements(
            "posting text",
            summary_llm_provider=FakeSummaryProvider(),
            reasoning_llm_provider=FakeReasoningProvider(),
            parse_fn=_fake_parse_fn,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "requirements.json"
            write_requirements_json(requirements, path)

            self.assertTrue(path.exists())
            with path.open() as handle:
                raw = json.load(handle)
            self.assertEqual(raw["role_title"], "ML Research Engineer")

            loaded = load_requirements_json(path)
            self.assertEqual(loaded, requirements)


if __name__ == "__main__":
    unittest.main()
