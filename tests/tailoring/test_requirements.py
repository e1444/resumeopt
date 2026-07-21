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
                    "chunk_verdicts": {},
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

    def test_raises_on_empty_parser_output(self) -> None:
        with self.assertRaises(ValueError):
            extract_job_requirements(
                "",
                summary_llm_provider=FakeSummaryProvider(),
                reasoning_llm_provider=FakeReasoningProvider(),
                parse_fn=_empty_parse_fn,
            )


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
