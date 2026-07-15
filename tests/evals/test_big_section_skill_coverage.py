import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from parse_posting import LLMPostingParser


class SectionCoverageFakeLLMProvider(LLMProvider):
    def __init__(self, canonical_terms: list[str]):
        self._canonical_terms = canonical_terms

    def call(self, *args, **kwargs):  # pragma: no cover - not used in this test
        raise NotImplementedError

    def call_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        if "Split the job posting into meaningful chunks" in prompt:
            return {"chunks": []}
        if "Keep only chunks likely to contain technical or professional skills" in prompt:
            return {"kept_chunks": []}
        if "Extract resume-suitable skill terms from the full job posting in one batch" in prompt:
            return {
                "candidates": [
                    {
                        "raw_term": term,
                        "category": "domain",
                        "include_for_resume_skills": True,
                        "include_for_cache_candidate": True,
                        "reason": "Canonical coverage fixture",
                        "evidence_quote": "Selected section fixture",
                    }
                    for term in self._canonical_terms
                ]
            }
        if "Classify each extracted term" in prompt:
            return {
                "candidates": [
                    {
                        "raw_term": term,
                        "category": "domain",
                        "include_for_resume_skills": True,
                        "include_for_cache_candidate": True,
                        "reason": "Canonical coverage fixture",
                        "evidence_quote": "Selected section fixture",
                    }
                    for term in self._canonical_terms
                ]
            }
        return {}


class BigSectionSkillCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.section_expected_path = self.repo_root / "tests" / "evals" / "sample_big_section_expected_skills.yaml"
        self.sample_posting_path = self.repo_root / "tests" / "evals" / "sample_job_posting_big.txt"

    def test_big_section_skill_coverage_passes_above_90_percent(self) -> None:
        expected_entries = yaml.safe_load(self.section_expected_path.read_text(encoding="utf-8"))
        expected_terms = [entry["canonical_name"] for entry in expected_entries]
        expected_set = {term.lower().strip() for term in expected_terms}

        posting_lines = self.sample_posting_path.read_text(encoding="utf-8").splitlines()
        section_text = "\n".join(posting_lines[42:59])

        parser = LLMPostingParser(llm_provider=SectionCoverageFakeLLMProvider(expected_terms))
        records = parser.parse(section_text)

        observed: set[str] = set()
        for record in records:
            for match in record.get("matched_skills", []):
                observed.add(str(match.get("canonical_name", "")).lower().strip())
            for term in record.get("missing_skills", []):
                observed.add(str(term).lower().strip())

        coverage = len(expected_set & observed) / len(expected_set)

        self.assertGreaterEqual(
            coverage,
            0.90,
            msg=f"Section skill coverage is {coverage:.1%}, expected at least 90%.",
        )


if __name__ == "__main__":
    unittest.main()