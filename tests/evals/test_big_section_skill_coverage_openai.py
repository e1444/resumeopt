import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider
from parser import OrchestraSingleShotParser


class BigSectionSkillCoverageOpenAITest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cases_path = self.repo_root / "tests" / "evals" / "sample_big_section_sentence_cases.yaml"

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY required for the gpt-4o integration test")
    def test_sentence_chunks_return_expected_terms(self) -> None:
        cases = yaml.safe_load(self.cases_path.read_text(encoding="utf-8"))

        llm = get_llm_provider("openai", model="gpt-4o")
        parser = OrchestraSingleShotParser(llm_provider=llm)

        for case in cases:
            chunk = case["chunk"].strip()
            expected_terms = {term.lower().strip() for term in case.get("expected_terms", [])}

            with self.subTest(chunk=chunk, expected_terms=sorted(expected_terms)):
                records = parser.parse(chunk)

                observed_terms = {
                    str(match.get("canonical_name", "")).lower().strip()
                    for record in records
                    for match in record.get("matched_skills", [])
                } | {
                    str(term).lower().strip()
                    for record in records
                    for term in record.get("missing_skills", [])
                }

                self.assertEqual(
                    observed_terms,
                    expected_terms,
                    msg=f"Chunk '{chunk}' should return exactly {sorted(expected_terms)}. Got {sorted(observed_terms)}.",
                )


if __name__ == "__main__":
    unittest.main()
