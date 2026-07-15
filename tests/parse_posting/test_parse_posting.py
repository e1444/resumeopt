import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from parse_posting import DeterministicPostingParser, LLMPostingParser, parse_posting
from parse_posting import validate_selected_skills


class FakeLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "Split the job posting into meaningful chunks" in prompt:
            return {"chunks": ["We need Python and PyTorch experience."]}
        if "Keep only chunks likely to contain technical or professional skills" in prompt:
            return {"kept_chunks": ["We need Python and PyTorch experience."]}
        if "Given a job posting chunk and a canonical skills cache" in prompt:
            return {
                "matched_skills": [
                    {
                        "raw_term": "Python",
                        "canonical_name": "python",
                        "match_type": "exact",
                        "confidence": 0.99,
                        "relevance_score": 5,
                        "evidence": "We need Python and PyTorch experience.",
                    },
                    {
                        "raw_term": "PyTorch",
                        "canonical_name": "pytorch",
                        "match_type": "exact",
                        "confidence": 0.98,
                        "relevance_score": 5,
                        "evidence": "We need Python and PyTorch experience.",
                    },
                    {
                        "raw_term": "HallucinatedSkill",
                        "canonical_name": "not-in-cache",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": "We need Python and PyTorch experience.",
                    },
                ]
            }
        if "Determine whether the skill is actually supported by the posting text" in prompt:
            return {"is_grounded": False, "reason": "No clear support"}
        return {}


class ValidationGroundingLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "Determine whether the skill is actually supported by the posting text" in prompt:
            prompt_lower = prompt.lower()
            if "skill canonical name: jupyter" in prompt_lower and "ipynb" in prompt_lower:
                return {"is_grounded": True, "reason": "ipynb is a jupyter notebook format"}
            return {"is_grounded": False, "reason": "Not supported"}
        return {}


class DeterministicPostingParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.skills_cache_path = self.repo_root / "data" / "skills.yaml"
        self.sample_posting_path = self.repo_root / "tests" / "evals" / "sample_job_posting.txt"
        self.expected_path = self.repo_root / "tests" / "evals" / "sample_expected_skills.yaml"

    def test_parse_posting_defaults_to_deterministic_parser(self) -> None:
        posting_text = self.sample_posting_path.read_text(encoding="utf-8")

        result = parse_posting(posting_text=posting_text, skills_cache_path=self.skills_cache_path)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_deterministic_parser_output_matches_schema_shape(self) -> None:
        posting_text = self.sample_posting_path.read_text(encoding="utf-8")
        parser = DeterministicPostingParser(skills_cache_path=self.skills_cache_path)

        result = parser.parse(posting_text)

        self.assertGreater(len(result), 0)
        for chunk_record in result:
            self.assertIn("posting_line", chunk_record)
            self.assertIn("extracted_raw_terms", chunk_record)
            self.assertIn("matched_skills", chunk_record)
            self.assertIn("validation", chunk_record)

            self.assertIsInstance(chunk_record["posting_line"], str)
            self.assertIsInstance(chunk_record["extracted_raw_terms"], list)
            self.assertIsInstance(chunk_record["matched_skills"], list)
            self.assertIsInstance(chunk_record["validation"], dict)

            for match in chunk_record["matched_skills"]:
                self.assertIn("raw_term", match)
                self.assertIn("canonical_name", match)
                self.assertIn("match_type", match)
                self.assertIn("confidence", match)
                self.assertIn("relevance_score", match)
                self.assertIn("evidence", match)

    def test_parser_covers_expected_sample_skills(self) -> None:
        posting_text = self.sample_posting_path.read_text(encoding="utf-8")
        expected = yaml.safe_load(self.expected_path.read_text(encoding="utf-8"))

        parser = DeterministicPostingParser(skills_cache_path=self.skills_cache_path)
        result = parser.parse(posting_text)

        strongest: dict[str, dict] = {}
        for chunk in result:
            for match in chunk["matched_skills"]:
                existing = strongest.get(match["canonical_name"])
                if existing is None or self._strength(match) > self._strength(existing):
                    strongest[match["canonical_name"]] = match

        for expected_skill in expected:
            canonical_name = expected_skill["canonical_name"]
            self.assertIn(canonical_name, strongest)

            actual = strongest[canonical_name]
            self.assertGreaterEqual(actual["confidence"], expected_skill["minimum_confidence"])
            self.assertGreaterEqual(
                self._match_strength(actual["match_type"]),
                self._match_strength(expected_skill["match_type"]),
            )

    def test_duplicate_canonical_names_are_rejected(self) -> None:
        duplicate_payload = [
            {"name": "python", "aliases": ["py"]},
            {"name": "Python", "aliases": ["python3"]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(duplicate_payload, handle)
            temp_path = Path(handle.name)

        try:
            with self.assertRaisesRegex(ValueError, "Duplicate canonical skill name"):
                DeterministicPostingParser(skills_cache_path=temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_llm_parser_returns_only_cache_backed_matches(self) -> None:
        parser = LLMPostingParser(
            llm_provider=FakeLLMProvider(),
            skills_cache_path=self.skills_cache_path,
        )

        result = parser.parse("Any posting text")

        self.assertEqual(len(result), 1)
        matched = result[0]["matched_skills"]
        canonical_names = {item["canonical_name"] for item in matched}

        self.assertIn("python", canonical_names)
        self.assertIn("pytorch", canonical_names)
        self.assertNotIn("not-in-cache", canonical_names)

    def test_parse_posting_uses_llm_parser_when_requested(self) -> None:
        result = parse_posting(
            posting_text="Any posting text",
            skills_cache_path=self.skills_cache_path,
            llm_provider=FakeLLMProvider(),
            use_llm=True,
        )

        self.assertEqual(len(result), 1)
        canonical_names = {item["canonical_name"] for item in result[0]["matched_skills"]}
        self.assertEqual(canonical_names, {"python", "pytorch"})

    def test_validate_selected_skills_passes_for_sample(self) -> None:
        posting_text = self.sample_posting_path.read_text(encoding="utf-8")
        parser = DeterministicPostingParser(skills_cache_path=self.skills_cache_path)
        records = parser.parse(posting_text)

        report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=self.skills_cache_path,
            min_confidence=0.7,
            max_unique_skills=12,
        )

        self.assertEqual(report["status"], "pass")
        self.assertGreater(len(report["selected_skills"]), 0)

    def test_validate_selected_skills_rejects_unsupported_and_weak(self) -> None:
        posting_text = "We need Python engineers."
        records = [
            {
                "posting_line": posting_text,
                "extracted_raw_terms": ["Python", "FakeSkill"],
                "matched_skills": [
                    {
                        "raw_term": "Python",
                        "canonical_name": "python",
                        "match_type": "exact",
                        "confidence": 0.4,
                        "relevance_score": 3,
                        "evidence": posting_text,
                    },
                    {
                        "raw_term": "FakeSkill",
                        "canonical_name": "fake-skill",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                ],
                "validation": {"status": "pass", "notes": []},
            }
        ]

        report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=self.skills_cache_path,
            min_confidence=0.7,
            max_unique_skills=12,
        )

        self.assertEqual(report["status"], "fail")
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertIn("weak_match", issue_types)
        self.assertIn("unsupported_skill", issue_types)

    def test_validate_selected_skills_enforces_max_unique_skills(self) -> None:
        posting_text = "Python Git NumPy Pandas Matplotlib Seaborn Jupyter TensorFlow PyTorch."
        records = [
            {
                "posting_line": posting_text,
                "extracted_raw_terms": [
                    "python",
                    "git",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "seaborn",
                ],
                "matched_skills": [
                    {
                        "raw_term": "python",
                        "canonical_name": "python",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                    {
                        "raw_term": "git",
                        "canonical_name": "git",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                    {
                        "raw_term": "numpy",
                        "canonical_name": "numpy",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                ],
                "validation": {"status": "pass", "notes": []},
            },
            {
                "posting_line": posting_text,
                "extracted_raw_terms": ["pandas", "matplotlib", "seaborn"],
                "matched_skills": [
                    {
                        "raw_term": "pandas",
                        "canonical_name": "pandas",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                    {
                        "raw_term": "matplotlib",
                        "canonical_name": "matplotlib",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                    {
                        "raw_term": "seaborn",
                        "canonical_name": "seaborn",
                        "match_type": "exact",
                        "confidence": 0.95,
                        "relevance_score": 5,
                        "evidence": posting_text,
                    },
                ],
                "validation": {"status": "pass", "notes": []},
            },
        ]

        report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=self.skills_cache_path,
            min_confidence=0.7,
            max_unique_skills=3,
        )

        self.assertEqual(report["status"], "fail")
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertIn("too_many_skills", issue_types)

    def test_validate_selected_skills_allows_llm_edgecase_grounding(self) -> None:
        posting_text = "Candidate has strong experience working with ipynb files."
        records = [
            {
                "posting_line": posting_text,
                "extracted_raw_terms": ["notebook tooling"],
                "matched_skills": [
                    {
                        "raw_term": "notebook tooling",
                        "canonical_name": "jupyter",
                        "match_type": "related",
                        "confidence": 0.8,
                        "relevance_score": 3,
                        "evidence": "portable notebook workflow",
                    }
                ],
                "validation": {"status": "pass", "notes": []},
            }
        ]

        report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=self.skills_cache_path,
            min_confidence=0.7,
            max_unique_skills=12,
            llm_provider=ValidationGroundingLLMProvider(),
        )

        self.assertEqual(report["status"], "pass")

    def test_validate_selected_skills_fails_when_llm_does_not_confirm_grounding(self) -> None:
        posting_text = "Candidate has strong experience working with ipynb files."
        records = [
            {
                "posting_line": posting_text,
                "extracted_raw_terms": ["notebook tooling"],
                "matched_skills": [
                    {
                        "raw_term": "notebook tooling",
                        "canonical_name": "jupyter",
                        "match_type": "related",
                        "confidence": 0.8,
                        "relevance_score": 3,
                        "evidence": "portable notebook workflow",
                    }
                ],
                "validation": {"status": "pass", "notes": []},
            }
        ]

        report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=self.skills_cache_path,
            min_confidence=0.7,
            max_unique_skills=12,
            llm_provider=FakeLLMProvider(),
        )

        self.assertEqual(report["status"], "fail")
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertIn("missing_grounding", issue_types)

    def _strength(self, match: dict) -> tuple[int, float]:
        return (self._match_strength(match["match_type"]), float(match["confidence"]))

    def _match_strength(self, match_type: str) -> int:
        return {"related": 1, "alias": 2, "exact": 3}[match_type]


if __name__ == "__main__":
    unittest.main()
