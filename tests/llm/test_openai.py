"""Live integration tests for the OpenAI provider.

These tests make real API calls and are skipped unless OPENAI_API_KEY is set,
consistent with tests/evals/test_big_section_skill_coverage_openai.py.
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY required for the OpenAI provider integration test")
class OpenAIProviderIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = get_llm_provider("openai", model="gpt-4o")

    def test_openai_text(self) -> None:
        response = self.llm.call(
            prompt="What is Python?",
            system_prompt="You are a helpful assistant. Keep your response to 1-2 sentences.",
        )

        self.assertGreater(len(response), 0, "Response should not be empty")

    def test_openai_json(self) -> None:
        prompt = (
            "Extract programming languages from this text:\n"
            '"I\'m proficient in Python, JavaScript, and C++."\n\n'
            'Return as JSON with format: {"languages": [...]}'
        )

        result = self.llm.call_json(
            prompt=prompt,
            system_prompt="You are a helpful assistant. Return valid JSON only.",
        )

        self.assertIsInstance(result, dict)
        self.assertIn("languages", result)

    def test_openai_skill_extraction(self) -> None:
        skills_cache = [
            {"name": "python", "aliases": ["py"], "related": ["scripting"]},
            {"name": "pytorch", "aliases": ["torch"], "related": ["deep learning"]},
        ]

        line = "Strong Python skills required; experience with PyTorch or similar ML frameworks."

        prompt = f"""Given the following line, extract skills and match them to the cache.

Line: "{line}"

Skills Cache:
{skills_cache}

Return JSON with format:
{{
    "extracted_raw_terms": ["skill1", "skill2"],
    "matched_skills": [
        {{
            "raw_term": "...",
            "canonical_name": "...",
            "match_type": "exact|alias|related",
            "confidence": 0.0-1.0,
            "evidence": "..."
        }}
    ]
}}"""

        result = self.llm.call_json(
            prompt=prompt,
            system_prompt="Extract skills and return valid JSON only.",
        )

        self.assertIn("matched_skills", result)


if __name__ == "__main__":
    unittest.main()
