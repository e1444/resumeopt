import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from main import _estimate_tokens_from_payload, _estimate_tokens_from_text, _merge_always_include_skills
from matcher import SkillRecord


class RunMetricsLoggingHelpersTest(unittest.TestCase):
    def test_estimate_tokens_from_text_is_positive_for_nonempty_input(self) -> None:
        estimated = _estimate_tokens_from_text("python machine learning")
        self.assertGreater(estimated, 0)

    def test_estimate_tokens_from_text_scales_with_input_size(self) -> None:
        small = _estimate_tokens_from_text("small")
        large = _estimate_tokens_from_text("small " * 100)
        self.assertGreater(large, small)

    def test_estimate_tokens_from_payload_handles_nested_data(self) -> None:
        payload = {
            "status": "pass",
            "skills": ["python", "machine learning", "git"],
            "details": {"count": 3, "notes": ["ok"]},
        }
        estimated = _estimate_tokens_from_payload(payload)
        self.assertGreater(estimated, 0)


class MergeAlwaysIncludeSkillsTest(unittest.TestCase):
    def test_prepends_always_include_skills_not_already_selected(self) -> None:
        cache = [
            SkillRecord(name="Python", aliases=(), always_include=False),
            SkillRecord(name="OOP", aliases=(), always_include=True),
            SkillRecord(name="SQL", aliases=(), always_include=True),
        ]

        merged, forced = _merge_always_include_skills(["Git", "Python"], cache)

        self.assertEqual(merged, ["OOP", "SQL", "Git", "Python"])
        self.assertEqual(forced, ["OOP", "SQL"])

    def test_does_not_duplicate_an_always_include_skill_already_selected(self) -> None:
        cache = [SkillRecord(name="Python", aliases=(), always_include=True)]

        merged, forced = _merge_always_include_skills(["Python", "Git"], cache)

        self.assertEqual(merged, ["Python", "Git"])
        self.assertEqual(forced, [])

    def test_case_insensitive_dedupe(self) -> None:
        cache = [SkillRecord(name="Python", aliases=(), always_include=True)]

        merged, forced = _merge_always_include_skills(["python", "Git"], cache)

        self.assertEqual(merged, ["python", "Git"])
        self.assertEqual(forced, [])

    def test_no_always_include_skills_is_a_no_op(self) -> None:
        cache = [SkillRecord(name="Python", aliases=(), always_include=False)]

        merged, forced = _merge_always_include_skills(["Git"], cache)

        self.assertEqual(merged, ["Git"])
        self.assertEqual(forced, [])


if __name__ == "__main__":
    unittest.main()