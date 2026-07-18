import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from main import (
    _build_skill_review_payload,
    _estimate_tokens_from_payload,
    _estimate_tokens_from_text,
    _merge_always_include_skills,
    _prioritize_always_include_skills,
)
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


class PrioritizeAlwaysIncludeSkillsTest(unittest.TestCase):
    def test_reprioritizes_an_included_always_include_skill_to_the_front(self) -> None:
        cache = [
            SkillRecord(name="Python", aliases=(), always_include=False),
            SkillRecord(name="OOP", aliases=(), always_include=True),
        ]

        reordered, prioritized = _prioritize_always_include_skills(["Git", "Python", "OOP"], cache)

        self.assertEqual(reordered, ["OOP", "Git", "Python"])
        self.assertEqual(prioritized, ["OOP"])

    def test_never_adds_an_always_include_skill_the_caller_did_not_include(self) -> None:
        # Regression test: an always-include skill the user explicitly
        # unchecked at the Phase 9 review checkpoint must NOT be force-added
        # back in - unlike _merge_always_include_skills (used only before
        # the user has reviewed anything), this function only reorders.
        cache = [SkillRecord(name="C++", aliases=(), always_include=True)]

        reordered, prioritized = _prioritize_always_include_skills(["Python"], cache)

        self.assertEqual(reordered, ["Python"])
        self.assertNotIn("C++", reordered)
        self.assertEqual(prioritized, [])


class BuildSkillReviewPayloadTest(unittest.TestCase):
    def _validation_report(self, selected_skills):
        return {"status": "pass", "issues": [], "selected_skills": selected_skills}

    def test_always_include_skills_sort_to_the_end(self) -> None:
        cache = [
            SkillRecord(name="Python", aliases=(), always_include=False),
            SkillRecord(name="OOP", aliases=(), always_include=True),
        ]
        validation_report = self._validation_report(
            [{"canonical_name": "Python", "match_type": "exact", "confidence": 0.98, "evidence": "We need Python."}]
        )

        payload = _build_skill_review_payload(
            validation_report=validation_report,
            missing_skills=["kubernetes"],
            missing_skill_evidence={"kubernetes": "We need Python and Kubernetes."},
            forced_skills=["OOP"],
            skill_cache=cache,
        )

        names_in_order = [entry["name"] for entry in payload["reviewable_skills"]]
        # Python (matched) and kubernetes (missing) both come before OOP
        # (always-include), even though OOP was passed in as a forced skill
        # before kubernetes was appended - always-include entries sort to
        # the very end regardless of source or original insertion order.
        self.assertEqual(names_in_order, ["Python", "kubernetes", "OOP"])

    def test_a_matched_skill_that_is_also_always_include_sorts_to_the_end_too(self) -> None:
        cache = [SkillRecord(name="Python", aliases=(), always_include=True)]
        validation_report = self._validation_report(
            [
                {"canonical_name": "Python", "match_type": "exact", "confidence": 0.98, "evidence": "..."},
                {"canonical_name": "SQL", "match_type": "exact", "confidence": 0.98, "evidence": "..."},
            ]
        )

        payload = _build_skill_review_payload(
            validation_report=validation_report,
            missing_skills=[],
            missing_skill_evidence={},
            forced_skills=[],
            skill_cache=cache,
        )

        names_in_order = [entry["name"] for entry in payload["reviewable_skills"]]
        self.assertEqual(names_in_order, ["SQL", "Python"])
        python_entry = next(entry for entry in payload["reviewable_skills"] if entry["name"] == "Python")
        self.assertEqual(python_entry["source"], "matched")
        self.assertTrue(python_entry["is_always_include"])

    def test_missing_skills_get_evidence_from_the_posting_line_they_were_found_in(self) -> None:
        validation_report = self._validation_report([])

        payload = _build_skill_review_payload(
            validation_report=validation_report,
            missing_skills=["Kubernetes"],
            missing_skill_evidence={"kubernetes": "We need Kubernetes experience."},
            forced_skills=[],
            skill_cache=[],
        )

        entry = payload["reviewable_skills"][0]
        self.assertEqual(entry["evidence"], "We need Kubernetes experience.")

    def test_no_locked_field_present(self) -> None:
        validation_report = self._validation_report(
            [{"canonical_name": "Python", "match_type": "exact", "confidence": 0.98, "evidence": "..."}]
        )

        payload = _build_skill_review_payload(
            validation_report=validation_report,
            missing_skills=[],
            missing_skill_evidence={},
            forced_skills=[],
            skill_cache=[],
        )

        self.assertNotIn("locked", payload["reviewable_skills"][0])


if __name__ == "__main__":
    unittest.main()