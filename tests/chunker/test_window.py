"""Deterministic tests for chunker.window (no API key needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import build_context_window, locate_quote


class LocateQuoteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (
            "Basic familiarity with version control systems, specifically Git and GitHub. "
            "Strong logical thinking and problem-solving skills are also expected."
        )

    def test_exact_match(self) -> None:
        span = locate_quote(self.text, "specifically Git and GitHub")
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(self.text[start:end], "specifically Git and GitHub")

    def test_case_insensitive_match(self) -> None:
        span = locate_quote(self.text, "SPECIFICALLY GIT AND GITHUB")
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(self.text[start:end].lower(), "specifically git and github")

    def test_fuzzy_match_for_minor_wording_drift(self) -> None:
        # Real quote has "GitHub."; this is a slightly paraphrased near-miss
        # that should still resolve via the fuzzy fallback tier.
        span = locate_quote(self.text, "specifically Git and GitHub,")
        self.assertIsNotNone(span)

    def test_returns_none_for_unrelated_quote(self) -> None:
        span = locate_quote(self.text, "completely unrelated kubernetes deployment pipeline")
        self.assertIsNone(span)

    def test_returns_none_for_empty_quote(self) -> None:
        self.assertIsNone(locate_quote(self.text, ""))
        self.assertIsNone(locate_quote(self.text, "   "))

    def test_returns_none_for_empty_text(self) -> None:
        self.assertIsNone(locate_quote("", "anything"))


class BuildContextWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (
            "Basic familiarity with version control systems, specifically Git and GitHub. "
            "Strong logical thinking and problem-solving skills are also expected."
        )

    def test_includes_surrounding_words(self) -> None:
        window = build_context_window(self.text, "Git and GitHub", window_words=3)
        self.assertIn("Git and GitHub", window)
        self.assertIn("specifically", window)  # 3 words before
        self.assertIn("Strong", window)  # words after

    def test_respects_window_word_count(self) -> None:
        window = build_context_window(self.text, "Git and GitHub", window_words=1)
        self.assertIn("Git and GitHub", window)
        self.assertNotIn("familiarity", window)  # too far before with window=1

    def test_handles_quote_near_start_of_text(self) -> None:
        window = build_context_window(self.text, "Basic familiarity", window_words=10)
        self.assertTrue(window.startswith("Basic familiarity"))

    def test_handles_quote_near_end_of_text(self) -> None:
        window = build_context_window(self.text, "also expected", window_words=10)
        # The source sentence ends "...also expected." - the trailing period
        # is its own whitespace-split token, so it rides along after the quote.
        self.assertTrue(window.endswith("also expected .") or window.endswith("also expected"))

    def test_falls_back_to_quote_when_not_found(self) -> None:
        window = build_context_window(self.text, "totally unrelated phrase never present", window_words=5)
        self.assertEqual(window, "totally unrelated phrase never present")

    def test_falls_back_to_explicit_fallback_when_given(self) -> None:
        window = build_context_window(
            self.text,
            "totally unrelated phrase never present",
            window_words=5,
            fallback="use this instead",
        )
        self.assertEqual(window, "use this instead")

    def test_falls_back_to_whole_text_when_quote_and_fallback_both_empty(self) -> None:
        window = build_context_window(self.text, "", window_words=5)
        self.assertEqual(window, self.text)


if __name__ == "__main__":
    unittest.main()
