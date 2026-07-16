"""Deterministic tests for chunker.normalize (no API key needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import normalize_whitespace


class NormalizeWhitespaceTest(unittest.TestCase):
    def test_collapses_newlines_to_spaces(self) -> None:
        text = "Position Overview:\n\nWe are seeking an engineer."
        self.assertEqual(normalize_whitespace(text), "Position Overview: We are seeking an engineer.")

    def test_collapses_multiple_whitespace_runs(self) -> None:
        text = "Key Responsibilities\n\n\n\n:Code Implementation:   Write clean code."
        self.assertEqual(
            normalize_whitespace(text),
            "Key Responsibilities :Code Implementation: Write clean code.",
        )

    def test_collapses_mid_word_line_break_to_a_single_space(self) -> None:
        # Doesn't perfectly reconstruct "bootcamp." - that's an accepted,
        # minor, typo-like imperfection (see module docstring rationale).
        text = "an intensive software engineering bootcam\np.Foundational Coding:"
        self.assertEqual(
            normalize_whitespace(text),
            "an intensive software engineering bootcam p.Foundational Coding:",
        )

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        self.assertEqual(normalize_whitespace("  \n hello world \n\n  "), "hello world")

    def test_handles_empty_string(self) -> None:
        self.assertEqual(normalize_whitespace(""), "")
        self.assertEqual(normalize_whitespace("   \n\n  "), "")

    def test_is_idempotent(self) -> None:
        text = "Line one.\nLine two.\n\nLine three."
        once = normalize_whitespace(text)
        twice = normalize_whitespace(once)
        self.assertEqual(once, twice)

    def test_preserves_single_spaces_between_words(self) -> None:
        text = "already clean single spaced text"
        self.assertEqual(normalize_whitespace(text), text)

    def test_handles_tabs(self) -> None:
        text = "Python\tSkills\tRequired"
        self.assertEqual(normalize_whitespace(text), "Python Skills Required")


if __name__ == "__main__":
    unittest.main()
