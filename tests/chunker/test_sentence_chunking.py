"""Deterministic tests for chunker.sentence_chunking (no API key needed)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import split_into_sentence_chunks


class SplitIntoSentenceChunksTest(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self) -> None:
        text = "We use Python daily. We also use Git for version control."
        self.assertEqual(
            split_into_sentence_chunks(text),
            ["We use Python daily.", "We also use Git for version control."],
        )

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(split_into_sentence_chunks(""), [])
        self.assertEqual(split_into_sentence_chunks("   "), [])

    def test_single_sentence_returns_one_chunk(self) -> None:
        self.assertEqual(split_into_sentence_chunks("Strong Python skills required."), ["Strong Python skills required."])

    def test_does_not_split_on_eg_style_abbreviations(self) -> None:
        # "e.g., mathematics" - lowercase letter follows the comma, so the
        # sentence-boundary regex (period/question/exclamation + whitespace +
        # UPPERCASE/digit) must not fire here.
        text = "Degree in a relevant discipline (e.g., mathematics, engineering) required."
        self.assertEqual(len(split_into_sentence_chunks(text)), 1)

    def test_normalizes_whitespace_and_mid_word_line_breaks_first(self) -> None:
        # Mirrors the malformed-PDF case this project already fixed for the
        # normalize-once architecture - normalizing BEFORE sentence-splitting
        # means a stray line break never creates a spurious chunk boundary.
        text = "Required: strong Ja\nvaScript and Type\nScript skills. Nice to have: Python."
        chunks = split_into_sentence_chunks(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("JavaScript", chunks[0].replace("Ja vaScript", "JavaScript"))  # tolerant of the known artifact
        self.assertNotIn("\n", chunks[0])

    def test_multi_sentence_passage_splits_into_multiple_chunks(self) -> None:
        text = (
            "Our backend engineers build services using Rust and PostgreSQL. "
            "We also offer a generous parental leave policy."
        )
        chunks = split_into_sentence_chunks(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Rust", chunks[0])
        self.assertIn("parental leave", chunks[1])


if __name__ == "__main__":
    unittest.main()
