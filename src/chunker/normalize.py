"""Whitespace normalization for raw posting text.

Real-world postings (pasted from PDFs, websites, Word docs) have unreliable
newline/whitespace placement that doesn't correlate with sentence or word
boundaries - line-wrap artifacts can even split a single word across two
lines with no hyphen. Rather than trying to detect and reverse-engineer
every possible artifact pattern (fragile, prone to overfitting to whatever
sample was on hand), this collapses ALL whitespace runs (including
newlines) into single spaces, so downstream code never has to reason about
line/paragraph structure at all - it only ever sees one continuous string.

This doesn't perfectly repair a mid-word break (e.g. "bootcam" + newline +
"p." becomes "bootcam p.", not "bootcamp.") - but that's an acceptable,
minor, typo-like imperfection sitting inside a normal-sized local context
window, which an LLM tolerates far better than the alternative failure mode
(the word's two halves ending up in two separate, mutually-invisible chunks
with no shared context at all).
"""

from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines) into a single space.

    Also strips leading/trailing whitespace. Idempotent: normalizing an
    already-normalized string returns it unchanged.
    """

    return " ".join(text.split())
