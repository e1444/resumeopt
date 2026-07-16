"""Per-candidate local context windows, replacing line/sentence-based chunking.

The actual purpose of "chunking" was never to produce grammatically perfect
units - it was to give an LLM a small, locally-relevant amount of text to
reason about per candidate. Line-based splitting is brittle (pasted text
easily malforms it); sentence-based splitting is brittle too (bullet points
aren't always punctuated), and there's no simple way to combine the two into
a robust, general solution.

Instead: normalize the whole posting into one continuous string
(normalize.py), decompose candidates from it once, then for each candidate
locate its `evidence_quote` in the normalized text and slice out a small
window of surrounding words. This sidesteps the line-vs-sentence dilemma
entirely, since neither boundary needs to be "correct" anymore - only the
window needs to be roughly the right size.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional, Tuple

DEFAULT_WINDOW_WORDS = 10


def locate_quote(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """Find the (start, end) character span of `quote` within `text`.

    Tries, in order:
    1. Exact substring match.
    2. Case-insensitive substring match.
    3. Fuzzy best-effort match (difflib longest common substring) for minor
       wording drift - decomposition is instructed to quote verbatim, but an
       LLM occasionally paraphrases slightly anyway.

    Returns None if no reasonably confident match is found, rather than
    anchoring on noise.
    """

    quote = quote.strip()
    if not quote or not text:
        return None

    index = text.find(quote)
    if index != -1:
        return index, index + len(quote)

    lower_text = text.lower()
    lower_quote = quote.lower()
    index = lower_text.find(lower_quote)
    if index != -1:
        return index, index + len(quote)

    matcher = SequenceMatcher(None, lower_text, lower_quote, autojunk=False)
    match = matcher.find_longest_match(0, len(lower_text), 0, len(lower_quote))
    # Require the fuzzy match to cover a meaningful majority of the quote,
    # otherwise treat it as unlocatable rather than anchoring on a coincidence.
    if match.size >= max(4, int(len(lower_quote) * 0.6)):
        return match.a, match.a + match.size
    return None


def build_context_window(
    text: str,
    quote: str,
    window_words: int = DEFAULT_WINDOW_WORDS,
    fallback: Optional[str] = None,
) -> str:
    """Return a window of `window_words` words on each side of `quote`'s location.

    Falls back to `fallback` if given, else the bare `quote` itself, else the
    whole `text`, if `quote` cannot be located - so a candidate never ends up
    with empty context, only a less-focused one.
    """

    span = locate_quote(text, quote)
    if span is None:
        if fallback is not None:
            return fallback
        return quote.strip() or text

    start, end = span
    before_words = text[:start].split()
    after_words = text[end:].split()
    window_before = before_words[-window_words:] if window_words > 0 else []
    window_after = after_words[:window_words] if window_words > 0 else []
    matched_text = text[start:end]

    return " ".join([*window_before, matched_text, *window_after]).strip()
