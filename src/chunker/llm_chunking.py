"""LLM-based sentence/bullet chunking - the DEFAULT chunking mechanism for
the parser pipeline, replacing `sentence_chunking.py`'s deterministic regex
splitter as the primary path (that module remains the fallback).

Validated via `tests/chunker/llm_chunking_benchmark.py` against 2 real
postings with different formatting styles:
  - `sample_job_posting_big1.txt`: regex chunker produced 34 chunks (merging
    section headers into the following bullet, and merging 2 distinct
    bullets together whenever one lacked terminal punctuation); LLM chunker
    produced 47 chunks, correctly separating every header and bullet, 0
    ungrounded, 100% text coverage.
  - `sample_job_posting_big2.txt` (heavier bullet/label formatting, no
    terminal punctuation between labeled sub-items): regex chunker collapsed
    3 entire multi-item sections (5 responsibilities, 6 qualifications, 4
    benefits) into 3 giant blobs; LLM chunker correctly isolated every
    individual labeled sub-item as its own chunk, 0 ungrounded, 100% text
    coverage.

The whole parser pipeline (Stage 1 extraction, Stage 2 categorization, and
ESPECIALLY Stage 3's WITHIN-CHUNK redundancy check) depends on chunk
boundaries actually isolating one coherent requirement/bullet per chunk - a
chunker that merges multiple unrelated bullets together dilutes Stage 1's
local context and makes Stage 3's same-chunk redundancy signal far less
reliable. Given this, the accuracy gap measured above is a first-order
concern for the whole pipeline, not a minor detail.

Grounding is enforced the same way extracted TERMS are grounded
(`locate_quote`): any LLM-returned chunk that isn't actually a locatable
substring of the posting is dropped (fail-safe toward losing that one
chunk's content rather than risking a fabricated/altered chunk propagating
through the rest of the pipeline). If the LLM call fails entirely (or
returns nothing grounded), falls back to the deterministic regex chunker
(`sentence_chunking.split_into_sentence_chunks`) rather than losing the
whole posting - graceful degradation over a hard failure.
"""

from __future__ import annotations

from typing import List

from llm import LLMProvider, call_json_with_retry_async

from .sentence_chunking import split_into_sentence_chunks
from .window import locate_quote

_CHUNKING_JSON_SCHEMA = {
    "name": "posting_chunks",
    "schema": {
        "type": "object",
        "properties": {
            "chunks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["chunks"],
        "additionalProperties": False,
    },
}

_TASK_PROMPT = (
    "Task: split the ENTIRE job posting below into individual sentence/bullet-point-level chunks - one "
    "requirement, responsibility, benefit, or statement per chunk.\n"
    "Rules:\n"
    "1. Preserve the EXACT original wording VERBATIM for every chunk - do not paraphrase, summarize, "
    "reword, correct typos, or alter punctuation/spacing beyond what is necessary to separate chunks. "
    "Every chunk must be an exact substring of the posting below.\n"
    "2. Treat each bullet point or list item as its OWN separate chunk, even if it has no terminal "
    "punctuation (period/semicolon) separating it from the next item - use the semantic structure "
    "(section headers, bullet-like phrasing, topic shifts) to find the real boundary, not just "
    "punctuation.\n"
    "3. A section header (e.g. 'What You Bring To The Table', 'Assets (nice to have)') must be its OWN "
    "separate chunk - never merged with the bullet/sentence that follows it.\n"
    "4. Do not skip, omit, or drop any part of the posting - every word should appear in exactly one "
    "chunk.\n"
    "5. Cover the ENTIRE posting text below, in order, from start to end.\n\n"
    "Job posting:\n{posting_text}"
)


async def split_into_sentence_chunks_via_llm(llm_provider: LLMProvider, posting_text: str) -> List[str]:
    """LLM-based chunking with grounding-checked fallback to the deterministic
    regex chunker.

    `posting_text` should already be whitespace-normalized (`normalize_whitespace`)
    - same expectation as `sentence_chunking.split_into_sentence_chunks`.
    """

    if not posting_text.strip():
        return []

    prompt = _TASK_PROMPT.format(posting_text=posting_text)
    payload = await call_json_with_retry_async(
        llm_provider,
        "llm_chunking",
        prompt=prompt,
        system_prompt=(
            "You split a job posting into sentence/bullet-level chunks, preserving exact original "
            "wording. Return valid JSON only."
        ),
        temperature=0.1,
        max_tokens=4000,
        json_schema=_CHUNKING_JSON_SCHEMA,
    )

    if payload is None:
        return split_into_sentence_chunks(posting_text)

    raw_chunks = [str(chunk).strip() for chunk in payload.get("chunks", []) if str(chunk).strip()]
    grounded_chunks = [chunk for chunk in raw_chunks if locate_quote(posting_text, chunk) is not None]

    if not grounded_chunks:
        return split_into_sentence_chunks(posting_text)
    return grounded_chunks
