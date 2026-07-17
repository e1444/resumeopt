"""Experimental comparison (standalone, run via `-m`, not a gated unittest,
same convention as `tests/relevance/`'s benchmark scripts): does an
LLM-based sentence/bullet chunker produce more accurate chunks than
`src/chunker/sentence_chunking.py`'s deterministic regex-based sentence
splitter?

Added 2026-07-17 per explicit user request/hypothesis: "this methodology
relies greatly on chunking being accurate, per sentence. it might be
simplest to use an llm to chunk a posting by sentence."

Motivating defects found empirically in the CURRENT deterministic chunker
when run against a real posting (`sample_job_posting_big1.txt`): it merges a
section header with the very next bullet whenever the header lacks terminal
punctuation ("What You Bring To The Table Degree in a relevant discipline
..."), merges TWO DIFFERENT bullets together whenever one lacks a trailing
period ("AI Governance experience / AI tool building for underwriting Call
Center Optimization (e.g., ..."), and merges an entire punctuation-free
benefits list into one giant chunk ("Flexible work arrangements and a hybrid
work model Possibility to purchase up to 5 extra days off per year Mul...").
All 3 are real, reproducible failures of a purely punctuation-driven
sentence-boundary regex on genuinely messy real-world posting formatting.

The LLM chunker asks a model to split the SAME normalized posting text into
chunks directly, instructed to preserve the EXACT original wording
verbatim (no paraphrasing) so grounding can be verified afterward via
`chunker.locate_quote` - the same grounding check `src/chunker/ or src/parser/
factory.py` already applies to extracted TERMS, reused here to validate
chunks themselves.

Run: `python -m tests.chunker.llm_chunking_benchmark [model] [posting_path]`
from repo root (needs OPENAI_API_KEY). `model` defaults to `gpt-5-mini`
(same reasoning model already used elsewhere in this package); `posting_path`
defaults to `tests/evals/sample_job_posting_big1.txt`. Writes
`build/benchmarks/llm_chunking_benchmark_<model>.json`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chunker import split_into_sentence_chunks  # noqa: E402
from chunker import locate_quote, normalize_whitespace  # noqa: E402
from llm import LLMProvider, get_llm_provider  # noqa: E402

_DEFAULT_MODEL = "gpt-5-mini"
_DEFAULT_POSTING_PATH = "tests/evals/sample_job_posting_big1.txt"

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


def _llm_chunk_posting(llm_provider: LLMProvider, posting_text: str) -> List[str]:
    prompt = _TASK_PROMPT.format(posting_text=posting_text)
    payload = llm_provider.call_json(
        prompt=prompt,
        system_prompt=(
            "You split a job posting into sentence/bullet-level chunks, preserving exact original "
            "wording. Return valid JSON only."
        ),
        temperature=0.1,
        max_tokens=4000,
        json_schema=_CHUNKING_JSON_SCHEMA,
    )
    chunks = (payload or {}).get("chunks", [])
    return [str(chunk).strip() for chunk in chunks if str(chunk).strip()]


def _check_grounding(posting_text: str, chunks: List[str]) -> Dict[str, Any]:
    ungrounded = [chunk for chunk in chunks if locate_quote(posting_text, chunk) is None]
    return {
        "chunk_count": len(chunks),
        "ungrounded_count": len(ungrounded),
        "ungrounded_chunks": ungrounded,
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for the live benchmark")

    model = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_MODEL
    posting_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(_DEFAULT_POSTING_PATH)

    repo_root = Path(__file__).resolve().parents[2]
    artifact_path = repo_root / "build" / "benchmarks" / f"llm_chunking_benchmark_{model}.json"

    posting_text = normalize_whitespace((repo_root / posting_path).read_text(encoding="utf-8"))

    print("Running deterministic (regex) chunker...")
    regex_chunks = split_into_sentence_chunks(posting_text)
    regex_grounding = _check_grounding(posting_text, regex_chunks)

    print(f"Running LLM chunker via {model}...")
    llm_provider: LLMProvider = get_llm_provider("openai", model=model)
    llm_chunks = _llm_chunk_posting(llm_provider, posting_text)
    llm_grounding = _check_grounding(posting_text, llm_chunks)

    # Coverage check: does concatenating all chunks (whitespace-insensitive)
    # account for roughly the same amount of text as the source posting, or
    # did the LLM drop/skip large sections? A crude but useful sanity check.
    def _char_count_no_space(text: str) -> int:
        return len("".join(text.split()))

    posting_chars = _char_count_no_space(posting_text)
    regex_chars = sum(_char_count_no_space(c) for c in regex_chunks)
    llm_chars = sum(_char_count_no_space(c) for c in llm_chunks)

    report = {
        "benchmark": "llm_chunking",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "posting_path": str(posting_path),
        "description": (
            "Compares the deterministic regex-based sentence chunker (src/chunker/sentence_chunking.py) "
            "against an LLM-based chunker on the same real posting - chunk count, grounding "
            "(is each chunk an exact substring of the source), and rough text-coverage."
        ),
        "posting_char_count_no_whitespace": posting_chars,
        "regex_chunker": {
            **regex_grounding,
            "coverage_ratio": round(regex_chars / posting_chars, 4) if posting_chars else 0.0,
            "chunks": regex_chunks,
        },
        "llm_chunker": {
            **llm_grounding,
            "coverage_ratio": round(llm_chars / posting_chars, 4) if posting_chars else 0.0,
            "chunks": llm_chunks,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        f"\nregex chunker: {len(regex_chunks)} chunks, {regex_grounding['ungrounded_count']} ungrounded, "
        f"coverage={report['regex_chunker']['coverage_ratio']:.2%}"
    )
    print(
        f"llm chunker:   {len(llm_chunks)} chunks, {llm_grounding['ungrounded_count']} ungrounded, "
        f"coverage={report['llm_chunker']['coverage_ratio']:.2%}"
    )
    print(f"\nWrote {artifact_path}")


if __name__ == "__main__":
    main()
