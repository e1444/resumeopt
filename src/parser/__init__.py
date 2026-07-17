"""Posting parser package: sentence/chunk-level skill extraction via a
reasoning model, as a pipeline (Stage 0.5 cheap chunk screening; Stage 1
extraction, deliberately recall-first; Stage 2 multi-class categorical
classification, precision-tightening; Stage 3a context-free keyword-
atomicity gate; Stage 3b within-chunk redundancy check for non-atomic terms
only).

Layout:
- base.py             PostingParser ABC + DeterministicPostingParser
  (cache loading + cache-backed matching; also the no-LLM fallback strategy)
- selection.py         select_skills / validate_selected_skills (shared
  final-selection stage, used regardless of parse strategy)
- summary.py           Stage 0: generate_posting_summary + format_summary_block
- chunk_screening.py    Stage 0.5: cheap, batched, non-reasoning-model screen
  that skips chunks unlikely to contain any resume-worthy skill at all
- extraction.py         Stage 1: per-chunk candidate skill extraction
- categorization.py     Stage 2: 4-category classification (resume_technical_skill /
  degree_or_qualification / soft_skill / non_skill)
- keyword_atomicity.py  Stage 3a: context-free "is this an independently
  standalone keyword" gate
- redundancy.py         Stage 3b: within-chunk redundancy check, only for
  terms that failed the Stage 3a atomicity gate
- models.py             PostingSummary, ChunkSkillVerdict data models
- pipeline.py           run_parser_pipeline() - orchestrates all stages
- factory.py            parse_posting() - top-level entry point, wires the
  pipeline into cache matching and produces the final parser-record shape

Chunking (`split_into_sentence_chunks`, `split_into_sentence_chunks_via_llm`)
lives in the `chunker` package. Matching (ExactAliasMatcher, SemanticMatcher,
LLMGroundingMatcher, SkillRecord, EmbeddingCache, etc.) lives in the
`matcher` package - import those directly, not from `parser`.

Note: earlier decompose-then-classify architectures (a 6-parallel-classifier
pipeline, self-consistency-voted single-shot/orchestra-single-shot
strategies, and a separate multi-call `relevance` package) were retired
after this pipeline was benchmarked to match or beat all of them on
precision/recall while being simpler - see repo memory for the full
architecture history.
"""

from __future__ import annotations

from .base import DeterministicPostingParser, PostingParser, load_skill_cache
from .categorization import CATEGORIES, INCLUDED_CATEGORY, categorize_candidates_for_chunks
from .chunk_screening import screen_chunks_for_skill_likelihood
from .extraction import extract_candidates_for_chunks
from .factory import parse_posting
from .keyword_atomicity import check_keyword_atomicity
from .models import ChunkSkillVerdict, PostingSummary
from .pipeline import run_parser_pipeline
from .redundancy import check_redundancy_for_chunks
from .selection import select_skills, validate_selected_skills
from .summary import format_summary_block, generate_posting_summary

__all__ = [
    "PostingParser",
    "DeterministicPostingParser",
    "load_skill_cache",
    "select_skills",
    "validate_selected_skills",
    "parse_posting",
    "run_parser_pipeline",
    "extract_candidates_for_chunks",
    "categorize_candidates_for_chunks",
    "screen_chunks_for_skill_likelihood",
    "check_keyword_atomicity",
    "check_redundancy_for_chunks",
    "generate_posting_summary",
    "format_summary_block",
    "CATEGORIES",
    "INCLUDED_CATEGORY",
    "ChunkSkillVerdict",
    "PostingSummary",
]
