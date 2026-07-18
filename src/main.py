"""CLI entry point for the skills-only resume tailoring pipeline.

This module defines the top-level configuration and command-line interface for
future implementation. The actual pipeline steps live in dedicated modules and
should be wired in here as they are built out.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Callable, Optional

from llm import get_llm_provider
from parser import load_skill_cache, parse_posting, validate_selected_skills
from render_resume import (
    build_sectioned_skills,
    render_pdf_with_pdflatex,
    render_skills_lines,
    validate_pdf,
    write_tex_from_template,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single resume tailoring run."""

    posting_path: Path
    skills_cache_path: Path = Path("data/skills.yaml")
    template_path: Path = Path("data/template.tex")
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    reasoning_llm_model: str = "gpt-5-mini"
    screening_llm_model: str = "gpt-4o-mini"
    use_llm_parser: bool = True
    max_concurrency: int = 24
    run_name: str | None = None


# Coarse, ordered, human-meaningful stages surfaced to callers via `on_stage`
# (e.g. the webapp's run-progress UI) - deliberately coarser than the
# per-substage `mark_stage(...)` timing labels below (e.g. the whole
# render-compile-validate-trim loop is reported as one "rendering" stage,
# since it can repeat several times and a caller-facing progress bar should
# stay monotonic rather than looping backwards).
PIPELINE_STAGES: list[str] = [
    "read_posting",
    "init_llm_provider",
    "parse_posting",
    "validate_selected_skills",
    "group_skills",
    "rendering",
    "finalizing",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Tailor a resume for a job posting using the skills pipeline."
    )
    parser.add_argument(
        "posting_path",
        type=Path,
        help="Path to a plain-text job posting.",
    )
    parser.add_argument(
        "--skills-cache",
        type=Path,
        default=Path("data/skills.yaml"),
        help="Path to the canonical skills cache YAML file.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/template.tex"),
        help="Path to the LaTeX template used for rendering.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider to use for parsing, validation edge cases, and skill sectioning.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name used for validation grounding fallback and skill-section grouping.",
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="gpt-5-mini",
        help=(
            "Reasoning-tier model driving the parser pipeline's chunking, extraction, "
            "categorization, and Stage 3 atomicity/redundancy stages."
        ),
    )
    parser.add_argument(
        "--screening-model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Cheaper, non-reasoning model used for Stage 0.5 chunk screening - a coarse, batched "
            "pre-filter that skips chunks unlikely to contain any resume-worthy skill before they "
            "reach the more expensive reasoning-model extraction/categorization stages."
        ),
    )
    parser.add_argument(
        "--no-llm-parser",
        action="store_true",
        help="Disable LLM parser path and use deterministic cache-only parsing instead.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=24,
        help="Max concurrent reasoning-model calls in flight at once across all pipeline stages.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name under build/. Defaults to a timestamp-based value.",
    )
    return parser


def _default_run_name() -> str:
    return datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S_%f")


def _build_run_paths(run_name: str) -> dict[str, Path]:
    run_root = Path("build") / run_name
    return {
        "run_root": run_root,
        "aux_dir": run_root / "aux",
        "logs_dir": run_root / "logs",
        "output_tex": run_root / "aux" / "tailored_resume.tex",
        "output_pdf": run_root / "tailored_resume.pdf",
    }


def _setup_run_logger(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("resumeopt.pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    return logger


def _write_json_log(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _merge_always_include_skills(
    ranked_canonical_skills: list[str], skill_cache: list[object]
) -> tuple[list[str], list[str]]:
    """Prepend the user's "always include" skills (see webapp `SkillsPage`/
    `skills_cache_io.update_skill`) onto the ranked, tailored skill list.

    These are the user's fixed baseline (e.g. languages/practices they always
    want listed) - included regardless of tailoring/matching, and NOT subject
    to `validate_selected_skills`' grounding/confidence checks since they
    aren't claimed to appear in this specific posting. Prepended (not
    appended) so the fit-to-budget trim loop, which drops from the END of the
    ranked list first, removes a user-designated "always include" skill only
    as an absolute last resort - only after every tailored/matched skill has
    already been dropped. Already-selected skills (case-insensitive) aren't
    duplicated.

    Returns `(merged_ranked_skills, forced_skills)` - `forced_skills` is the
    subset of always-include skills that weren't already selected via
    tailoring, useful for logging/metrics.
    """

    always_include_skills = [skill.name for skill in skill_cache if skill.always_include]
    already_selected_lower = {name.lower() for name in ranked_canonical_skills}
    forced_skills = [name for name in always_include_skills if name.lower() not in already_selected_lower]
    return forced_skills + ranked_canonical_skills, forced_skills


def _prioritize_always_include_skills(
    included_skills: list[str], skill_cache: list[object]
) -> tuple[list[str], list[str]]:
    """Reorders (but never adds to) a user-confirmed skill list so any
    always-include skill the user chose to KEEP sorts to the front - still
    giving it the fit-to-budget trim loop's "last to be dropped" protection
    (see `_merge_always_include_skills`'s docstring for that rationale) -
    without forcing back in an always-include skill the user explicitly
    unchecked at the Phase 9 review checkpoint. Unlike
    `_merge_always_include_skills`, this never expands the list; it only
    reprioritizes what's already there.

    Returns `(reordered_skills, prioritized_skills)` - `prioritized_skills`
    is the always-include subset that was actually present, for metrics.
    """

    always_include_lower = {skill.name.lower() for skill in skill_cache if skill.always_include}
    prioritized = [name for name in included_skills if name.lower() in always_include_lower]
    others = [name for name in included_skills if name.lower() not in always_include_lower]
    return prioritized + others, prioritized


LOW_CONFIDENCE_THRESHOLD = 0.85
"""Below this confidence, a matched skill is flagged (not hidden) for extra
scrutiny at the Phase 9 human-in-the-loop skill-review checkpoint (see
`_build_skill_review_payload` and `docs/agent/FRONTEND_DEV_PLAN.md`'s Phase
9). Chosen strictly between the `alias` tier's baseline confidence (0.90)
and the `semantic` tier's baseline confidence (0.75, see
`matcher.models.BASE_CONFIDENCE`), so it flags semantic-only matches without
also flagging every alias match."""


def _build_skill_review_payload(
    validation_report: dict[str, object],
    missing_skills: list[str],
    missing_skill_evidence: dict[str, str],
    forced_skills: list[str],
    skill_cache: list[object],
) -> dict[str, object]:
    """Combines Stage 7's validated/deduped matches, the always-include
    skills not already among them, and Stage 1-3's grounded-but-uncached
    missing terms into one reviewable list for the Phase 9 human-in-the-loop
    checkpoint. Deliberately excludes `discarded_terms` (explicitly rejected
    during Stage 2/3 categorization/redundancy) - per explicit user decision,
    rejected terms are "almost always bad" noise, not useful review signal.

    Each reviewable entry is `{name, source, match_type, confidence,
    evidence, low_confidence, default_checked, is_always_include}`:
    - `source` is one of `"matched"` (from `validation_report`),
      `"always_include"` (forced in, not otherwise matched), or `"missing"`
      (extracted, grounded, but not yet in the cache).
    - `evidence` is the grounding sentence for `matched` entries, and the
      posting chunk a `missing` term was first extracted from (so both get
      the same review-time context, not just matched ones).
    - `default_checked` is True for `matched`/`always_include` (the pipeline
      already trusted these enough to select them) and False for `missing`
      (not yet cached - requires an explicit opt-in, consistent with
      `AGENTS.md`'s existing human-review gate on cache writes).
    - `is_always_include` is True for any entry (regardless of source)
      that's an always-include skill in the cache - used by the frontend to
      sort these to the end of the list. Unlike an earlier design, these
      stay a normal, toggleable checkbox (not disabled) - the user can still
      opt one out of a specific run's resume.

    Also returns `other_cache_skills` - every cached skill NOT already in the
    reviewable list, for a UI "add another skill from your cache" escape
    hatch (skills unrelated to this posting that the user still wants
    listed).
    """

    always_include_lower = {skill.name.lower() for skill in skill_cache if skill.always_include}
    seen_lower: set[str] = set()
    reviewable: list[dict[str, object]] = []

    for match in validation_report.get("selected_skills", []):
        name = str(match.get("canonical_name", "")).strip()
        if not name or name.lower() in seen_lower:
            continue
        seen_lower.add(name.lower())
        confidence = match.get("confidence")
        reviewable.append(
            {
                "name": name,
                "source": "matched",
                "match_type": match.get("match_type"),
                "confidence": confidence,
                "evidence": match.get("evidence"),
                "low_confidence": isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD,
                "default_checked": True,
                "is_always_include": name.lower() in always_include_lower,
            }
        )

    for term in missing_skills:
        name = str(term).strip()
        if not name or name.lower() in seen_lower:
            continue
        seen_lower.add(name.lower())
        reviewable.append(
            {
                "name": name,
                "source": "missing",
                "match_type": None,
                "confidence": None,
                "evidence": missing_skill_evidence.get(name.lower()),
                "low_confidence": False,
                "default_checked": False,
                "is_always_include": name.lower() in always_include_lower,
            }
        )

    for name in forced_skills:
        name = str(name).strip()
        if not name or name.lower() in seen_lower:
            continue
        seen_lower.add(name.lower())
        reviewable.append(
            {
                "name": name,
                "source": "always_include",
                "match_type": None,
                "confidence": None,
                "evidence": None,
                "low_confidence": False,
                "default_checked": True,
                "is_always_include": True,
            }
        )

    # Always-include skills sort to the very end of the "from job posting"
    # list (per explicit user decision) - stable sort, so relative order is
    # otherwise unchanged within the non-always-include and always-include
    # groups.
    reviewable.sort(key=lambda entry: entry["is_always_include"])

    other_cache_skills = sorted(
        {skill.name for skill in skill_cache if skill.name.lower() not in seen_lower}
    )

    return {"reviewable_skills": reviewable, "other_cache_skills": other_cache_skills}


def _estimate_tokens_from_text(text: str) -> int:
    """Approximate token count using a conservative char-based heuristic."""

    return (len(text) + 3) // 4


def _estimate_tokens_from_payload(payload: object) -> int:
    serialized = json.dumps(payload, ensure_ascii=True)
    return _estimate_tokens_from_text(serialized)


def _llm_usage_summary(providers: dict[str, object]) -> dict[str, object]:
    """Aggregate real, provider-reported token usage across LLM provider instances.

    Falls back gracefully (actual_usage_available=False) for providers that
    don't expose authoritative usage (currently only OpenAI populates it).
    """

    combined = {
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
    }
    by_role: dict[str, object] = {}
    actual_usage_available = False

    for role, provider in providers.items():
        if provider is None:
            continue
        usage_totals = getattr(provider, "usage_totals", None)
        usage_available = bool(getattr(provider, "usage_available", False))
        if usage_totals is None:
            continue

        by_role[role] = {
            "model": getattr(provider, "model", None),
            "usage_available": usage_available,
            **usage_totals,
        }
        if usage_available:
            actual_usage_available = True
            for key in combined:
                combined[key] += usage_totals.get(key, 0)

    summary: dict[str, object] = {
        "actual_usage_available": actual_usage_available,
        "by_role": by_role,
        "combined": combined,
    }
    if not actual_usage_available:
        summary["note"] = (
            "LLM provider does not expose authoritative token usage for this configuration; "
            "see estimated_token_usage instead."
        )
    return summary


def _merge_llm_usage_summaries(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    # Combines two _llm_usage_summary-shaped dicts, summing per-role and
    # combined totals. Needed because run_pipeline_to_review and
    # run_pipeline_from_review each construct their own LLM provider
    # instances (the Phase 9 checkpoint may pause for an arbitrary amount of
    # time, even across a backend restart, so provider objects can not be
    # kept alive across the split). Both phases can use the same role name
    # (summary_validation_and_sectioning) - Stage 0 posting summary and
    # Stage 7 validation grounding run in the first phase, Stage 8 section
    # grouping runs in the second - so that role's numbers are summed
    # rather than one phase's simply overwriting the other's.
    combined_keys = ("call_count", "prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")
    by_role: dict[str, object] = {}
    for role, data in (first.get("by_role") or {}).items():
        by_role[role] = dict(data)
    for role, data in (second.get("by_role") or {}).items():
        if role in by_role:
            merged = dict(by_role[role])
            for key in combined_keys:
                merged[key] = merged.get(key, 0) + data.get(key, 0)
            merged["usage_available"] = merged.get("usage_available", False) or data.get("usage_available", False)
            by_role[role] = merged
        else:
            by_role[role] = dict(data)

    combined = {key: 0 for key in combined_keys}
    for data in by_role.values():
        for key in combined_keys:
            combined[key] += data.get(key, 0)

    actual_usage_available = bool(first.get("actual_usage_available")) or bool(second.get("actual_usage_available"))
    result: dict[str, object] = {
        "actual_usage_available": actual_usage_available,
        "by_role": by_role,
        "combined": combined,
    }
    if not actual_usage_available:
        result["note"] = first.get("note") or second.get("note")
    return result


def _write_llm_call_log(
    logs_dir: Path, providers: dict[str, object], merge_existing: bool = False
) -> None:
    """Write each provider's per-call structured log (`call_label`,
    prompt/completion/reasoning tokens per call) to `llm_call_log.json`.

    Added 2026-07-17 alongside `reasoning_effort` support to make call-count
    and token-spend questions ("where exactly are N calls coming from?")
    directly answerable from a run's own logs - grouped by role and, within
    each role, listed per call in call order - instead of only ever seeing
    aggregate totals via `_llm_usage_summary`.

    `merge_existing` (for the Phase 9 two-phase pipeline split): when True,
    reads back any `llm_call_log.json` already on disk first and extends
    each role's call list rather than overwriting it - used by
    `run_pipeline_from_review` so its own (e.g. `group_skills`) calls are
    appended to `run_pipeline_to_review`'s, not lost.
    """

    by_role: dict[str, object] = {}
    if merge_existing:
        existing_path = logs_dir / "llm_call_log.json"
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                for role, data in (existing.get("by_role") or {}).items():
                    by_role[role] = {
                        "model": data.get("model"),
                        "call_count": data.get("call_count", 0),
                        "calls": list(data.get("calls", [])),
                    }
            except (OSError, ValueError):
                pass

    for role, provider in providers.items():
        if provider is None:
            continue
        call_log = getattr(provider, "call_log", None)
        if call_log is None:
            continue
        if role in by_role:
            by_role[role]["calls"].extend(call_log)
            by_role[role]["call_count"] = len(by_role[role]["calls"])
            by_role[role]["model"] = by_role[role]["model"] or getattr(provider, "model", None)
        else:
            by_role[role] = {
                "model": getattr(provider, "model", None),
                "call_count": len(call_log),
                "calls": call_log,
            }
    _write_json_log(logs_dir / "llm_call_log.json", {"by_role": by_role})


def run_pipeline_to_review(
    config: PipelineConfig,
    on_stage: Optional[Callable[[str], None]] = None,
    on_substage: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, object]:
    """Runs the pipeline through Stage 7 (parse, match, validate/rank/dedupe)
    and stops - the human-in-the-loop skill-review checkpoint (Phase 9, see
    docs/agent/FRONTEND_DEV_PLAN.md). Writes the same Stage 0-7 artifacts a
    full run always has (parsed_records.json, extraction_debug.json,
    missing_skills*.json, validation_report.json), PLUS a new
    skill_review.json (the reviewable skill list - see
    _build_skill_review_payload) and a PARTIAL run_metrics.json (posting/
    parse/validation sections only - run_pipeline_from_review fills in the
    rest once the user confirms their selection).

    Returns the same payload written to skill_review.json, so a caller (e.g.
    webapp.run_manager.RunManager) can serve it without a disk round-trip
    for a still-in-process run.

    on_stage/on_substage: same progress-reporting hooks as the old
    single-shot run_pipeline used to accept - see run_pipeline below.
    """

    run_name = config.run_name or _default_run_name()
    run_paths = _build_run_paths(run_name)
    run_paths["aux_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    logger = _setup_run_logger(run_paths["logs_dir"])
    run_start = time.perf_counter()
    stage_timings_ms: dict[str, int] = {}
    metrics: dict[str, object] = {}

    def mark_stage(stage: str, stage_start: float) -> None:
        stage_timings_ms[stage] = int((time.perf_counter() - stage_start) * 1000)

    def notify_stage(stage: str) -> None:
        if on_stage is None:
            return
        try:
            on_stage(stage)
        except Exception:
            logger.exception("on_stage callback failed for stage=%s", stage)

    def notify_substage(name: str, completed: int, total: int) -> None:
        if on_substage is None:
            return
        try:
            on_substage(name, completed, total)
        except Exception:
            logger.exception("on_substage callback failed for substage=%s", name)

    logger.info("Starting pipeline (to-review phase) run=%s", run_name)
    logger.info("Paths aux=%s logs=%s", run_paths["aux_dir"], run_paths["logs_dir"])
    reasoning_llm: object | None = None
    screening_llm: object | None = None
    llm: object | None = None
    _write_json_log(
        run_paths["logs_dir"] / "run_config.json",
        {
            "run_name": run_name,
            "posting_path": str(config.posting_path),
            "skills_cache_path": str(config.skills_cache_path),
            "template_path": str(config.template_path),
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "reasoning_llm_model": config.reasoning_llm_model,
            "screening_llm_model": config.screening_llm_model,
            "use_llm_parser": config.use_llm_parser,
            "max_concurrency": config.max_concurrency,
            "output_tex": str(run_paths["output_tex"]),
            "output_pdf": str(run_paths["output_pdf"]),
        },
    )

    try:
        notify_stage("read_posting")
        stage_start = time.perf_counter()
        posting_text = config.posting_path.read_text(encoding="utf-8")
        mark_stage("read_posting", stage_start)

        posting_estimated_tokens = _estimate_tokens_from_text(posting_text)
        metrics["posting"] = {
            "characters": len(posting_text),
            "words": len(posting_text.split()),
            "estimated_tokens": posting_estimated_tokens,
        }

        notify_stage("init_llm_provider")
        stage_start = time.perf_counter()
        llm = get_llm_provider(config.llm_provider, model=config.llm_model)
        if config.use_llm_parser:
            reasoning_llm = get_llm_provider(config.llm_provider, model=config.reasoning_llm_model)
            screening_llm = get_llm_provider(config.llm_provider, model=config.screening_llm_model)
        mark_stage("init_llm_provider", stage_start)

        notify_stage("parse_posting")
        stage_start = time.perf_counter()
        # Sentence/chunk-level reasoning-model pipeline (Stage 0.5 cheap chunk
        # screening, Stage 1 recall-first extraction, Stage 2 4-category
        # classification, Stage 3a context-free keyword-atomicity gate, Stage
        # 3b within-chunk redundancy check for non-atomic terms only). Reuses
        # the same judge-tier `llm` (config.llm_model, gpt-4o) for the Stage 0
        # posting summary and for skill-section grouping/validation.
        records = parse_posting(
            posting_text=posting_text,
            skills_cache_path=config.skills_cache_path,
            use_llm=config.use_llm_parser,
            summary_llm_provider=llm,
            reasoning_llm_provider=reasoning_llm,
            screening_llm_provider=screening_llm,
            max_concurrency=config.max_concurrency,
            on_substage=notify_substage,
        )
        mark_stage("parse_posting", stage_start)
        _write_json_log(run_paths["logs_dir"] / "parsed_records.json", records)
        _write_json_log(
            run_paths["logs_dir"] / "extraction_debug.json",
            [
                {
                    "posting_line": record.get("posting_line", ""),
                    "extraction_debug_samples": record.get("extraction_debug_samples", []),
                }
                for record in records
            ],
        )

        missing_skills: list[str] = []
        seen_missing: set[str] = set()
        missing_skill_evidence: dict[str, str] = {}
        discarded_terms: list[dict[str, object]] = []
        for record in records:
            record_missing_evidence = record.get("missing_skills_evidence", {})
            for term in record.get("missing_skills", []):
                normalized = str(term).strip().lower()
                if not normalized or normalized in seen_missing:
                    continue
                missing_skills.append(str(term).strip())
                seen_missing.add(normalized)
                missing_skill_evidence[normalized] = record_missing_evidence.get(
                    term, record.get("posting_line", "")
                )
            for discarded in record.get("missing_skills_discarded", []):
                if isinstance(discarded, dict):
                    discarded_terms.append(discarded)
        _write_json_log(
            run_paths["logs_dir"] / "missing_skills_candidates.json",
            {"missing_skills": missing_skills, "count": len(missing_skills)},
        )
        _write_json_log(
            run_paths["logs_dir"] / "missing_skills.json",
            {"missing_skills": missing_skills, "count": len(missing_skills)},
        )
        _write_json_log(
            run_paths["logs_dir"] / "missing_skills_discarded.json",
            {"discarded_terms": discarded_terms, "count": len(discarded_terms)},
        )

        parsed_match_count = sum(len(record.get("matched_skills", [])) for record in records)
        parse_estimated_tokens = _estimate_tokens_from_payload(records)
        metrics["parse"] = {
            "record_count": len(records),
            "matched_skill_count": parsed_match_count,
            "missing_skill_count": len(missing_skills),
            "discarded_term_count": len(discarded_terms),
            "estimated_output_tokens": parse_estimated_tokens,
        }

        stage_start = time.perf_counter()
        posting_summary = None
        if records:
            debug_samples = records[0].get("extraction_debug_samples", [])
            if debug_samples and isinstance(debug_samples[0], dict):
                posting_summary = debug_samples[0].get("posting_summary")
        notify_stage("validate_selected_skills")
        validation_report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=config.skills_cache_path,
            llm_provider=llm,
            posting_summary=posting_summary,
        )
        mark_stage("validate_selected_skills", stage_start)
        _write_json_log(run_paths["logs_dir"] / "validation_report.json", validation_report)
        if validation_report["status"] != "pass":
            raise ValueError(f"Validation failed: {validation_report}")

        validation_estimated_tokens = _estimate_tokens_from_payload(validation_report)
        metrics["validation"] = {
            "status": validation_report.get("status"),
            "issue_count": len(validation_report.get("issues", [])),
            "selected_skill_count": len(validation_report.get("selected_skills", [])),
            "estimated_output_tokens": validation_estimated_tokens,
        }

        skill_cache = load_skill_cache(config.skills_cache_path)
        ranked_canonical_skills = [
            str(match.get("canonical_name", "")).strip()
            for match in validation_report["selected_skills"]
            if str(match.get("canonical_name", "")).strip()
        ]
        _, forced_skills = _merge_always_include_skills(ranked_canonical_skills, skill_cache)

        review_payload = _build_skill_review_payload(
            validation_report=validation_report,
            missing_skills=missing_skills,
            missing_skill_evidence=missing_skill_evidence,
            forced_skills=forced_skills,
            skill_cache=skill_cache,
        )
        _write_json_log(run_paths["logs_dir"] / "skill_review.json", review_payload)

        stage_timings_ms["total"] = int((time.perf_counter() - run_start) * 1000)
        metrics["timings_ms"] = stage_timings_ms
        metrics["estimated_token_usage"] = {
            "posting_input": posting_estimated_tokens,
            "parse_output": parse_estimated_tokens,
            "validation_output": validation_estimated_tokens,
        }
        metrics["llm_usage"] = _llm_usage_summary(
            {
                "reasoning": reasoning_llm,
                "screening": screening_llm,
                "summary_validation_and_sectioning": llm,
            }
        )
        _write_llm_call_log(
            run_paths["logs_dir"],
            {
                "reasoning": reasoning_llm,
                "screening": screening_llm,
                "summary_validation_and_sectioning": llm,
            },
        )
        _write_json_log(run_paths["logs_dir"] / "run_metrics.json", metrics)
        logger.info("Stage timings ms=%s", stage_timings_ms)
        logger.info("LLM usage=%s", metrics["llm_usage"])
        logger.info("Pipeline paused for skill review run=%s", run_name)
        return review_payload
    except Exception as exc:
        stage_timings_ms["total"] = int((time.perf_counter() - run_start) * 1000)
        metrics["timings_ms"] = stage_timings_ms
        metrics["llm_usage"] = _llm_usage_summary(
            {
                "reasoning": reasoning_llm,
                "screening": screening_llm,
                "summary_validation_and_sectioning": llm,
            }
        )
        _write_llm_call_log(
            run_paths["logs_dir"],
            {
                "reasoning": reasoning_llm,
                "screening": screening_llm,
                "summary_validation_and_sectioning": llm,
            },
        )
        _write_json_log(run_paths["logs_dir"] / "run_metrics.json", metrics)
        logger.exception("Pipeline (to-review phase) failed run=%s error=%s", run_name, exc)
        (run_paths["logs_dir"] / "error_traceback.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def run_pipeline_from_review(
    config: PipelineConfig,
    included_skills: list[str],
    on_stage: Optional[Callable[[str], None]] = None,
) -> None:
    """Resumes a run after the user has confirmed their final skill
    selection at the Phase 9 review checkpoint - groups, renders, compiles,
    PDF-validates, and fit-to-budget-trims exactly like the back half of the
    old single-shot run_pipeline used to.

    Safe to call more than once for the same run (e.g. the user goes back
    and re-confirms a different selection after seeing a fit-to-budget trim
    warning, per the "allow rerendering" decision in FRONTEND_DEV_PLAN.md's
    Phase 9) - each call simply re-renders from scratch and overwrites the
    previous tex/pdf output, merging its own metrics into whatever
    run_pipeline_to_review already persisted.

    config.run_name MUST already be set - this resumes an existing run
    directory, it never creates a fresh timestamp-based one.
    """

    run_name = config.run_name
    if not run_name:
        raise ValueError("run_pipeline_from_review requires config.run_name to be set")
    run_paths = _build_run_paths(run_name)
    logger = _setup_run_logger(run_paths["logs_dir"])
    run_start = time.perf_counter()
    stage_timings_ms: dict[str, int] = {}
    metrics: dict[str, object] = {}

    def mark_stage(stage: str, stage_start: float) -> None:
        stage_timings_ms[stage] = int((time.perf_counter() - stage_start) * 1000)

    def notify_stage(stage: str) -> None:
        if on_stage is None:
            return
        try:
            on_stage(stage)
        except Exception:
            logger.exception("on_stage callback failed for stage=%s", stage)

    logger.info("Resuming pipeline (from-review phase) run=%s", run_name)

    prior_metrics: dict[str, object] = {}
    metrics_path = run_paths["logs_dir"] / "run_metrics.json"
    if metrics_path.exists():
        try:
            prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("Failed to read prior run_metrics.json for run=%s", run_name)

    posting_summary = None
    debug_path = run_paths["logs_dir"] / "extraction_debug.json"
    if debug_path.exists():
        try:
            debug_records = json.loads(debug_path.read_text(encoding="utf-8"))
            if debug_records:
                debug_samples = debug_records[0].get("extraction_debug_samples", [])
                if debug_samples and isinstance(debug_samples[0], dict):
                    posting_summary = debug_samples[0].get("posting_summary")
        except (OSError, ValueError):
            logger.exception("Failed to read extraction_debug.json for run=%s", run_name)

    llm: object | None = None
    try:
        llm = get_llm_provider(config.llm_provider, model=config.llm_model)

        skill_cache = load_skill_cache(config.skills_cache_path)
        # Re-prioritizes (never force-adds) - the user's confirmed selection
        # at the Phase 9 review checkpoint is authoritative, including any
        # always-include skill they explicitly unchecked. See
        # _prioritize_always_include_skills' docstring for why this is a
        # different function than _merge_always_include_skills (used only
        # by run_pipeline_to_review/the CLI wrapper, where nothing has been
        # explicitly reviewed/opted-out of yet).
        ranked_canonical_skills, forced_skills = _prioritize_always_include_skills(
            [name for name in included_skills if str(name).strip()], skill_cache
        )

        stage_start = time.perf_counter()
        notify_stage("group_skills")
        posting_context = None
        if posting_summary:
            posting_context = (
                f"Role: {posting_summary.get('role_title', '')} ({posting_summary.get('seniority', '')})\n"
                f"Domain: {posting_summary.get('industry_domain', '')}\n"
                f"{posting_summary.get('summary_paragraph', '')}"
            )
        sectioned = build_sectioned_skills(
            canonical_skills=ranked_canonical_skills, llm_provider=llm, posting_context=posting_context
        )
        mark_stage("group_skills", stage_start)
        _write_json_log(run_paths["logs_dir"] / "sectioned_skills.json", sectioned)

        notify_stage("rendering")
        # Iterative fit-to-budget loop (2026-07-17, replaces the old one-shot
        # "truncate to a fixed skill count, render once, hope it fits"
        # approach): render -> compile -> check the ACTUAL rendered line
        # count (LaTeX line-wrapping is fragile to predict without actually
        # compiling) -> if over budget, drop the single LOWEST-ranked
        # remaining skill and retry. Categories/grouping are computed ONCE
        # above (not re-asked from the LLM each iteration) - a dropped skill
        # is simply removed from whichever section list already contains it.
        remaining_ranked = list(ranked_canonical_skills)
        trim_iterations = 0
        pdf_validation_report: dict[str, object] = {}
        while True:
            stage_start = time.perf_counter()
            skills_block = render_skills_lines(sectioned)
            mark_stage("render_skills_lines", stage_start)

            stage_start = time.perf_counter()
            write_tex_from_template(
                template_path=config.template_path,
                output_tex_path=run_paths["output_tex"],
                skills_block=skills_block,
            )
            mark_stage("write_tex", stage_start)

            stage_start = time.perf_counter()
            render_pdf_with_pdflatex(
                run_paths["output_tex"],
                run_paths["output_pdf"],
                logs_dir=run_paths["logs_dir"],
            )
            mark_stage("render_pdf", stage_start)

            stage_start = time.perf_counter()
            pdf_validation_report = validate_pdf(run_paths["output_pdf"])
            mark_stage("validate_pdf", stage_start)

            if pdf_validation_report["status"] == "pass":
                break

            too_long = any(
                issue.get("type") == "skills_section_too_long"
                for issue in pdf_validation_report.get("issues", [])
            )
            if not too_long or len(remaining_ranked) <= 1:
                break

            dropped = remaining_ranked.pop()
            for skills in sectioned.values():
                if dropped in skills:
                    skills.remove(dropped)
                    break
            sectioned = {name: skills for name, skills in sectioned.items() if skills}
            trim_iterations += 1
            logger.info("Skills section over budget - dropped %r and retrying (iteration %d)", dropped, trim_iterations)

        (run_paths["logs_dir"] / "skills_block.tex.log").write_text(skills_block + "\n", encoding="utf-8")
        _write_json_log(run_paths["logs_dir"] / "pdf_validation.json", pdf_validation_report)
        if pdf_validation_report["status"] != "pass":
            raise ValueError(f"PDF validation failed: {pdf_validation_report}")

        # Records exactly what the user confirmed at the Phase 9 review
        # checkpoint (`included_skills` - the full, pre-trim intent, so
        # re-opening the review UI later can restore their previous
        # checkbox choices instead of resetting to the original defaults)
        # alongside what actually survived the fit-to-budget trim loop and
        # is on the rendered PDF (`final_skills`) - the "Selected skills"/
        # "Missing skills" sections on a completed run read this instead of
        # the pre-review `validation_report.json`/`missing_skills.json`,
        # which don't reflect the user's review-time choices at all.
        _write_json_log(
            run_paths["logs_dir"] / "confirmed_skills.json",
            {"included_skills": ranked_canonical_skills, "final_skills": remaining_ranked},
        )

        notify_stage("finalizing")
        skills_block_estimated_tokens = _estimate_tokens_from_text(skills_block)
        metrics["skills_block"] = {
            "active_sections": [section for section, skills in sectioned.items() if skills],
            "active_section_count": len([section for section, skills in sectioned.items() if skills]),
            "canonical_skill_count": len(remaining_ranked),
            "always_include_skill_count": len(forced_skills),
            "trim_iterations": trim_iterations,
            "characters": len(skills_block),
            "estimated_tokens": skills_block_estimated_tokens,
        }

        metrics["pdf_validation"] = {
            "status": pdf_validation_report.get("status"),
            "page_count": pdf_validation_report.get("page_count"),
            "skills_section_line_count": pdf_validation_report.get("skills_section_line_count"),
            "issue_count": len(pdf_validation_report.get("issues", [])),
        }

        metrics["artifacts"] = {
            "output_tex_bytes": run_paths["output_tex"].stat().st_size if run_paths["output_tex"].exists() else 0,
            "output_pdf_bytes": run_paths["output_pdf"].stat().st_size if run_paths["output_pdf"].exists() else 0,
            "aux_file_count": len(list(run_paths["aux_dir"].glob("*"))),
            "log_file_count": len(list(run_paths["logs_dir"].glob("*"))),
        }

        merged_metrics = dict(prior_metrics)
        merged_metrics.update(metrics)

        merged_timings = dict(prior_metrics.get("timings_ms", {}))
        merged_timings.update(stage_timings_ms)
        merged_timings["total"] = sum(value for key, value in merged_timings.items() if key != "total")
        merged_metrics["timings_ms"] = merged_timings

        prior_tokens = prior_metrics.get("estimated_token_usage", {}) or {}
        estimated_total = (
            prior_tokens.get("posting_input", 0)
            + prior_tokens.get("parse_output", 0)
            + prior_tokens.get("validation_output", 0)
            + skills_block_estimated_tokens
        )
        merged_metrics["estimated_token_usage"] = {
            **prior_tokens,
            "skills_block_output": skills_block_estimated_tokens,
            "estimated_total": estimated_total,
        }

        this_phase_usage = _llm_usage_summary({"summary_validation_and_sectioning": llm})
        merged_metrics["llm_usage"] = _merge_llm_usage_summaries(
            prior_metrics.get("llm_usage", {}) or {}, this_phase_usage
        )
        _write_llm_call_log(
            run_paths["logs_dir"],
            {"summary_validation_and_sectioning": llm},
            merge_existing=True,
        )
        _write_json_log(run_paths["logs_dir"] / "run_metrics.json", merged_metrics)
        logger.info("Stage timings ms=%s", merged_timings)
        logger.info("Pipeline completed successfully (from-review phase) run=%s", run_name)
    except Exception as exc:
        stage_timings_ms["total"] = int((time.perf_counter() - run_start) * 1000)
        merged_timings = dict(prior_metrics.get("timings_ms", {}))
        merged_timings.update(stage_timings_ms)
        merged_metrics = dict(prior_metrics)
        merged_metrics["timings_ms"] = merged_timings
        _write_json_log(run_paths["logs_dir"] / "run_metrics.json", merged_metrics)
        logger.exception("Pipeline (from-review phase) failed run=%s error=%s", run_name, exc)
        (run_paths["logs_dir"] / "error_traceback.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise

    print(f"Run output directory: {run_paths['run_root']}")
    print(f"Aux files: {run_paths['aux_dir']}")
    print(f"Logs: {run_paths['logs_dir']}")


def run_pipeline(
    config: PipelineConfig,
    on_stage: Optional[Callable[[str], None]] = None,
    on_substage: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Runs the full pipeline end-to-end with no human review pause - the
    CLI's entry point, and the default for any caller that doesn't want the
    Phase 9 skill-review checkpoint. Auto-accepts every matched/always-include
    skill and no missing skills (equivalent to what a user clicking
    "Confirm" with no changes at the review checkpoint would produce), so
    CLI/test behavior is unchanged from before the Phase 9 split.

    See run_pipeline_to_review/run_pipeline_from_review for the two
    separately-invokable halves this wraps - webapp.run_manager.RunManager
    calls those directly (not this wrapper) so it can actually pause for
    human review between them.
    """

    review_payload = run_pipeline_to_review(config, on_stage=on_stage, on_substage=on_substage)
    default_included = [
        skill["name"] for skill in review_payload["reviewable_skills"] if skill["default_checked"]
    ]
    run_pipeline_from_review(config, default_included, on_stage=on_stage)


def main() -> None:
    """Parse CLI arguments and dispatch the pipeline."""

    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig(
        posting_path=args.posting_path,
        skills_cache_path=args.skills_cache,
        template_path=args.template,
        llm_provider=args.provider,
        llm_model=args.model,
        reasoning_llm_model=args.reasoning_model,
        screening_llm_model=args.screening_model,
        use_llm_parser=not args.no_llm_parser,
        max_concurrency=args.max_concurrency,
        run_name=args.run_name,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
