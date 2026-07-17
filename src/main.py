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


def _write_llm_call_log(logs_dir: Path, providers: dict[str, object]) -> None:
    """Write each provider's per-call structured log (`call_label`,
    prompt/completion/reasoning tokens per call) to `llm_call_log.json`.

    Added 2026-07-17 alongside `reasoning_effort` support to make call-count
    and token-spend questions ("where exactly are N calls coming from?")
    directly answerable from a run's own logs - grouped by role and, within
    each role, listed per call in call order - instead of only ever seeing
    aggregate totals via `_llm_usage_summary`.
    """

    by_role: dict[str, object] = {}
    for role, provider in providers.items():
        if provider is None:
            continue
        call_log = getattr(provider, "call_log", None)
        if call_log is None:
            continue
        by_role[role] = {
            "model": getattr(provider, "model", None),
            "call_count": len(call_log),
            "calls": call_log,
        }
    _write_json_log(logs_dir / "llm_call_log.json", {"by_role": by_role})


def run_pipeline(
    config: PipelineConfig,
    on_stage: Optional[Callable[[str], None]] = None,
    on_substage: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Run parse, validate, section, template injection, and PDF rendering.

    `on_stage` (optional) is called with one of `PIPELINE_STAGES` each time the
    pipeline moves into that stage - purely a progress-reporting hook (e.g.
    for `webapp.run_manager.RunManager` to expose real stage-by-stage progress
    instead of an indeterminate UI animation). A raising/misbehaving callback
    is caught and logged, never allowed to fail the actual pipeline run.

    `on_substage` (optional) is called as `(substage_name, completed, total)`
    for real, incremental batch-level progress WITHIN the `"parse_posting"`
    stage (see `parser.pipeline.run_parser_pipeline`'s own `on_substage`
    docstring for the 5 substage names) - `parse_posting` is typically the
    single longest-running, most opaque stage, so this gives a caller-facing
    progress bar real motion during it instead of a single static segment.
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

    logger.info("Starting pipeline run=%s", run_name)
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
        discarded_terms: list[dict[str, object]] = []
        for record in records:
            for term in record.get("missing_skills", []):
                normalized = str(term).strip().lower()
                if not normalized or normalized in seen_missing:
                    continue
                missing_skills.append(str(term).strip())
                seen_missing.add(normalized)
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

        stage_start = time.perf_counter()
        # Ranked best-first (requirement-tier/relevance/confidence/match-type -
        # see selection.select_skills) - the fit-to-budget trim loop below
        # drops from the END of this order first.
        ranked_canonical_skills = [
            str(match.get("canonical_name", "")).strip()
            for match in validation_report["selected_skills"]
            if str(match.get("canonical_name", "")).strip()
        ]

        ranked_canonical_skills, forced_skills = _merge_always_include_skills(
            ranked_canonical_skills, load_skill_cache(config.skills_cache_path)
        )

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

        stage_timings_ms["total"] = int((time.perf_counter() - run_start) * 1000)
        metrics["timings_ms"] = stage_timings_ms
        estimated_token_total = (
            posting_estimated_tokens
            + parse_estimated_tokens
            + validation_estimated_tokens
            + skills_block_estimated_tokens
        )
        metrics["estimated_token_usage"] = {
            "posting_input": posting_estimated_tokens,
            "parse_output": parse_estimated_tokens,
            "validation_output": validation_estimated_tokens,
            "skills_block_output": skills_block_estimated_tokens,
            "estimated_total": estimated_token_total,
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
        logger.info("Estimated token usage=%s", metrics["estimated_token_usage"])
        logger.info("LLM usage=%s", metrics["llm_usage"])
        logger.info("Pipeline completed successfully run=%s", run_name)
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
        logger.exception("Pipeline failed run=%s error=%s", run_name, exc)
        (run_paths["logs_dir"] / "error_traceback.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise

    print(f"Run output directory: {run_paths['run_root']}")
    print(f"Aux files: {run_paths['aux_dir']}")
    print(f"Logs: {run_paths['logs_dir']}")


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
