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

from llm import get_llm_provider
from parse_posting import parse_posting, validate_selected_skills
from render_resume import (
    build_sectioned_skills,
    render_pdf_with_pdflatex,
    render_skills_lines,
    write_tex_from_template,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single resume tailoring run."""

    posting_path: Path
    skills_cache_path: Path = Path("data/skills.yaml")
    template_path: Path = Path("data/template.tex")
    output_tex_path: Path = Path("build/tailored_resume.tex")
    output_pdf_path: Path = Path("build/tailored_resume.pdf")
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    use_llm_parser: bool = True
    run_name: str | None = None


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
        "--output-tex",
        type=Path,
        default=Path("build/tailored_resume.tex"),
        help="Where the generated LaTeX file should be written.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("build/tailored_resume.pdf"),
        help="Where the rendered PDF should be written.",
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
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--no-llm-parser",
        action="store_true",
        help="Disable LLM parser path and use deterministic parsing only.",
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


def _estimate_tokens_from_text(text: str) -> int:
    """Approximate token count using a conservative char-based heuristic."""

    return (len(text) + 3) // 4


def _estimate_tokens_from_payload(payload: object) -> int:
    serialized = json.dumps(payload, ensure_ascii=True)
    return _estimate_tokens_from_text(serialized)


def run_pipeline(config: PipelineConfig) -> None:
    """Run parse, validate, section, template injection, and PDF rendering."""

    run_name = config.run_name or _default_run_name()
    run_paths = _build_run_paths(run_name)
    run_paths["aux_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    logger = _setup_run_logger(run_paths["logs_dir"])
    run_start = time.perf_counter()
    stage_timings_ms: dict[str, int] = {}
    metrics: dict[str, object] = {
        "llm_usage": {
            "actual_usage_available": False,
            "note": "LLM provider wrapper currently exposes no authoritative token usage; values below are estimates.",
        }
    }

    def mark_stage(stage: str, stage_start: float) -> None:
        stage_timings_ms[stage] = int((time.perf_counter() - stage_start) * 1000)

    logger.info("Starting pipeline run=%s", run_name)
    logger.info("Paths aux=%s logs=%s", run_paths["aux_dir"], run_paths["logs_dir"])
    _write_json_log(
        run_paths["logs_dir"] / "run_config.json",
        {
            "run_name": run_name,
            "posting_path": str(config.posting_path),
            "skills_cache_path": str(config.skills_cache_path),
            "template_path": str(config.template_path),
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "use_llm_parser": config.use_llm_parser,
            "output_tex": str(run_paths["output_tex"]),
            "output_pdf": str(run_paths["output_pdf"]),
        },
    )

    try:
        stage_start = time.perf_counter()
        posting_text = config.posting_path.read_text(encoding="utf-8")
        mark_stage("read_posting", stage_start)

        posting_estimated_tokens = _estimate_tokens_from_text(posting_text)
        metrics["posting"] = {
            "characters": len(posting_text),
            "words": len(posting_text.split()),
            "estimated_tokens": posting_estimated_tokens,
        }

        stage_start = time.perf_counter()
        llm = get_llm_provider(config.llm_provider, model=config.llm_model)
        mark_stage("init_llm_provider", stage_start)

        stage_start = time.perf_counter()
        records = parse_posting(
            posting_text=posting_text,
            skills_cache_path=config.skills_cache_path,
            llm_provider=llm,
            use_llm=config.use_llm_parser,
        )
        mark_stage("parse_posting", stage_start)
        _write_json_log(run_paths["logs_dir"] / "parsed_records.json", records)

        parsed_match_count = sum(len(record.get("matched_skills", [])) for record in records)
        parse_estimated_tokens = _estimate_tokens_from_payload(records)
        metrics["parse"] = {
            "record_count": len(records),
            "matched_skill_count": parsed_match_count,
            "estimated_output_tokens": parse_estimated_tokens,
        }

        stage_start = time.perf_counter()
        validation_report = validate_selected_skills(
            records=records,
            posting_text=posting_text,
            skills_cache_path=config.skills_cache_path,
            llm_provider=llm,
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
        canonical_skills = [
            str(match.get("canonical_name", "")).strip()
            for match in validation_report["selected_skills"]
            if str(match.get("canonical_name", "")).strip()
        ]
        sectioned = build_sectioned_skills(canonical_skills=canonical_skills, llm_provider=llm)
        mark_stage("group_skills", stage_start)
        _write_json_log(run_paths["logs_dir"] / "sectioned_skills.json", sectioned)

        stage_start = time.perf_counter()
        skills_block = render_skills_lines(sectioned)
        mark_stage("render_skills_lines", stage_start)
        (run_paths["logs_dir"] / "skills_block.tex.log").write_text(skills_block + "\n", encoding="utf-8")

        skills_block_estimated_tokens = _estimate_tokens_from_text(skills_block)
        metrics["skills_block"] = {
            "active_sections": [section for section, skills in sectioned.items() if skills],
            "active_section_count": len([section for section, skills in sectioned.items() if skills]),
            "canonical_skill_count": len(canonical_skills),
            "characters": len(skills_block),
            "estimated_tokens": skills_block_estimated_tokens,
        }

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
        _write_json_log(run_paths["logs_dir"] / "run_metrics.json", metrics)
        logger.info("Stage timings ms=%s", stage_timings_ms)
        logger.info("Estimated token usage=%s", metrics["estimated_token_usage"])
        logger.info("Pipeline completed successfully run=%s", run_name)
    except Exception as exc:
        stage_timings_ms["total"] = int((time.perf_counter() - run_start) * 1000)
        metrics["timings_ms"] = stage_timings_ms
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
        output_tex_path=args.output_tex,
        output_pdf_path=args.output_pdf,
        llm_provider=args.provider,
        llm_model=args.model,
        use_llm_parser=not args.no_llm_parser,
        run_name=args.run_name,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
