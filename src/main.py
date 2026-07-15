"""CLI entry point for the skills-only resume tailoring pipeline.

This module defines the top-level configuration and command-line interface for
future implementation. The actual pipeline steps live in dedicated modules and
should be wired in here as they are built out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single resume tailoring run."""

    posting_path: Path
    skills_cache_path: Path = Path("data/skills.yaml")
    template_path: Path = Path("data/template.tex")
    output_tex_path: Path = Path("build/tailored_resume.tex")
    output_pdf_path: Path = Path("build/tailored_resume.pdf")


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
    return parser


def run_pipeline(config: PipelineConfig) -> None:
    """Run the resume tailoring pipeline.

    The pipeline implementation is intentionally not wired up yet. The repo now
    has explicit docs, schemas, and test fixtures that define the expected
    behavior before the implementation is filled in.
    """

    raise NotImplementedError(
        "Resume tailoring pipeline is not implemented yet. "
        "Follow docs/agent/SPEC.md and docs/agent/DEV_PLAN.md to build it."
    )


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
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
