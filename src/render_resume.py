"""LaTeX rendering utilities for skills section generation and PDF output."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

from llm import LLMProvider
from llm.schemas import SKILL_GROUPING_JSON_SCHEMA


SECTION_ORDER = ["Languages", "ML & Data", "Tools"]


def build_sectioned_skills(
    canonical_skills: Sequence[str],
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, List[str]]:
    """Group canonical skills into three display sections."""

    unique_skills = sorted({skill.strip() for skill in canonical_skills if skill.strip()})
    if not unique_skills:
        return {section: [] for section in SECTION_ORDER}

    if llm_provider is not None:
        llm_grouped = _llm_group_skills(unique_skills, llm_provider)
        if llm_grouped is not None:
            return llm_grouped

    return _deterministic_group_skills(unique_skills)


def render_skills_lines(sectioned_skills: Dict[str, List[str]]) -> str:
    """Render grouped skills into LaTeX lines for the template placeholder."""

    lines: List[str] = []
    for section in SECTION_ORDER:
        skills = sectioned_skills.get(section, [])
        if not skills:
            continue
        escaped = [_escape_latex(_display_skill_name(skill)) for skill in skills]
        lines.append(f"\\textbf{{{_escape_latex(section)}}}: {', '.join(escaped)}")

    if not lines:
        return "\\textbf{Skills}: None extracted"

    return "\\\\\n".join(lines)


def inject_skills_into_template(template_text: str, skills_block: str) -> str:
    """Replace the skills placeholder in template text."""

    marker = "[INSERT SKILLS HERE]"
    if marker not in template_text:
        raise ValueError(f"Template placeholder '{marker}' not found")
    return template_text.replace(marker, skills_block)


def write_tex_from_template(template_path: Path, output_tex_path: Path, skills_block: str) -> Path:
    """Create a rendered .tex file from the template and skills content."""

    template_text = template_path.read_text(encoding="utf-8")
    rendered_text = inject_skills_into_template(template_text, skills_block)

    output_tex_path.parent.mkdir(parents=True, exist_ok=True)
    output_tex_path.write_text(rendered_text, encoding="utf-8")
    return output_tex_path


def render_pdf_with_pdflatex(
    output_tex_path: Path,
    output_pdf_path: Path,
    logs_dir: Optional[Path] = None,
) -> Path:
    """Render PDF from .tex using a bash pdflatex command."""

    output_dir = output_tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    command = (
        "pdflatex -interaction=nonstopmode -halt-on-error "
        f"-output-directory '{output_dir}' '{output_tex_path}'"
    )
    completed = subprocess.run(["bash", "-lc", command], capture_output=True, text=True)

    if logs_dir is not None:
        (logs_dir / "pdflatex.command.log").write_text(command + "\n", encoding="utf-8")
        (logs_dir / "pdflatex.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs_dir / "pdflatex.stderr.log").write_text(completed.stderr, encoding="utf-8")

        engine_log_path = output_tex_path.with_suffix(".log")
        if engine_log_path.exists():
            shutil.copy2(engine_log_path, logs_dir / "pdflatex.engine.log")

    if completed.returncode != 0:
        raise RuntimeError(
            "pdflatex failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    rendered_pdf = output_tex_path.with_suffix(".pdf")
    if not rendered_pdf.exists():
        raise RuntimeError("pdflatex completed but PDF was not generated")

    if rendered_pdf != output_pdf_path:
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        output_pdf_path.write_bytes(rendered_pdf.read_bytes())

    return output_pdf_path


def _llm_group_skills(canonical_skills: Sequence[str], llm_provider: LLMProvider) -> Optional[Dict[str, List[str]]]:
    prompt = (
        "Group the following canonical skills into resume sections. "
        "You may include only relevant sections among Languages, ML & Data, Tools. "
        f"\n\nSkills: {list(canonical_skills)}\n"
        "Use each provided skill at most once. Omit unclear skills."
        "If a section is not relevant for the target role, exclude it from active_sections."
    )

    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt=(
                "You are a strict resume formatter. Return valid JSON only and do not invent skills."
            ),
            temperature=0.1,
            max_tokens=500,
            json_schema=SKILL_GROUPING_JSON_SCHEMA,
        )
    except Exception:
        return None

    normalized_input = {skill.lower(): skill for skill in canonical_skills}
    active_sections = payload.get("active_sections", [])
    if isinstance(active_sections, list):
        ordered_active_sections = [
            section for section in SECTION_ORDER if section in {str(item).strip() for item in active_sections}
        ]
    else:
        ordered_active_sections = []

    if not ordered_active_sections:
        ordered_active_sections = SECTION_ORDER.copy()

    grouped = {section: [] for section in ordered_active_sections}
    seen: set[str] = set()

    grouped_payload = payload.get("grouped_skills", {})
    if not isinstance(grouped_payload, dict):
        grouped_payload = {}

    for section in ordered_active_sections:
        raw_values = grouped_payload.get(section, payload.get(section, []))
        if not isinstance(raw_values, list):
            continue
        for raw in raw_values:
            candidate = str(raw).strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered not in normalized_input or lowered in seen:
                continue
            grouped[section].append(normalized_input[lowered])
            seen.add(lowered)

    # Preserve skill coverage even when the LLM omits assignments.
    for skill in canonical_skills:
        lowered = skill.lower()
        if lowered in seen:
            continue

        fallback_section = _fallback_section_for_skill(skill)
        if fallback_section in grouped:
            grouped[fallback_section].append(skill)
        else:
            grouped[ordered_active_sections[0]].append(skill)

    return grouped


def _deterministic_group_skills(canonical_skills: Sequence[str]) -> Dict[str, List[str]]:
    grouped = {section: [] for section in SECTION_ORDER}
    for skill in canonical_skills:
        grouped[_fallback_section_for_skill(skill)].append(skill)
    return grouped


def _fallback_section_for_skill(skill: str) -> str:
    lowered = skill.lower()

    language_terms = {"python", "java", "c++", "c", "r", "sql", "bash", "javascript", "typescript"}
    ml_data_terms = {
        "pytorch",
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "machine learning",
        "statistics",
        "seaborn",
        "matplotlib",
        "jupyter",
    }

    if lowered in language_terms:
        return "Languages"
    if lowered in ml_data_terms:
        return "ML & Data"
    return "Tools"


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = value
    for original, escaped in replacements.items():
        result = result.replace(original, escaped)
    return result


def _display_skill_name(skill: str) -> str:
    """Convert canonical skill names to display-friendly capitalization."""

    return " ".join(part.capitalize() for part in skill.split())
