"""LaTeX rendering utilities for skills section generation and PDF output."""

from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from pypdf import PdfReader

from llm import LLMProvider
from llm.schemas import SKILL_GROUPING_JSON_SCHEMA


_FALLBACK_SECTION_ORDER = ["Languages", "ML & Data", "Tools"]
SKILLS_SECTION_START_MARKER = "SKILLS"
SKILLS_SECTION_END_MARKER = "EXPERIENCE"

MIN_DYNAMIC_SECTIONS = 2
MAX_DYNAMIC_SECTIONS = 4


def build_sectioned_skills(
    canonical_skills: Sequence[str],
    llm_provider: Optional[LLMProvider] = None,
    posting_context: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Group canonical skills into 2-4 posting-appropriate display sections.

    `posting_context` (optional) is a short free-text description of the
    role/domain (e.g. derived from the Stage 0 posting summary) used to
    tailor section names to the specific posting (e.g. 'Cloud & DevOps',
    'Security', 'Data & ML') rather than a fixed, generic set. Added
    2026-07-17 replacing the previous fixed 3-category scheme
    (Languages/ML & Data/Tools) per explicit user request that categories
    should depend on the posting and skills selected, not be hard-coded.

    Falls back to the fixed 3-category deterministic grouping
    (`_deterministic_group_skills`) when no `llm_provider` is given or the
    LLM call fails - this remains the offline/no-LLM behavior.
    """

    unique_skills = sorted({skill.strip() for skill in canonical_skills if skill.strip()})
    if not unique_skills:
        return {}

    if llm_provider is not None:
        llm_grouped = _llm_group_skills(unique_skills, llm_provider, posting_context)
        if llm_grouped is not None:
            return llm_grouped

    return _deterministic_group_skills(unique_skills)


def render_skills_lines(sectioned_skills: Dict[str, List[str]]) -> str:
    """Render grouped skills into LaTeX lines for the template placeholder.

    Iterates `sectioned_skills` in its own key order (dynamic section names,
    no fixed ordering) rather than a hard-coded section list.
    """

    lines: List[str] = []
    for section, skills in sectioned_skills.items():
        if not skills:
            continue
        escaped = [_escape_latex(capitalize_skill_name(skill)) for skill in skills]
        lines.append(f"\\textbf{{{_escape_latex(section)}}}: {', '.join(escaped)}")

    if not lines:
        return "\\textbf{Skills}: None extracted"

    return "\\\\\n".join(lines)



SKILLS_PLACEHOLDER_MARKER = "[INSERT SKILLS HERE]"


def inject_skills_into_template(template_text: str, skills_block: str) -> str:
    """Replace the skills placeholder in template text."""

    marker = SKILLS_PLACEHOLDER_MARKER
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
    """Render PDF from .tex using pdflatex, invoked directly (no shell)."""

    output_dir = output_tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        str(output_tex_path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True)

    if logs_dir is not None:
        (logs_dir / "pdflatex.command.log").write_text(shlex.join(args) + "\n", encoding="utf-8")
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


def validate_pdf(
    pdf_path: Path,
    max_pages: int = 1,
    max_skills_section_lines: int = 4,
) -> Dict[str, Any]:
    """Validate a rendered resume PDF: page count, readability, and skills section length.

    Returns a report following the project's status/notes/issues validation schema
    (`status` is one of pass/fail; `notes` and `issues` are lists).
    """

    notes: List[str] = []
    issues: List[Dict[str, Any]] = []

    if not pdf_path.exists():
        return {
            "status": "fail",
            "notes": notes,
            "issues": [{"type": "missing_pdf", "message": f"PDF not found at {pdf_path}"}],
            "page_count": 0,
            "skills_section_line_count": 0,
        }

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except Exception as exc:
        return {
            "status": "fail",
            "notes": notes,
            "issues": [{"type": "unreadable_pdf", "message": str(exc)}],
            "page_count": 0,
            "skills_section_line_count": 0,
        }

    if page_count == 0:
        issues.append({"type": "empty_pdf", "message": "PDF has zero pages"})
    elif page_count > max_pages:
        issues.append(
            {"type": "page_count_exceeded", "page_count": page_count, "max_pages": max_pages}
        )
    else:
        notes.append(f"Page count {page_count} is within target of {max_pages}")

    skills_section_line_count = 0
    try:
        first_page_text = reader.pages[0].extract_text() or "" if page_count else ""
    except Exception as exc:
        first_page_text = ""
        issues.append({"type": "text_extraction_failed", "message": str(exc)})

    if first_page_text:
        start_index = first_page_text.find(SKILLS_SECTION_START_MARKER)
        end_index = (
            first_page_text.find(SKILLS_SECTION_END_MARKER, start_index + 1)
            if start_index != -1
            else -1
        )
        if start_index == -1 or end_index == -1:
            issues.append(
                {
                    "type": "skills_section_not_found",
                    "message": (
                        f"Could not locate skills section between '{SKILLS_SECTION_START_MARKER}' "
                        f"and '{SKILLS_SECTION_END_MARKER}' markers"
                    ),
                }
            )
        else:
            section_text = first_page_text[start_index + len(SKILLS_SECTION_START_MARKER) : end_index]
            skills_section_line_count = len([line for line in section_text.splitlines() if line.strip()])
            if skills_section_line_count > max_skills_section_lines:
                issues.append(
                    {
                        "type": "skills_section_too_long",
                        "line_count": skills_section_line_count,
                        "max_lines": max_skills_section_lines,
                    }
                )
            else:
                notes.append(
                    f"Skills section line count {skills_section_line_count} is within "
                    f"target of {max_skills_section_lines}"
                )

    status = "fail" if issues else "pass"
    return {
        "status": status,
        "notes": notes,
        "issues": issues,
        "page_count": page_count,
        "skills_section_line_count": skills_section_line_count,
    }


def _llm_group_skills(
    canonical_skills: Sequence[str],
    llm_provider: LLMProvider,
    posting_context: Optional[str] = None,
) -> Optional[Dict[str, List[str]]]:
    context_section = f"Posting context:\n{posting_context}\n\n" if posting_context else ""
    prompt = (
        f"Task: propose {MIN_DYNAMIC_SECTIONS} to {MAX_DYNAMIC_SECTIONS} concise, resume-appropriate "
        "section names TAILORED to the specific job posting and skill list below, then assign every "
        "skill to exactly one section.\n"
        "Guidelines:\n"
        f"- Prefer FEWER sections ({MIN_DYNAMIC_SECTIONS}) for a small or narrow skill set; use more "
        f"(up to {MAX_DYNAMIC_SECTIONS}) only when the skills are genuinely diverse enough to warrant "
        "separate categories.\n"
        "- Section names should be short (1-3 words), professional, and specific to what's actually "
        "being grouped (e.g. 'Cloud & DevOps', 'Data & ML', 'Security', 'Languages') - avoid vague "
        "catch-alls like 'Other' or 'Miscellaneous'.\n"
        "- Every skill in the list must be assigned to exactly one section - do not omit or invent "
        "skills, and do not use a skill more than once.\n"
        "- Order sections from most to least central to the role.\n\n"
        f"{context_section}"
        f"Skills: {list(canonical_skills)}"
    )

    try:
        payload = llm_provider.call_json(
            prompt=prompt,
            system_prompt=(
                "You are a strict resume formatter. Return valid JSON only and do not invent skills."
            ),
            temperature=0.1,
            max_tokens=600,
            json_schema=SKILL_GROUPING_JSON_SCHEMA,
        )
    except Exception:
        return None

    normalized_input = {skill.lower(): skill for skill in canonical_skills}
    raw_sections = payload.get("sections", [])
    if not isinstance(raw_sections, list) or not raw_sections:
        return None

    grouped: Dict[str, List[str]] = {}
    seen: set[str] = set()
    for entry in raw_sections:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        skills_list = entry.get("skills", [])
        if not isinstance(skills_list, list):
            continue
        bucket = grouped.setdefault(name, [])
        for raw in skills_list:
            candidate = str(raw).strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered not in normalized_input or lowered in seen:
                continue
            bucket.append(normalized_input[lowered])
            seen.add(lowered)

    # Drop any section the model proposed but left empty after validation.
    grouped = {name: skills for name, skills in grouped.items() if skills}
    if not grouped:
        return None

    # Clamp to at most MAX_DYNAMIC_SECTIONS - merge the smallest extra
    # sections into the largest remaining one rather than silently dropping
    # skills, if the model returned more sections than requested.
    if len(grouped) > MAX_DYNAMIC_SECTIONS:
        ordered = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        kept = dict(ordered[: MAX_DYNAMIC_SECTIONS - 1])
        overflow_name, overflow_skills = ordered[MAX_DYNAMIC_SECTIONS - 1][0], []
        for name, skills in ordered[MAX_DYNAMIC_SECTIONS - 1 :]:
            overflow_skills.extend(skills)
        kept[overflow_name] = overflow_skills
        grouped = kept

    # Preserve skill coverage even when the LLM omits assignments - add any
    # missed skill to the largest section rather than inventing a new
    # single-skill category for it.
    largest_section = max(grouped, key=lambda name: len(grouped[name])) if grouped else None
    for skill in canonical_skills:
        lowered = skill.lower()
        if lowered in seen:
            continue
        if largest_section is None:
            largest_section = "Other"
            grouped[largest_section] = []
        grouped[largest_section].append(skill)
        seen.add(lowered)

    return grouped


def _deterministic_group_skills(canonical_skills: Sequence[str]) -> Dict[str, List[str]]:
    """No-LLM fallback: the original fixed 3-category scheme
    (Languages/ML & Data/Tools), used only when no LLM provider is given or
    the dynamic LLM grouping call fails."""

    grouped = {section: [] for section in _FALLBACK_SECTION_ORDER}
    for skill in canonical_skills:
        grouped[_fallback_section_for_skill(skill)].append(skill)
    return {section: skills for section, skills in grouped.items() if skills}


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


def capitalize_skill_name(name: str) -> str:
    """Apply display-friendly capitalization to a canonical skill name.

    Per-word: if a word already contains an uppercase letter anywhere
    (e.g. "PostgreSQL", "JavaScript", "iOS"), it's trusted verbatim and left
    alone - re-capitalizing it naively (the old `_display_skill_name`
    behavior) would mangle stylized/branded names and acronyms (e.g. the
    real bug this fixed: "sql" stored in the cache rendered as "Sql" instead
    of "SQL"). Otherwise (a plain lowercase/uppercase word with no internal
    capitalization), applies standard title-casing via `str.capitalize()`.

    This function is the single source of truth for skill-name
    capitalization, applied both when a new/edited canonical name is stored
    in `data/skills.yaml` (see `webapp.skills_cache_io`) and, defensively, at
    render time (see `render_skills_lines` below) - so already-correctly-cased
    names (freshly stored or hand-edited in the YAML directly, e.g. "SQL")
    pass through unchanged (idempotent), while any not-yet-migrated legacy
    entries still get a reasonable display fallback.
    """

    words = []
    for word in name.split():
        if any(char.isupper() for char in word):
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)

