"""Read/write helpers for the LaTeX resume template used by the webapp.

Validates that an uploaded/edited template still contains the skills
placeholder marker before accepting it, mirroring
`render_resume.inject_skills_into_template`'s own marker check - reusing the
same constant (`SKILLS_PLACEHOLDER_MARKER`) rather than duplicating the
literal string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from render_resume import SKILLS_PLACEHOLDER_MARKER

DEFAULT_BACKUP_DIR = Path("build/template_history")


class TemplateError(ValueError):
    """Raised when a template read/write operation is invalid."""


def get_template(template_path: Path) -> str:
    template_path = Path(template_path)
    if not template_path.exists():
        raise TemplateError(f"Template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _backup(template_path: Path, backup_dir: Path) -> None:
    if not template_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    backup_path = backup_dir / f"{template_path.stem}_{timestamp}{template_path.suffix}"
    backup_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")


def save_template(template_path: Path, content: str, backup_dir: Optional[Path] = None) -> None:
    """Replace the template, after validating the skills placeholder is present.

    Backs up the previous template content first, so a bad upload is
    recoverable.
    """

    if backup_dir is None:
        backup_dir = DEFAULT_BACKUP_DIR

    if SKILLS_PLACEHOLDER_MARKER not in content:
        raise TemplateError(
            f"Template must contain the placeholder '{SKILLS_PLACEHOLDER_MARKER}'"
        )

    template_path = Path(template_path)
    _backup(template_path, backup_dir)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(content, encoding="utf-8")
