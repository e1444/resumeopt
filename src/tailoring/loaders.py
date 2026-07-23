"""YAML loaders for baseline tailoring source data (Phase 0).

Pure I/O + shape mapping into the typed models in `tailoring.models` - no
validation logic lives here (see `tailoring.validation`), so a loader can be
reused to load intentionally-invalid fixtures for validation tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from tailoring.models import BaselineBullet, FactAtom, ProjectBaseline, ResumeManifest


def load_fact_atoms(path: Path) -> List[FactAtom]:
    """Load `<project>_fact_atoms.yaml` into a list of `FactAtom`.

    Tolerates a missing/empty `fact_atoms` key (returns an empty list) so
    callers can validate "file exists but is empty" as its own case.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("fact_atoms") or []
    atoms: List[FactAtom] = []
    for entry in entries:
        atoms.append(
            FactAtom(
                id=entry["id"],
                fact=entry["fact"],
                skill_tags=tuple(entry.get("skill_tags") or ()),
                rationale=entry.get("rationale"),
            )
        )
    return atoms


def load_project_baseline(path: Path) -> ProjectBaseline:
    """Load `<project>_bullets.yaml` into a `ProjectBaseline`.

    Each bullet's `project_id` is read from its own entry if present,
    defaulting to the file's top-level `project_id` otherwise - this is
    deliberate: it lets a bullet entry declare a MISMATCHED project_id (an
    invalid-source-data case `tailoring.validation.validate_baseline_bullets`
    must catch), rather than the loader silently normalizing every bullet to
    the file's own project_id regardless of what the source data says.

    Does NOT cross-validate `fact_ids` against a fact-atoms file - that is
    `tailoring.validation.validate_baseline_bullets`'s job, since a caller
    may want to load bullets referencing facts from a different (e.g. test
    fixture) fact-atoms source.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    file_project_id = raw.get("project_id", "")
    bullets: List[BaselineBullet] = []
    for entry in raw.get("bullets") or []:
        bullets.append(
            BaselineBullet(
                id=entry["id"],
                project_id=entry.get("project_id", file_project_id),
                order=entry.get("order", 0),
                text=entry.get("text", ""),
                position=entry.get("position", ""),
                fact_ids=tuple(entry.get("fact_ids") or ()),
            )
        )
    return ProjectBaseline(
        project_id=raw.get("project_id", ""),
        project_title=raw.get("project_title", ""),
        role_context=raw.get("role_context", ""),
        dates=raw.get("dates", ""),
        resume_section=raw.get("resume_section", ""),
        bullets=tuple(bullets),
        project_summary=raw.get("project_summary", ""),
    )


def load_resume_manifest(experience_dir: Path) -> ResumeManifest:
    """Load `resume_manifest.yaml` plus every referenced project's bullets file."""

    manifest_path = experience_dir / "resume_manifest.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    projects: List[ProjectBaseline] = []
    for entry in sorted(raw.get("projects") or [], key=lambda item: item.get("order", 0)):
        project_id = entry["project_id"]
        bullets_path = experience_dir / project_id / f"{project_id}_bullets.yaml"
        projects.append(load_project_baseline(bullets_path))
    return ResumeManifest(
        source_template_path=raw.get("source_template_path", ""),
        projects=tuple(projects),
    )
