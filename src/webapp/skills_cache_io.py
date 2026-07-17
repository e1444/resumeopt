"""Read/write helpers for `data/skills.yaml`, shared by the webapp's skills CRUD endpoints.

Every write goes through `parser.load_skill_cache` for validation (unique
canonical names, `aliases` must be a list of strings) before being committed
- form input is never written straight to the real cache file - and the
previous file content is backed up first, so a bad edit is recoverable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from parser import load_skill_cache
from render_resume import capitalize_skill_name

DEFAULT_BACKUP_DIR = Path("build/cache_history")


class SkillCacheError(ValueError):
    """Raised when a skills-cache read/write operation is invalid."""


def list_skills(skills_cache_path: Path) -> List[Dict[str, Any]]:
    """Return every cache entry as a plain dict:
    `{"name": ..., "aliases": [...], "always_include": bool}`."""

    if not Path(skills_cache_path).exists():
        return []
    skills = load_skill_cache(skills_cache_path)
    return [
        {"name": record.name, "aliases": list(record.aliases), "always_include": record.always_include}
        for record in skills
    ]


def _backup(skills_cache_path: Path, backup_dir: Path) -> None:
    if not skills_cache_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    backup_path = backup_dir / f"{skills_cache_path.stem}_{timestamp}.yaml"
    backup_path.write_text(skills_cache_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_validated(skills_cache_path: Path, payload: List[Dict[str, Any]], backup_dir: Path) -> None:
    """Write `payload` to a temp file, validate it, then commit.

    The temp file is only promoted to the real cache path if
    `load_skill_cache` accepts it - this guarantees the real file never ends
    up in a broken/duplicate-name state from a bad edit.
    """

    skills_cache_path = Path(skills_cache_path)
    temp_path = skills_cache_path.with_suffix(".tmp.yaml")
    temp_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    try:
        load_skill_cache(temp_path)
    except ValueError as exc:
        temp_path.unlink(missing_ok=True)
        raise SkillCacheError(str(exc)) from exc

    _backup(skills_cache_path, backup_dir)
    temp_path.replace(skills_cache_path)


def _to_write_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Trim an in-memory skill entry dict (as returned by `list_skills`) back
    down to the minimal YAML shape before writing: omit `aliases` when empty,
    omit `always_include` when `False` - keeps `data/skills.yaml` clean/
    diffable rather than padding every entry with default-valued keys."""

    write_entry: Dict[str, Any] = {"name": entry["name"]}
    if entry.get("aliases"):
        write_entry["aliases"] = list(entry["aliases"])
    if entry.get("always_include"):
        write_entry["always_include"] = True
    return write_entry


def add_skill(
    skills_cache_path: Path,
    name: str,
    aliases: Optional[List[str]] = None,
    backup_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Add a new canonical skill entry. Raises `SkillCacheError` on a duplicate name.

    `name` is passed through `capitalize_skill_name` before storing (e.g.
    "data visualization" -> "Data Visualization", "PostgreSQL" -> unchanged)
    so canonical names are stored display-ready rather than relying on
    render-time re-capitalization to guess the right casing.
    """

    if backup_dir is None:
        backup_dir = DEFAULT_BACKUP_DIR

    name = capitalize_skill_name(name.strip())
    if not name:
        raise SkillCacheError("Skill name must not be empty")

    current = list_skills(skills_cache_path)
    if any(entry["name"].strip().lower() == name.lower() for entry in current):
        raise SkillCacheError(f"Duplicate canonical skill name: {name}")

    entry: Dict[str, Any] = {"name": name}
    cleaned_aliases = [alias.strip() for alias in (aliases or []) if alias and alias.strip()]
    if cleaned_aliases:
        entry["aliases"] = cleaned_aliases

    payload = [_to_write_entry(existing) for existing in current] + [entry]
    _write_validated(skills_cache_path, payload, backup_dir)
    return list_skills(skills_cache_path)


def remove_skill(
    skills_cache_path: Path,
    name: str,
    backup_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Remove a canonical skill entry by name (case-insensitive). Raises `SkillCacheError` if not found."""

    if backup_dir is None:
        backup_dir = DEFAULT_BACKUP_DIR

    current = list_skills(skills_cache_path)
    lowered = name.strip().lower()
    remaining = [entry for entry in current if entry["name"].strip().lower() != lowered]
    if len(remaining) == len(current):
        raise SkillCacheError(f"Skill not found: {name}")

    payload = [_to_write_entry(entry) for entry in remaining]
    _write_validated(skills_cache_path, payload, backup_dir)
    return list_skills(skills_cache_path)


def update_skill(
    skills_cache_path: Path,
    name: str,
    aliases: Optional[List[str]] = None,
    always_include: Optional[bool] = None,
    backup_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Update an existing skill's aliases and/or "always include" flag.

    `name` matches the existing canonical name case-insensitively (this
    never renames a skill - remove + add for that). Only the fields
    explicitly passed (not `None`) are changed; omitted fields keep their
    current value. Raises `SkillCacheError` if no matching skill exists.
    """

    if backup_dir is None:
        backup_dir = DEFAULT_BACKUP_DIR

    current = list_skills(skills_cache_path)
    lowered = name.strip().lower()
    found = False
    payload: List[Dict[str, Any]] = []
    for entry in current:
        if entry["name"].strip().lower() == lowered:
            found = True
            if aliases is not None:
                entry["aliases"] = [alias.strip() for alias in aliases if alias and alias.strip()]
            if always_include is not None:
                entry["always_include"] = bool(always_include)
        payload.append(_to_write_entry(entry))

    if not found:
        raise SkillCacheError(f"Skill not found: {name}")

    _write_validated(skills_cache_path, payload, backup_dir)
    return list_skills(skills_cache_path)


def promote_missing_skill(
    skills_cache_path: Path,
    term: str,
    aliases: Optional[List[str]] = None,
    backup_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Promote a missing-skill candidate term into the cache as a new canonical entry.

    Thin wrapper over `add_skill` - kept as its own function since the
    frontend's "promote to cache" action (see `FRONTEND_DEV_PLAN.md`) is
    conceptually distinct from the generic add-skill form, even though the
    underlying write path (and its validation/backup guarantees) is shared.
    """

    return add_skill(skills_cache_path, name=term, aliases=aliases, backup_dir=backup_dir)
