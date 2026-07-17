"""FastAPI backend for the resumeopt frontend.

See `docs/agent/FRONTEND_DEV_PLAN.md` for the full design. This package only
wraps the existing pipeline (`src/parser`, `src/render_resume.py`,
`src/main.py`) behind HTTP endpoints - it does not change parsing, matching,
or rendering behavior.

Layout:
- skills_cache_io.py  read/write helpers for `data/skills.yaml`, reusing
  `parser.load_skill_cache`'s validation (unique names, `aliases` shape)
  rather than reimplementing it, plus a versioned backup-before-write.
- template_io.py       read/write helpers for the LaTeX template, validating
  the `[INSERT SKILLS HERE]` placeholder before accepting a new template.
- run_manager.py        in-memory run registry: triggers `main.run_pipeline`
  in a background thread, tracks status, and exposes artifact paths.
- app.py                the FastAPI app and its routes.
"""

from __future__ import annotations
