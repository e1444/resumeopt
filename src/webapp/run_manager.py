"""Run registry: triggers the pipeline in a background thread, tracks status,
and persists a lightweight index to disk so run history survives a backend
restart (only run_id/status/created_at/error - full details like metrics,
the PDF, and the posting text are never duplicated into the index; they stay
on disk under `build/<run_id>/` and are only read when a specific run is
selected, matching how `GET /api/runs/{id}` already lazily reads
`run_metrics.json`).

Since Phase 9 (see `docs/agent/FRONTEND_DEV_PLAN.md`), a run is no longer one
uninterrupted background thread from start to finish - it's split into two
separately-invokable pipeline phases (`main.run_pipeline_to_review` /
`main.run_pipeline_from_review`), with a human-in-the-loop skill-review
checkpoint (`"awaiting_review"` status) in between. This is the "simpler
two-pipeline design" per explicit user decision, rather than a single
function blocking mid-execution on a `threading.Event`."""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from main import PipelineConfig, run_pipeline_from_review, run_pipeline_to_review
from webapp import skills_cache_io

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("build/webapp_uploads")
DEFAULT_INDEX_PATH = Path("build/webapp_runs_index.json")


@dataclass
class RunRecord:
    run_id: str
    status: str  # "running" | "awaiting_review" | "completed" | "failed"
    created_at: str
    run_root: Path
    error: Optional[str] = None
    current_stage: Optional[str] = None
    substage: Optional[str] = None
    substage_completed: Optional[int] = None
    substage_total: Optional[int] = None
    # In-memory only - NOT persisted to the index (see `_persist_index`).
    # Needed so `confirm_skills` can resume a run without re-deriving its
    # config; reconstructed from `run_config.json` on disk if a restart
    # cleared it (see `_load_config_for_run`).
    config: Optional[PipelineConfig] = field(default=None, repr=False, compare=False)


class RunManager:
    """Tracks pipeline runs triggered via the web API.

    `to_review_runner`/`from_review_runner` default to
    `main.run_pipeline_to_review`/`main.run_pipeline_from_review` but can be
    overridden (e.g. in tests) with stubs that don't make real LLM calls.
    `to_review_runner` is `(PipelineConfig, on_stage=..., on_substage=...) ->
    dict` (the reviewable-skill payload); `from_review_runner` is
    `(PipelineConfig, included_skills, on_stage=...) -> None`. Both raise on
    failure and (optionally) call `on_stage(stage_name)` as they progress, so
    `GET /api/runs/{id}` can expose real stage-by-stage progress instead of
    an indeterminate UI animation.
    """

    def __init__(
        self,
        to_review_runner: Callable[..., dict] = run_pipeline_to_review,
        from_review_runner: Callable[..., None] = run_pipeline_from_review,
        index_path: Optional[Path] = None,
    ):
        self._to_review_runner = to_review_runner
        self._from_review_runner = from_review_runner
        # Resolved here (not as a `= DEFAULT_INDEX_PATH` default argument) so
        # tests that monkeypatch the module-level constant before
        # constructing a `RunManager` still take effect - a default argument
        # value is bound once at function-definition time, not at call time.
        self._index_path = index_path if index_path is not None else DEFAULT_INDEX_PATH
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self) -> None:
        """Restores run history from the persisted index on startup.

        A run still marked `"running"` in the index was orphaned by a
        previous process shutting down mid-phase (no background thread
        survives a restart to finish it), so it's surfaced as failed instead
        of stuck "running" forever in the UI. A run marked
        `"awaiting_review"`, however, IS safely resumable - `Stage 1
        (run_pipeline_to_review)` already completed and wrote its output
        durably to disk (including `skill_review.json`), so it's restored
        as-is rather than treated as interrupted."""

        if not self._index_path.exists():
            return
        try:
            entries = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("Failed to load run index from %s", self._index_path)
            return

        for entry in entries:
            run_id = entry.get("run_id")
            if not run_id:
                continue
            status = entry.get("status", "failed")
            error = entry.get("error")
            if status == "running":
                status = "failed"
                error = error or "Run interrupted: the backend restarted before this run finished."
            self._runs[run_id] = RunRecord(
                run_id=run_id,
                status=status,
                created_at=entry.get("created_at", ""),
                run_root=Path("build") / run_id,
                error=error,
            )

    def _persist_index(self) -> None:
        """Best-effort write of the lightweight index - never raises, since a
        failed write here should not affect an in-progress or just-finished
        pipeline run."""

        with self._lock:
            entries = [
                {
                    "run_id": record.run_id,
                    "status": record.status,
                    "created_at": record.created_at,
                    "error": record.error,
                }
                for record in self._runs.values()
            ]
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to persist run index to %s", self._index_path)

    def start_run(
        self,
        posting_text: str,
        skills_cache_path: Path = Path("data/skills.yaml"),
        template_path: Path = Path("data/template.tex"),
        **config_overrides: Any,
    ) -> str:
        run_id = f"webapp_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}"

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        posting_path = UPLOADS_DIR / f"{run_id}.txt"
        posting_path.write_text(posting_text, encoding="utf-8")

        config = PipelineConfig(
            posting_path=posting_path,
            skills_cache_path=Path(skills_cache_path),
            template_path=Path(template_path),
            run_name=run_id,
            **config_overrides,
        )

        record = RunRecord(
            run_id=run_id,
            status="running",
            created_at=datetime.now(UTC).isoformat(),
            run_root=Path("build") / run_id,
            config=config,
        )
        with self._lock:
            self._runs[run_id] = record
        self._persist_index()

        thread = threading.Thread(target=self._execute_to_review, args=(run_id, config), daemon=True)
        thread.start()
        return run_id

    def _execute_to_review(self, run_id: str, config: PipelineConfig) -> None:
        on_stage, on_substage = self._make_progress_callbacks(run_id)

        try:
            self._to_review_runner(config, on_stage=on_stage, on_substage=on_substage)
            status = "awaiting_review"
            error = None
        except Exception:
            status = "failed"
            error = traceback.format_exc()

        with self._lock:
            record = self._runs[run_id]
            record.status = status
            record.error = error
            record.current_stage = None
            record.substage = None
            record.substage_completed = None
            record.substage_total = None
        self._persist_index()

    def confirm_skills(self, run_id: str, included_skills: List[str]) -> None:
        """The Phase 9 checkpoint: the user has reviewed `skill_review.json`
        (see `main._build_skill_review_payload`) and confirmed their final
        skill list. Any of `included_skills` not already in the skills cache
        is promoted into it first (through the same validated
        `skills_cache_io.add_skill` write path manual promotion already
        uses - a backup is kept, duplicates are rejected), THEN the run
        resumes via `run_pipeline_from_review` in a new background thread.

        Callable more than once for the same run (per the "allow
        rerendering" decision) as long as it isn't currently `"running"` -
        going back and re-confirming a different selection re-runs Stage 8
        and overwrites the previous render.
        """

        record = self.get(run_id)
        if record is None:
            raise ValueError(f"Run not found: {run_id}")
        if record.status == "running":
            raise ValueError("Run is currently in progress - wait for it to finish before confirming again.")

        config = record.config or self._load_config_for_run(run_id)
        if config is None:
            raise ValueError(f"Run configuration not available for: {run_id}")

        review_path = config_run_root(run_id) / "logs" / "skill_review.json"
        if not review_path.exists():
            raise ValueError(f"No skill review available yet for run: {run_id}")

        cleaned_skills = [str(name).strip() for name in included_skills if str(name).strip()]
        self._promote_new_skills(config, cleaned_skills)

        with self._lock:
            record.status = "running"
            record.error = None
            record.config = config
        self._persist_index()

        thread = threading.Thread(
            target=self._execute_from_review, args=(run_id, config, cleaned_skills), daemon=True
        )
        thread.start()

    def _promote_new_skills(self, config: PipelineConfig, included_skills: List[str]) -> None:
        try:
            from parser import load_skill_cache

            existing_lower = {skill.name.lower() for skill in load_skill_cache(config.skills_cache_path)}
        except Exception:
            logger.exception("Failed to load skill cache while promoting new skills for confirm-skills")
            existing_lower = set()

        for name in included_skills:
            if name.lower() in existing_lower:
                continue
            try:
                skills_cache_io.add_skill(config.skills_cache_path, name, aliases=[])
                existing_lower.add(name.lower())
            except skills_cache_io.SkillCacheError:
                logger.exception("Failed to promote skill %r into the cache during confirm-skills", name)

    def _execute_from_review(self, run_id: str, config: PipelineConfig, included_skills: List[str]) -> None:
        on_stage, _ = self._make_progress_callbacks(run_id)

        try:
            self._from_review_runner(config, included_skills, on_stage=on_stage)
            status = "completed"
            error = None
        except Exception:
            status = "failed"
            error = traceback.format_exc()

        with self._lock:
            record = self._runs[run_id]
            record.status = status
            record.error = error
            record.current_stage = None
            record.substage = None
            record.substage_completed = None
            record.substage_total = None
        self._persist_index()

    def _make_progress_callbacks(self, run_id: str):
        def on_stage(stage: str) -> None:
            with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record.current_stage = stage
                    # A newly-entered coarse stage has no substage progress
                    # yet - clear stale substage info from the previous stage.
                    record.substage = None
                    record.substage_completed = None
                    record.substage_total = None

        def on_substage(name: str, completed: int, total: int) -> None:
            with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record.substage = name
                    record.substage_completed = completed
                    record.substage_total = total

        return on_stage, on_substage

    def _load_config_for_run(self, run_id: str) -> Optional[PipelineConfig]:
        """Reconstructs a `PipelineConfig` from the `run_config.json` a run
        already wrote at Stage 1 start - needed when `confirm_skills` is
        called for a run whose in-memory `RunRecord.config` was cleared by a
        backend restart (see `_load_index`'s "awaiting_review" handling)."""

        config_path = Path("build") / run_id / "logs" / "run_config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("Failed to read run_config.json for run=%s", run_id)
            return None

        return PipelineConfig(
            posting_path=Path(payload["posting_path"]),
            skills_cache_path=Path(payload.get("skills_cache_path", "data/skills.yaml")),
            template_path=Path(payload.get("template_path", "data/template.tex")),
            llm_provider=payload.get("llm_provider", "openai"),
            llm_model=payload.get("llm_model", "gpt-4o"),
            reasoning_llm_model=payload.get("reasoning_llm_model", "gpt-5-mini"),
            screening_llm_model=payload.get("screening_llm_model", "gpt-4o-mini"),
            use_llm_parser=payload.get("use_llm_parser", True),
            max_concurrency=payload.get("max_concurrency", 24),
            run_name=payload.get("run_name", run_id),
        )

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def all_runs(self) -> List[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda record: record.created_at, reverse=True)


def config_run_root(run_id: str) -> Path:
    """The deterministic `build/<run_id>/` root for a run - a tiny helper so
    `RunManager.confirm_skills` doesn't need a live `RunRecord.run_root` to
    check for `skill_review.json`."""

    return Path("build") / run_id

