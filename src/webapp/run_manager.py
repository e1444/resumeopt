"""In-memory run registry: triggers the pipeline in a background thread and tracks status."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from main import PipelineConfig, run_pipeline

UPLOADS_DIR = Path("build/webapp_uploads")


@dataclass
class RunRecord:
    run_id: str
    status: str  # "running" | "completed" | "failed"
    created_at: str
    run_root: Path
    error: Optional[str] = None
    current_stage: Optional[str] = None
    substage: Optional[str] = None
    substage_completed: Optional[int] = None
    substage_total: Optional[int] = None


class RunManager:
    """Tracks pipeline runs triggered via the web API.

    `pipeline_runner` defaults to `main.run_pipeline` but can be overridden
    (e.g. in tests) with a stub that doesn't make real LLM calls - the
    manager itself only cares about a callable
    `(PipelineConfig, on_stage=...) -> None` that raises on failure and
    (optionally) calls `on_stage(stage_name)` as it progresses, so
    `GET /api/runs/{id}` can expose real stage-by-stage progress instead of
    an indeterminate UI animation.
    """

    def __init__(self, pipeline_runner: Callable[..., None] = run_pipeline):
        self._pipeline_runner = pipeline_runner
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

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
        )
        with self._lock:
            self._runs[run_id] = record

        thread = threading.Thread(target=self._execute, args=(run_id, config), daemon=True)
        thread.start()
        return run_id

    def _execute(self, run_id: str, config: PipelineConfig) -> None:
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

        try:
            self._pipeline_runner(config, on_stage=on_stage, on_substage=on_substage)
            status = "completed"
            error = None
        except Exception:
            status = "failed"
            error = traceback.format_exc()

        with self._lock:
            record = self._runs[run_id]
            record.status = status
            record.error = error

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def all_runs(self) -> List[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda record: record.created_at, reverse=True)
