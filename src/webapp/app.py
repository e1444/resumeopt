"""FastAPI app exposing the resumeopt pipeline over HTTP.

See `docs/agent/FRONTEND_DEV_PLAN.md` for the design. Binds to 127.0.0.1 only
by default when run via `uvicorn` (see `if __name__ == "__main__"` below) -
this app can trigger billed LLM calls per request and should not be exposed
beyond localhost without deliberately adding authentication first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from main import PIPELINE_STAGES
from webapp import skills_cache_io, template_io
from webapp.run_manager import UPLOADS_DIR, RunManager

DEFAULT_SKILLS_CACHE_PATH = Path("data/skills.yaml")
DEFAULT_TEMPLATE_PATH = Path("data/template.tex")

# The Vite dev server normally proxies /api to this app (see
# frontend/vite.config.ts), so no cross-origin request should ever be
# needed - this is only a safety net for running the dev server without the
# proxy. Deliberately NOT a wildcard: this app can trigger billed LLM calls,
# so only the known local dev-server origins are allowed.
_DEV_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Explicit allow-list for log filenames served over HTTP: `log_name` comes
# from the URL, and the pipeline never writes attacker-controlled filenames,
# but this still guards against path-traversal (`../../etc/passwd`-style)
# requests before any filesystem join happens.
_ALLOWED_LOG_NAMES = {
    "parsed_records.json",
    "extraction_debug.json",
    "validation_report.json",
    "sectioned_skills.json",
    "pdf_validation.json",
    "run_metrics.json",
    "missing_skills.json",
    "missing_skills_candidates.json",
    "missing_skills_discarded.json",
    "run_config.json",
    "llm_call_log.json",
    "confirmed_skills.json",
}

app = FastAPI(title="resumeopt")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_FRONTEND_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["Content-Type"],
)
run_manager = RunManager()


class SkillIn(BaseModel):
    name: str
    aliases: List[str] = []


class SkillUpdateIn(BaseModel):
    """All fields optional - only fields explicitly set are changed (see
    `skills_cache_io.update_skill`)."""

    aliases: List[str] | None = None
    always_include: bool | None = None


class ConfirmSkillsIn(BaseModel):
    """Body for `POST /api/runs/{run_id}/confirm-skills` - the Phase 9
    human-in-the-loop skill-review checkpoint (see
    `docs/agent/FRONTEND_DEV_PLAN.md`). `included_skills` is the user's
    final, confirmed canonical/raw skill-name list (checked entries from
    `skill_review.json`, plus anything they added via the "other cache
    skills" escape hatch) - any name not already in the skills cache is
    promoted into it first."""

    included_skills: List[str]


class TemplateIn(BaseModel):
    content: str


class RunIn(BaseModel):
    posting_text: str
    provider: str = "openai"
    model: str = "gpt-4o"
    reasoning_model: str = "gpt-5-mini"
    screening_model: str = "gpt-4o-mini"
    use_llm_parser: bool = True
    max_concurrency: int = 24


@app.get("/api/skills")
def get_skills() -> List[Dict[str, Any]]:
    return skills_cache_io.list_skills(DEFAULT_SKILLS_CACHE_PATH)


@app.post("/api/skills")
def post_skill(skill: SkillIn) -> List[Dict[str, Any]]:
    try:
        return skills_cache_io.add_skill(DEFAULT_SKILLS_CACHE_PATH, skill.name, skill.aliases)
    except skills_cache_io.SkillCacheError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/skills/{name}")
def delete_skill(name: str) -> List[Dict[str, Any]]:
    try:
        return skills_cache_io.remove_skill(DEFAULT_SKILLS_CACHE_PATH, name)
    except skills_cache_io.SkillCacheError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/skills/{name}")
def patch_skill(name: str, update: SkillUpdateIn) -> List[Dict[str, Any]]:
    try:
        return skills_cache_io.update_skill(
            DEFAULT_SKILLS_CACHE_PATH, name, aliases=update.aliases, always_include=update.always_include
        )
    except skills_cache_io.SkillCacheError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/template", response_class=PlainTextResponse)
def get_template() -> str:
    try:
        return template_io.get_template(DEFAULT_TEMPLATE_PATH)
    except template_io.TemplateError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/template")
def post_template(payload: TemplateIn) -> Dict[str, str]:
    try:
        template_io.save_template(DEFAULT_TEMPLATE_PATH, payload.content)
    except template_io.TemplateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/runs")
def post_run(payload: RunIn) -> Dict[str, str]:
    run_id = run_manager.start_run(
        posting_text=payload.posting_text,
        skills_cache_path=DEFAULT_SKILLS_CACHE_PATH,
        template_path=DEFAULT_TEMPLATE_PATH,
        llm_provider=payload.provider,
        llm_model=payload.model,
        reasoning_llm_model=payload.reasoning_model,
        screening_llm_model=payload.screening_model,
        use_llm_parser=payload.use_llm_parser,
        max_concurrency=payload.max_concurrency,
    )
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    return [
        {"run_id": record.run_id, "status": record.status, "created_at": record.created_at}
        for record in run_manager.all_runs()
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    record = run_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    response: Dict[str, Any] = {
        "run_id": record.run_id,
        "status": record.status,
        "created_at": record.created_at,
    }
    if record.error:
        response["error"] = record.error

    if record.status == "running" and record.current_stage in PIPELINE_STAGES:
        stage_index = PIPELINE_STAGES.index(record.current_stage)
        response["current_stage"] = record.current_stage
        response["stage_index"] = stage_index
        response["stage_total"] = len(PIPELINE_STAGES)

        if record.substage is not None and record.substage_total:
            response["substage"] = record.substage
            response["substage_completed"] = record.substage_completed
            response["substage_total"] = record.substage_total

    # Exposed whenever it exists on disk (not gated to "awaiting_review") so
    # the frontend can also re-open the review UI for a completed/failed run
    # and re-confirm a different selection (the "allow rerendering" Phase 9
    # decision).
    review_path = record.run_root / "logs" / "skill_review.json"
    if review_path.exists():
        response["skill_review"] = json.loads(review_path.read_text(encoding="utf-8"))

    metrics_path = record.run_root / "logs" / "run_metrics.json"
    if metrics_path.exists():
        response["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return response


@app.post("/api/runs/{run_id}/confirm-skills")
def post_confirm_skills(run_id: str, payload: ConfirmSkillsIn) -> Dict[str, str]:
    try:
        run_manager.confirm_skills(run_id, payload.included_skills)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}/pdf")
def get_run_pdf(run_id: str) -> FileResponse:
    record = run_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    pdf_path = record.run_root / "tailored_resume.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not available yet")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{run_id}_output.pdf",
        # "inline" (not the default "attachment") so the iframe preview in
        # the UI still renders the PDF in place - the filename above only
        # kicks in when the user explicitly saves/downloads it (e.g. via the
        # browser's built-in PDF viewer download button), giving each run a
        # unique, collision-free suggested filename instead of every run
        # suggesting the same generic "tailored_resume.pdf".
        content_disposition_type="inline",
    )


@app.get("/api/runs/{run_id}/posting", response_class=PlainTextResponse)
def get_run_posting(run_id: str) -> str:
    """The raw job-posting text a run's skills were generated against -
    saved once per run at trigger time (`RunManager.start_run`), independent
    of the run's in-memory record, so it's available for historical runs
    even across a backend restart."""

    record = run_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    posting_path = UPLOADS_DIR / f"{run_id}.txt"
    if not posting_path.exists():
        raise HTTPException(status_code=404, detail="Posting text not available")
    return posting_path.read_text(encoding="utf-8")



@app.get("/api/runs/{run_id}/logs/{log_name}")
def get_run_log(run_id: str, log_name: str) -> Any:
    if log_name not in _ALLOWED_LOG_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown log name: {log_name}")

    record = run_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    log_path = record.run_root / "logs" / log_name
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not available")
    return json.loads(log_path.read_text(encoding="utf-8"))


@app.post("/api/runs/{run_id}/missing-skills/{term}/promote")
def promote_missing_skill(run_id: str, term: str) -> List[Dict[str, Any]]:
    record = run_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        return skills_cache_io.promote_missing_skill(DEFAULT_SKILLS_CACHE_PATH, term)
    except skills_cache_io.SkillCacheError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1-only by default - see the module docstring's security note.
    uvicorn.run(app, host="127.0.0.1", port=8000)
