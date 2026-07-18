"""Tests for the webapp FastAPI backend (deterministic, no live LLM calls).

Uses a fake pipeline runner (writes the same artifact shapes `main.run_pipeline`
would, without calling any LLM) so run-trigger/status/artifact-retrieval
endpoints can be tested without network access or API keys.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi.testclient import TestClient

from webapp import app as app_module
from webapp import skills_cache_io, template_io
from webapp.run_manager import RunManager

from main import PIPELINE_STAGES


def _fake_to_review_runner(config, on_stage=None, on_substage=None) -> dict:
    """Writes the same Stage 0-7 artifact shapes `main.run_pipeline_to_review`
    would, instantly, and returns a reviewable-skill payload matching
    `main._build_skill_review_payload`'s shape."""

    for stage in ("read_posting", "init_llm_provider", "parse_posting", "validate_selected_skills"):
        if on_stage is not None:
            on_stage(stage)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Real run_pipeline_to_review always writes this - needed so
    # RunManager._load_config_for_run can reconstruct a PipelineConfig after
    # a simulated backend restart (see
    # test_a_run_awaiting_review_at_restart_stays_awaiting_review).
    (logs_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": config.run_name,
                "posting_path": str(config.posting_path),
                "skills_cache_path": str(config.skills_cache_path),
                "template_path": str(config.template_path),
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model,
                "reasoning_llm_model": config.reasoning_llm_model,
                "screening_llm_model": config.screening_llm_model,
                "use_llm_parser": config.use_llm_parser,
                "max_concurrency": config.max_concurrency,
            }
        ),
        encoding="utf-8",
    )

    (logs_dir / "run_metrics.json").write_text(
        json.dumps({"parse": {"matched_skill_count": 1}}), encoding="utf-8"
    )
    (logs_dir / "missing_skills.json").write_text(
        json.dumps({"missing_skills": ["kubernetes"], "count": 1}), encoding="utf-8"
    )
    review_payload = {
        "reviewable_skills": [
            {
                "name": "python",
                "source": "matched",
                "match_type": "exact",
                "confidence": 0.98,
                "evidence": "We need Python.",
                "low_confidence": False,
                "default_checked": True,
                "is_always_include": False,
            },
            {
                "name": "kubernetes",
                "source": "missing",
                "match_type": None,
                "confidence": None,
                "evidence": "We need Python and Kubernetes.",
                "low_confidence": False,
                "default_checked": False,
                "is_always_include": False,
            },
        ],
        "other_cache_skills": [],
    }
    (logs_dir / "skill_review.json").write_text(json.dumps(review_payload), encoding="utf-8")
    return review_payload


def _fake_from_review_runner(config, included_skills, on_stage=None) -> None:
    """Writes the same Stage 8+ artifact shapes `main.run_pipeline_from_review`
    would, instantly."""

    for stage in ("group_skills", "rendering", "finalizing"):
        if on_stage is not None:
            on_stage(stage)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    existing_metrics: dict = {}
    metrics_path = logs_dir / "run_metrics.json"
    if metrics_path.exists():
        existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing_metrics["skills_block"] = {"canonical_skill_count": len(included_skills)}
    metrics_path.write_text(json.dumps(existing_metrics), encoding="utf-8")

    (logs_dir / "confirmed_skills.json").write_text(
        json.dumps({"included_skills": included_skills, "final_skills": included_skills}), encoding="utf-8"
    )

    (run_root / "tailored_resume.pdf").write_bytes(b"%PDF-1.4 fake pdf content")


def _failing_to_review_runner(config, on_stage=None, on_substage=None) -> dict:
    raise RuntimeError("simulated pipeline failure")


def _slow_to_review_runner(config, on_stage=None, on_substage=None) -> dict:
    """Calls on_stage for the first stage only, then blocks - lets a test
    observe an in-flight run's current_stage before it reaches
    awaiting_review."""

    if on_stage is not None:
        on_stage(PIPELINE_STAGES[0])
    time.sleep(2)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_metrics.json").write_text(json.dumps({}), encoding="utf-8")
    review_payload = {"reviewable_skills": [], "other_cache_skills": []}
    (logs_dir / "skill_review.json").write_text(json.dumps(review_payload), encoding="utf-8")
    return review_payload


def _slow_with_substage_to_review_runner(config, on_stage=None, on_substage=None) -> dict:
    """Like `_slow_to_review_runner`, but also reports substage batch
    progress within the `parse_posting` stage."""

    if on_stage is not None:
        on_stage("parse_posting")
    if on_substage is not None:
        on_substage("extraction", 1, 3)
    time.sleep(2)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_metrics.json").write_text(json.dumps({}), encoding="utf-8")
    review_payload = {"reviewable_skills": [], "other_cache_skills": []}
    (logs_dir / "skill_review.json").write_text(json.dumps(review_payload), encoding="utf-8")
    return review_payload


class WebappApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

        self.skills_cache_path = Path(self.tmp_dir) / "skills.yaml"
        self.skills_cache_path.write_text(
            "- name: python\n  aliases:\n    - py\n", encoding="utf-8"
        )

        self.template_path = Path(self.tmp_dir) / "template.tex"
        self.template_path.write_text("before\n[INSERT SKILLS HERE]\nafter\n", encoding="utf-8")

        self.backup_dir = Path(self.tmp_dir) / "cache_history"
        self.template_backup_dir = Path(self.tmp_dir) / "template_history"
        self.index_path = Path(self.tmp_dir) / "runs_index.json"

        self._orig_skills_path = app_module.DEFAULT_SKILLS_CACHE_PATH
        self._orig_template_path = app_module.DEFAULT_TEMPLATE_PATH
        self._orig_run_manager = app_module.run_manager
        app_module.DEFAULT_SKILLS_CACHE_PATH = self.skills_cache_path
        app_module.DEFAULT_TEMPLATE_PATH = self.template_path
        app_module.run_manager = RunManager(
            to_review_runner=_fake_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        self._orig_backup_dir = skills_cache_io.DEFAULT_BACKUP_DIR
        self._orig_template_backup_dir = template_io.DEFAULT_BACKUP_DIR
        skills_cache_io.DEFAULT_BACKUP_DIR = self.backup_dir
        template_io.DEFAULT_BACKUP_DIR = self.template_backup_dir

        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        app_module.DEFAULT_SKILLS_CACHE_PATH = self._orig_skills_path
        app_module.DEFAULT_TEMPLATE_PATH = self._orig_template_path
        app_module.run_manager = self._orig_run_manager
        skills_cache_io.DEFAULT_BACKUP_DIR = self._orig_backup_dir
        template_io.DEFAULT_BACKUP_DIR = self._orig_template_backup_dir

        # RunManager writes real `build/webapp_*` run folders/uploads (it
        # mirrors main.py's real `build/<run_name>/` layout) - clean those up
        # so test runs don't accumulate on disk.
        for path in Path("build").glob("webapp_*"):
            shutil.rmtree(path, ignore_errors=True)
        uploads_dir = Path("build/webapp_uploads")
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir, ignore_errors=True)

    # --- skills CRUD ---

    def test_list_skills_returns_seeded_cache(self) -> None:
        response = self.client.get("/api/skills")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), [{"name": "python", "aliases": ["py"], "always_include": False}]
        )

    def test_add_skill_persists_to_cache_file(self) -> None:
        response = self.client.post("/api/skills", json={"name": "docker", "aliases": []})

        self.assertEqual(response.status_code, 200)
        # capitalize_skill_name is applied at storage time (see task 1's
        # display-bug fix) - a plain lowercase, no-internal-caps name like
        # docker is stored title-cased as Docker.
        names = {entry["name"] for entry in response.json()}
        self.assertIn("Docker", names)

        reloaded = self.client.get("/api/skills").json()
        self.assertIn("Docker", {entry["name"] for entry in reloaded})

    def test_add_duplicate_skill_returns_400(self) -> None:
        response = self.client.post("/api/skills", json={"name": "Python", "aliases": []})

        self.assertEqual(response.status_code, 400)

    def test_delete_skill_removes_entry(self) -> None:
        response = self.client.delete("/api/skills/python")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_delete_missing_skill_returns_404(self) -> None:
        response = self.client.delete("/api/skills/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_add_skill_backs_up_previous_cache_content(self) -> None:
        self.client.post("/api/skills", json={"name": "docker"})

        backups = list(self.backup_dir.glob("*.yaml"))
        self.assertEqual(len(backups), 1)
        self.assertIn("python", backups[0].read_text(encoding="utf-8"))

    def test_patch_skill_updates_always_include_flag(self) -> None:
        response = self.client.patch("/api/skills/python", json={"always_include": True})

        self.assertEqual(response.status_code, 200)
        entry = next(item for item in response.json() if item["name"] == "python")
        self.assertTrue(entry["always_include"])

        reloaded = self.client.get("/api/skills").json()
        self.assertTrue(reloaded[0]["always_include"])

    def test_patch_skill_updates_aliases(self) -> None:
        response = self.client.patch("/api/skills/python", json={"aliases": ["py", "python3"]})

        self.assertEqual(response.status_code, 200)
        entry = next(item for item in response.json() if item["name"] == "python")
        self.assertEqual(entry["aliases"], ["py", "python3"])

    def test_patch_missing_skill_returns_404(self) -> None:
        response = self.client.patch("/api/skills/does-not-exist", json={"always_include": True})

        self.assertEqual(response.status_code, 404)

    # --- template CRUD ---

    def test_get_template_returns_current_content(self) -> None:
        response = self.client.get("/api/template")

        self.assertEqual(response.status_code, 200)
        self.assertIn("[INSERT SKILLS HERE]", response.text)

    def test_post_template_replaces_content(self) -> None:
        new_content = "header\n[INSERT SKILLS HERE]\nfooter\n"

        response = self.client.post("/api/template", json={"content": new_content})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.template_path.read_text(encoding="utf-8"), new_content)

    def test_post_template_without_placeholder_returns_400(self) -> None:
        response = self.client.post("/api/template", json={"content": "no placeholder here"})

        self.assertEqual(response.status_code, 400)

    # --- run trigger / status / artifacts ---

    def test_post_run_pauses_at_awaiting_review(self) -> None:
        response = self.client.post("/api/runs", json={"posting_text": "We need Python."})
        self.assertEqual(response.status_code, 200)
        run_id = response.json()["run_id"]

        payload = self._wait_for_status(run_id, "awaiting_review")

        self.assertIn("skill_review", payload)
        names = {skill["name"] for skill in payload["skill_review"]["reviewable_skills"]}
        self.assertEqual(names, {"python", "kubernetes"})

    def test_confirm_skills_resumes_and_completes_the_run(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_status(run_id, "awaiting_review")

        response = self.client.post(f"/api/runs/{run_id}/confirm-skills", json={"included_skills": ["python"]})
        self.assertEqual(response.status_code, 200)

        payload = self._wait_for_status(run_id, "completed")
        self.assertIn("metrics", payload)

    def test_confirm_skills_promotes_a_new_skill_into_the_cache(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_status(run_id, "awaiting_review")

        response = self.client.post(
            f"/api/runs/{run_id}/confirm-skills", json={"included_skills": ["python", "kubernetes"]}
        )
        self.assertEqual(response.status_code, 200)
        self._wait_for_status(run_id, "completed")

        names = {entry["name"] for entry in self.client.get("/api/skills").json()}
        # Stored capitalized (Kubernetes) - see task 1's display-bug fix.
        self.assertIn("Kubernetes", names)

    def test_confirm_skills_for_unknown_run_returns_400(self) -> None:
        response = self.client.post(
            "/api/runs/does-not-exist/confirm-skills", json={"included_skills": ["python"]}
        )

        self.assertEqual(response.status_code, 400)

    def test_confirm_skills_allows_rerendering_a_completed_run(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.post(
            f"/api/runs/{run_id}/confirm-skills", json={"included_skills": ["python", "kubernetes"]}
        )
        self.assertEqual(response.status_code, 200)

        payload = self._wait_for_status(run_id, "completed")
        self.assertEqual(payload["metrics"]["skills_block"]["canonical_skill_count"], 2)

    def test_get_run_exposes_current_stage_while_running(self) -> None:
        app_module.run_manager = RunManager(
            to_review_runner=_slow_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]

        deadline = time.time() + 2
        payload: dict = {}
        while time.time() < deadline:
            payload = self.client.get(f"/api/runs/{run_id}").json()
            if payload.get("current_stage"):
                break
            time.sleep(0.02)

        self.assertEqual(payload.get("status"), "running")
        self.assertEqual(payload.get("current_stage"), PIPELINE_STAGES[0])
        self.assertEqual(payload.get("stage_index"), 0)
        self.assertEqual(payload.get("stage_total"), len(PIPELINE_STAGES))

        self._wait_for_status(run_id, "awaiting_review")

    def test_get_run_exposes_substage_progress_while_running(self) -> None:
        app_module.run_manager = RunManager(
            to_review_runner=_slow_with_substage_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]

        deadline = time.time() + 2
        payload: dict = {}
        while time.time() < deadline:
            payload = self.client.get(f"/api/runs/{run_id}").json()
            if payload.get("substage"):
                break
            time.sleep(0.02)

        self.assertEqual(payload.get("status"), "running")
        self.assertEqual(payload.get("current_stage"), "parse_posting")
        self.assertEqual(payload.get("substage"), "extraction")
        self.assertEqual(payload.get("substage_completed"), 1)
        self.assertEqual(payload.get("substage_total"), 3)

        self._wait_for_status(run_id, "awaiting_review")


    def test_get_run_pdf_available_after_completion(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.get(f"/api/runs/{run_id}/pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        # Each run's PDF suggests a unique, run-specific filename (instead of
        # every run suggesting the same generic "tailored_resume.pdf"), but
        # stays "inline" so the frontend's <iframe> preview still renders it.
        content_disposition = response.headers["content-disposition"]
        self.assertIn(f"{run_id}_output.pdf", content_disposition)
        self.assertTrue(content_disposition.startswith("inline"))

    def test_get_run_log_returns_allowed_log(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.get(f"/api/runs/{run_id}/logs/missing_skills.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["missing_skills"], ["kubernetes"])

    def test_get_run_log_returns_confirmed_skills(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.get(f"/api/runs/{run_id}/logs/confirmed_skills.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["included_skills"], ["python"])
        self.assertEqual(response.json()["final_skills"], ["python"])

    def test_get_run_log_rejects_unknown_log_name(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.get(f"/api/runs/{run_id}/logs/../../etc/passwd")

        self.assertIn(response.status_code, (400, 404))

    def test_get_run_returns_404_for_unknown_run(self) -> None:
        response = self.client.get("/api/runs/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_failed_run_reports_error(self) -> None:
        app_module.run_manager = RunManager(
            to_review_runner=_failing_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_status(run_id, "failed")

        response = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("simulated pipeline failure", response.json()["error"])

    def test_promote_missing_skill_adds_to_cache(self) -> None:
        run_id = self._run_to_completion()

        response = self.client.post(f"/api/runs/{run_id}/missing-skills/kubernetes/promote")

        self.assertEqual(response.status_code, 200)
        # Stored capitalized (Kubernetes) - see task 1's display-bug fix.
        names = {entry["name"] for entry in response.json()}
        self.assertIn("Kubernetes", names)

    # --- posting text ---

    def test_get_run_posting_returns_the_uploaded_text(self) -> None:
        posting_text = "We need Python and SQL skills."
        run_id = self.client.post("/api/runs", json={"posting_text": posting_text}).json()["run_id"]

        response = self.client.get(f"/api/runs/{run_id}/posting")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, posting_text)

    def test_get_run_posting_returns_404_for_unknown_run(self) -> None:
        response = self.client.get("/api/runs/does-not-exist/posting")

        self.assertEqual(response.status_code, 404)

    # --- run history persistence across a backend restart ---

    def test_run_history_survives_a_run_manager_restart(self) -> None:
        run_id = self._run_to_completion()

        # Simulate a backend restart: a brand new RunManager instance,
        # pointed at the same persisted index file, with no in-memory state
        # carried over from the old one.
        app_module.run_manager = RunManager(
            to_review_runner=_fake_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        response = self.client.get("/api/runs")

        self.assertEqual(response.status_code, 200)
        run_ids = {entry["run_id"] for entry in response.json()}
        self.assertIn(run_id, run_ids)
        restored = next(entry for entry in response.json() if entry["run_id"] == run_id)
        self.assertEqual(restored["status"], "completed")

        # Full run detail (metrics, etc.) is still lazily readable from disk
        # for the restored run, not just the lightweight list entry.
        detail = self.client.get(f"/api/runs/{run_id}").json()
        self.assertEqual(detail["status"], "completed")
        self.assertIn("metrics", detail)

    def test_a_run_still_marked_running_at_restart_is_surfaced_as_failed(self) -> None:
        app_module.run_manager = RunManager(
            to_review_runner=_slow_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        # Don't wait for anything - simulate the process dying mid-phase-1,
        # then a fresh RunManager loading the index back in.
        app_module.run_manager = RunManager(
            to_review_runner=_fake_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        response = self.client.get(f"/api/runs/{run_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("interrupted", response.json()["error"].lower())

    def test_a_run_awaiting_review_at_restart_stays_awaiting_review(self) -> None:
        # Per the "save states to disk" Phase 9 decision: unlike a genuinely
        # interrupted "running" phase, "awaiting_review" is a stable,
        # resumable state (Stage 1 already finished and wrote its output
        # durably to disk) - it must survive a restart as-is, not be
        # reclassified as failed.
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_status(run_id, "awaiting_review")

        app_module.run_manager = RunManager(
            to_review_runner=_fake_to_review_runner,
            from_review_runner=_fake_from_review_runner,
            index_path=self.index_path,
        )

        payload = self.client.get(f"/api/runs/{run_id}").json()
        self.assertEqual(payload["status"], "awaiting_review")
        self.assertIn("skill_review", payload)

        # And still resumable after the restart.
        response = self.client.post(f"/api/runs/{run_id}/confirm-skills", json={"included_skills": ["python"]})
        self.assertEqual(response.status_code, 200)
        self._wait_for_status(run_id, "completed")

    def _wait_for_status(self, run_id: str, expected_status: str, timeout: float = 5) -> dict:
        deadline = time.time() + timeout
        payload: dict = {}
        while time.time() < deadline:
            payload = self.client.get(f"/api/runs/{run_id}").json()
            if payload.get("status") == expected_status:
                return payload
            time.sleep(0.05)
        self.fail(f"Run {run_id} did not reach status {expected_status!r} in time (last={payload})")

    def _run_to_completion(self, posting_text: str = "We need Python.") -> str:
        run_id = self.client.post("/api/runs", json={"posting_text": posting_text}).json()["run_id"]
        self._wait_for_status(run_id, "awaiting_review")
        response = self.client.post(f"/api/runs/{run_id}/confirm-skills", json={"included_skills": ["python"]})
        self.assertEqual(response.status_code, 200)
        self._wait_for_status(run_id, "completed")
        return run_id


if __name__ == "__main__":
    unittest.main()
