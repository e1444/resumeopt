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


def _fake_pipeline_runner(config, on_stage=None, on_substage=None) -> None:
    """Writes the same artifact shapes `main.run_pipeline` would, instantly.

    Calls `on_stage` for every stage in `PIPELINE_STAGES` (like the real
    pipeline would) so stage-progress-reporting behavior can be tested
    without a real, slow run.
    """

    for stage in PIPELINE_STAGES:
        if on_stage is not None:
            on_stage(stage)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "run_metrics.json").write_text(
        json.dumps({"parse": {"matched_skill_count": 1}}), encoding="utf-8"
    )
    (logs_dir / "missing_skills.json").write_text(
        json.dumps({"missing_skills": ["kubernetes"], "count": 1}), encoding="utf-8"
    )
    (run_root / "tailored_resume.pdf").write_bytes(b"%PDF-1.4 fake pdf content")


def _failing_pipeline_runner(config, on_stage=None, on_substage=None) -> None:
    raise RuntimeError("simulated pipeline failure")


def _slow_pipeline_runner(config, on_stage=None, on_substage=None) -> None:
    """Calls on_stage for the first stage only, then blocks - lets a test
    observe an in-flight run's current_stage before it completes."""

    if on_stage is not None:
        on_stage(PIPELINE_STAGES[0])
    time.sleep(2)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_metrics.json").write_text(json.dumps({}), encoding="utf-8")
    (run_root / "tailored_resume.pdf").write_bytes(b"%PDF-1.4 fake pdf content")


def _slow_with_substage_pipeline_runner(config, on_stage=None, on_substage=None) -> None:
    """Like `_slow_pipeline_runner`, but also reports substage batch progress
    within the `parse_posting` stage - lets a test observe an in-flight run's
    substage progress before it completes."""

    if on_stage is not None:
        on_stage("parse_posting")
    if on_substage is not None:
        on_substage("extraction", 1, 3)
    time.sleep(2)

    run_root = Path("build") / config.run_name
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_metrics.json").write_text(json.dumps({}), encoding="utf-8")
    (run_root / "tailored_resume.pdf").write_bytes(b"%PDF-1.4 fake pdf content")


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

        self._orig_skills_path = app_module.DEFAULT_SKILLS_CACHE_PATH
        self._orig_template_path = app_module.DEFAULT_TEMPLATE_PATH
        self._orig_run_manager = app_module.run_manager
        app_module.DEFAULT_SKILLS_CACHE_PATH = self.skills_cache_path
        app_module.DEFAULT_TEMPLATE_PATH = self.template_path
        app_module.run_manager = RunManager(pipeline_runner=_fake_pipeline_runner)

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

    def test_post_run_starts_and_completes(self) -> None:
        response = self.client.post("/api/runs", json={"posting_text": "We need Python."})
        self.assertEqual(response.status_code, 200)
        run_id = response.json()["run_id"]

        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            status_response = self.client.get(f"/api/runs/{run_id}")
            status = status_response.json()["status"]
            if status != "running":
                break
            time.sleep(0.05)

        self.assertEqual(status, "completed")
        self.assertIn("metrics", status_response.json())

    def test_get_run_exposes_current_stage_while_running(self) -> None:
        app_module.run_manager = RunManager(pipeline_runner=_slow_pipeline_runner)

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

        self._wait_for_completion(run_id)

    def test_get_run_exposes_substage_progress_while_running(self) -> None:
        app_module.run_manager = RunManager(pipeline_runner=_slow_with_substage_pipeline_runner)

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

        self._wait_for_completion(run_id)

    def test_get_run_pdf_available_after_completion(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_completion(run_id)

        response = self.client.get(f"/api/runs/{run_id}/pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")

    def test_get_run_log_returns_allowed_log(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_completion(run_id)

        response = self.client.get(f"/api/runs/{run_id}/logs/missing_skills.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["missing_skills"], ["kubernetes"])

    def test_get_run_log_rejects_unknown_log_name(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_completion(run_id)

        response = self.client.get(f"/api/runs/{run_id}/logs/../../etc/passwd")

        self.assertIn(response.status_code, (400, 404))

    def test_get_run_returns_404_for_unknown_run(self) -> None:
        response = self.client.get("/api/runs/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_failed_run_reports_error(self) -> None:
        app_module.run_manager = RunManager(pipeline_runner=_failing_pipeline_runner)

        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_completion(run_id, expected_status="failed")

        response = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("simulated pipeline failure", response.json()["error"])

    def test_promote_missing_skill_adds_to_cache(self) -> None:
        run_id = self.client.post("/api/runs", json={"posting_text": "We need Python."}).json()["run_id"]
        self._wait_for_completion(run_id)

        response = self.client.post(f"/api/runs/{run_id}/missing-skills/kubernetes/promote")

        self.assertEqual(response.status_code, 200)
        # Stored capitalized (Kubernetes) - see task 1's display-bug fix.
        names = {entry["name"] for entry in response.json()}
        self.assertIn("Kubernetes", names)

    def _wait_for_completion(self, run_id: str, expected_status: str = "completed") -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            status = self.client.get(f"/api/runs/{run_id}").json()["status"]
            if status == expected_status:
                return
            time.sleep(0.05)
        self.fail(f"Run {run_id} did not reach status {expected_status!r} in time")


if __name__ == "__main__":
    unittest.main()
