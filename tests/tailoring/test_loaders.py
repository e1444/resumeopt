"""Deterministic tests for `tailoring.loaders` YAML I/O (Phase 0)."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tailoring.loaders import load_fact_atoms, load_project_baseline, load_resume_manifest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "evals" / "tailoring"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE_DIR = REPO_ROOT / "data" / "experience"


class LoadFactAtomsTest(unittest.TestCase):
    def test_loads_ids_facts_and_skill_tags(self) -> None:
        atoms = load_fact_atoms(FIXTURES_DIR / "valid_baseline" / "demo_project_fact_atoms.yaml")

        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms[0].id, "demo_project_fact_001")
        self.assertEqual(
            atoms[0].fact, "Implemented a REST API endpoint for user authentication."
        )
        self.assertIn("rest api", atoms[0].skill_tags)


class LoadProjectBaselineTest(unittest.TestCase):
    def test_loads_bullets_with_fact_ids_and_position(self) -> None:
        project = load_project_baseline(
            FIXTURES_DIR / "valid_baseline" / "demo_project_bullets.yaml"
        )

        self.assertEqual(project.project_id, "demo_project")
        self.assertEqual(len(project.bullets), 1)
        bullet = project.bullets[0]
        self.assertEqual(bullet.id, "demo_project_b1")
        self.assertEqual(bullet.position, "start")
        self.assertEqual(
            bullet.fact_ids, ("demo_project_fact_001", "demo_project_fact_002")
        )


class LoadResumeManifestRealDataTest(unittest.TestCase):
    """Loads the REAL data/experience/ resources (not a fixture)."""

    def test_manifest_preserves_display_order_and_project_count(self) -> None:
        manifest = load_resume_manifest(EXPERIENCE_DIR)

        self.assertEqual(manifest.source_template_path, "data/template.tex")
        self.assertEqual(len(manifest.projects), 5)
        self.assertEqual(
            [project.project_id for project in manifest.projects],
            [
                "research_assistant",
                "c_discord_bot_platform",
                "benchmark_driven_llm_workflow_orchestration",
                "constrained_optimization_for_generative_classification",
                "university_of_toronto",
            ],
        )

    def test_every_project_has_at_least_one_bullet(self) -> None:
        manifest = load_resume_manifest(EXPERIENCE_DIR)

        for project in manifest.projects:
            self.assertTrue(project.bullets, f"{project.project_id} has no bullets")

    def test_bullet_positions_are_well_formed(self) -> None:
        manifest = load_resume_manifest(EXPERIENCE_DIR)

        for project in manifest.projects:
            positions = [bullet.position for bullet in project.bullets]
            self.assertEqual(positions[0], "start")
            self.assertEqual(positions[-1], "end")
            for position in positions[1:-1]:
                self.assertEqual(position, "middle")


if __name__ == "__main__":
    unittest.main()
