"""Deterministic Phase 0 contract tests for `tailoring.validation`.

No LLM calls, no network. Exercises the `tests/evals/tailoring/<case>/`
fixture package (each paired with an `expected_outcome.yaml` stating the
expected issues and the human-facing rationale) plus the REAL
`data/experience/` baseline resources, per the dev plan's Phase 0 validation
gate: every fact_id referenced by every baseline bullet must resolve,
protection derivation must be consistent, and invalid source data must fail
with actionable errors (duplicate ids, unknown references, invalid project
ownership, non-atomic fact-shape warnings).
"""

import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tailoring.loaders import load_fact_atoms, load_project_baseline, load_resume_manifest
from tailoring.validation import (
    derive_protection_states,
    has_errors,
    validate_baseline_bullets,
    validate_fact_atoms,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "evals" / "tailoring"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE_DIR = REPO_ROOT / "data" / "experience"


def _load_expected(case_dir: Path) -> dict:
    return yaml.safe_load((case_dir / "expected_outcome.yaml").read_text(encoding="utf-8"))


class ValidBaselineFixtureTest(unittest.TestCase):
    def test_no_issues(self) -> None:
        case_dir = FIXTURES_DIR / "valid_baseline"
        atoms = load_fact_atoms(case_dir / "demo_project_fact_atoms.yaml")
        project = load_project_baseline(case_dir / "demo_project_bullets.yaml")

        fact_issues = validate_fact_atoms(atoms)
        bullet_issues = validate_baseline_bullets(
            project.bullets, {atom.id for atom in atoms}, project.project_id
        )

        self.assertEqual(fact_issues, [])
        self.assertEqual(bullet_issues, [])


class DuplicateFactIdFixtureTest(unittest.TestCase):
    def test_duplicate_id_is_hard_error(self) -> None:
        case_dir = FIXTURES_DIR / "duplicate_fact_ids"
        atoms = load_fact_atoms(case_dir / "demo_project_fact_atoms.yaml")

        issues = validate_fact_atoms(atoms)
        expected = _load_expected(case_dir)

        self.assertTrue(has_errors(issues))
        codes = {(issue.severity, issue.code) for issue in issues}
        for expectation in expected["expected_fact_atom_issues"]:
            self.assertIn((expectation["severity"], expectation["code"]), codes)


class UnknownFactReferenceFixtureTest(unittest.TestCase):
    def test_unknown_reference_is_hard_error(self) -> None:
        case_dir = FIXTURES_DIR / "unknown_fact_reference"
        atoms = load_fact_atoms(case_dir / "demo_project_fact_atoms.yaml")
        project = load_project_baseline(case_dir / "demo_project_bullets.yaml")

        issues = validate_baseline_bullets(
            project.bullets, {atom.id for atom in atoms}, project.project_id
        )
        expected = _load_expected(case_dir)

        self.assertTrue(has_errors(issues))
        codes = {(issue.severity, issue.code) for issue in issues}
        for expectation in expected["expected_bullet_issues"]:
            self.assertIn((expectation["severity"], expectation["code"]), codes)


class InvalidPositionFixtureTest(unittest.TestCase):
    def test_invalid_position_is_hard_error(self) -> None:
        case_dir = FIXTURES_DIR / "invalid_position"
        project = load_project_baseline(case_dir / "demo_project_bullets.yaml")

        issues = validate_baseline_bullets(project.bullets, set(), project.project_id)
        expected = _load_expected(case_dir)

        self.assertTrue(has_errors(issues))
        codes = {(issue.severity, issue.code) for issue in issues}
        for expectation in expected["expected_bullet_issues"]:
            self.assertIn((expectation["severity"], expectation["code"]), codes)


class InvalidProjectOwnershipFixtureTest(unittest.TestCase):
    def test_invalid_ownership_is_hard_error(self) -> None:
        case_dir = FIXTURES_DIR / "invalid_project_ownership"
        project = load_project_baseline(case_dir / "demo_project_bullets.yaml")

        # The fixture's bullet entry declares project_id "other_project",
        # while the containing file's own project_id is "demo_project" -
        # the loader preserves that mismatch (see tailoring.loaders), and
        # validation must catch it against the file's real project_id.
        issues = validate_baseline_bullets(project.bullets, set(), "demo_project")
        expected = _load_expected(case_dir)

        self.assertTrue(has_errors(issues))
        codes = {(issue.severity, issue.code) for issue in issues}
        for expectation in expected["expected_bullet_issues"]:
            self.assertIn((expectation["severity"], expectation["code"]), codes)


class NonAtomicFactWarningFixtureTest(unittest.TestCase):
    def test_non_atomic_fact_is_warning_not_error(self) -> None:
        case_dir = FIXTURES_DIR / "non_atomic_fact_warning"
        atoms = load_fact_atoms(case_dir / "demo_project_fact_atoms.yaml")

        issues = validate_fact_atoms(atoms)
        expected = _load_expected(case_dir)

        self.assertFalse(has_errors(issues), "non-atomic shape must be a warning, not an error")
        warning_issues = [
            issue
            for issue in issues
            if issue.severity == "warning" and issue.code == "possibly_non_atomic_fact"
        ]
        self.assertTrue(warning_issues)
        for expectation in expected["expected_fact_atom_issues"]:
            self.assertEqual(expectation["severity"], "warning")


class ProtectionStateFixtureTest(unittest.TestCase):
    def test_protection_matches_expected(self) -> None:
        case_dir = FIXTURES_DIR / "protection_state"
        project = load_project_baseline(case_dir / "demo_project_bullets.yaml")
        triage = yaml.safe_load((case_dir / "triage.yaml").read_text(encoding="utf-8"))
        expected = _load_expected(case_dir)["expected_protection"]

        states_by_id = {
            state.bullet_id: state
            for state in derive_protection_states(project.bullets, triage)
        }

        for bullet_id, expectation in expected.items():
            state = states_by_id[bullet_id]
            self.assertEqual(state.protected, expectation["protected"], bullet_id)
            self.assertEqual(
                sorted(state.reserved_fact_ids),
                sorted(expectation["reserved_fact_ids"]),
                bullet_id,
            )


class RealBaselineResourcesTest(unittest.TestCase):
    """The dev plan's actual Phase 0 validation gate, run against real data."""

    def test_every_fact_id_referenced_by_every_baseline_bullet_resolves(self) -> None:
        manifest = load_resume_manifest(EXPERIENCE_DIR)
        self.assertTrue(manifest.projects, "expected at least one project in the real baseline")

        for project in manifest.projects:
            fact_path = (
                EXPERIENCE_DIR / project.project_id / f"{project.project_id}_fact_atoms.yaml"
            )
            atoms = load_fact_atoms(fact_path)

            fact_issues = validate_fact_atoms(atoms)
            bullet_issues = validate_baseline_bullets(
                project.bullets, {atom.id for atom in atoms}, project.project_id
            )

            self.assertFalse(
                has_errors(fact_issues),
                f"{project.project_id} fact atoms have hard errors: {fact_issues}",
            )
            self.assertFalse(
                has_errors(bullet_issues),
                f"{project.project_id} bullets have hard errors: {bullet_issues}",
            )


if __name__ == "__main__":
    unittest.main()
