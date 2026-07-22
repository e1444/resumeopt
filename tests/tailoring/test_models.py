"""Smoke tests for the Phase 0 typed artifact contracts in `tailoring.models`.

These are shape/immutability checks only - no business logic lives on these
dataclasses (see `tailoring.validation` for that), so the tests intentionally
stay thin.
"""

import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tailoring.models import (
    BaselineBullet,
    BulletPdfFitDiagnostic,
    CoreClaimMolecule,
    FactAtom,
    ProjectBaseline,
    ProjectFactMatch,
    ProtectionState,
    RepairStep,
    ResumeManifest,
    SelectedBullet,
    SlotCandidateSet,
    SlotTriageResult,
    VerificationResult,
)

ALL_MODEL_CLASSES = (
    BaselineBullet,
    BulletPdfFitDiagnostic,
    CoreClaimMolecule,
    FactAtom,
    ProjectBaseline,
    ProjectFactMatch,
    ProtectionState,
    RepairStep,
    ResumeManifest,
    SelectedBullet,
    SlotCandidateSet,
    SlotTriageResult,
    VerificationResult,
)


class ModelsAreFrozenDataclassesTest(unittest.TestCase):
    def test_every_artifact_model_is_a_frozen_dataclass(self) -> None:
        for model_class in ALL_MODEL_CLASSES:
            self.assertTrue(dataclasses.is_dataclass(model_class), model_class.__name__)
            self.assertTrue(model_class.__dataclass_params__.frozen, model_class.__name__)


class FactAtomTest(unittest.TestCase):
    def test_construction_and_immutability(self) -> None:
        atom = FactAtom(id="p_fact_001", fact="Did a thing.", skill_tags=("python",))

        self.assertEqual(atom.id, "p_fact_001")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            atom.fact = "Did a different thing."  # type: ignore[misc]


class BaselineBulletAndProjectBaselineTest(unittest.TestCase):
    def test_project_baseline_holds_ordered_bullets(self) -> None:
        bullet_one = BaselineBullet(
            id="p_b1", project_id="p", order=0, text="First.", position="start"
        )
        bullet_two = BaselineBullet(
            id="p_b2", project_id="p", order=1, text="Second.", position="end"
        )
        project = ProjectBaseline(
            project_id="p",
            project_title="P",
            role_context="Context",
            dates="2020",
            resume_section="PROJECTS",
            bullets=(bullet_one, bullet_two),
        )

        self.assertEqual([bullet.id for bullet in project.bullets], ["p_b1", "p_b2"])


class ProtectionStateTest(unittest.TestCase):
    def test_defaults_to_no_reserved_facts(self) -> None:
        state = ProtectionState(bullet_id="p_b1", project_id="p", protected=False)

        self.assertEqual(state.reserved_fact_ids, ())


if __name__ == "__main__":
    unittest.main()
