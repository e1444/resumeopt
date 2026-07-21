"""Fact-grounded experience-bullet tailoring package.

Phase 0 scope only (see docs/agent/BULLET_TAILORING_DEV_PLAN.md, the
authoritative source for sequencing/contracts): typed artifact models, YAML
loaders, and source-data validation. The baseline resume-manifest/bullets
resources under `data/experience/` were prepared manually for this template
snapshot (no preprocessing pipeline module exists - per explicit decision,
resume-template preprocessing is out of scope until the future resume-
upload/onboarding workflow is built). No LangGraph orchestration and no
LLM-backed pipeline behavior exist yet in this package - those are later
phases and must not be inferred from this module's presence.
"""

from __future__ import annotations

from tailoring.models import (
    BaselineBullet,
    BulletPdfFitDiagnostic,
    CoreClaimMolecule,
    ExpandedClaimMolecule,
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

__all__ = [
    "BaselineBullet",
    "BulletPdfFitDiagnostic",
    "CoreClaimMolecule",
    "ExpandedClaimMolecule",
    "FactAtom",
    "ProjectBaseline",
    "ProjectFactMatch",
    "ProtectionState",
    "RepairStep",
    "ResumeManifest",
    "SelectedBullet",
    "SlotCandidateSet",
    "SlotTriageResult",
    "VerificationResult",
]
