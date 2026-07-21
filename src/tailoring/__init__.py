"""Fact-grounded experience-bullet tailoring package.

See docs/agent/BULLET_TAILORING_DEV_PLAN.md (the authoritative source for
sequencing/contracts). Phase 0: typed artifact models, YAML loaders, and
source-data validation. The baseline resume-manifest/bullets resources under
`data/experience/` were prepared manually for this template snapshot (no
preprocessing pipeline module exists - per explicit decision, resume-
template preprocessing is out of scope until the future resume-upload/
onboarding workflow is built). Phase 1 (added 2026-07-20): job-requirements
extraction (`tailoring.requirements`, reusing `parser.factory.parse_posting`)
and advisory, non-mutating slot triage (`tailoring.triage`, one narrow LLM
call per bullet). Phase 2 (added 2026-07-21): project-level fact retrieval
(`tailoring.retrieval`, reuses `matcher.ExactAliasMatcher`/`SemanticMatcher`
rather than duplicating match logic). No LangGraph orchestration exists yet
in this package - later phases must not be inferred from this module's
presence.
"""

from __future__ import annotations

from tailoring.models import (
    BaselineBullet,
    BulletPdfFitDiagnostic,
    CoreClaimMolecule,
    ExpandedClaimMolecule,
    FactAtom,
    JobRequirements,
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
    "JobRequirements",
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
