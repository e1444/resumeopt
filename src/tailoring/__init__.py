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
rather than duplicating match logic). Phase 3 (added 2026-07-21): local
claim proposal and ranking (`tailoring.claims` - one structured LLM call
discovers 0-6 coherent claim molecules from a project's bounded fact pool;
ranking/selection is a separate, deterministic step). Phase 4 (added
2026-07-21): bounded support expansion and a verbosity prefilter
(`tailoring.expansion` - up to 3 additional local facts added to a
selected core claim via up to two narrow LLM calls per candidate (evidence
+ same-accomplishment integrity); a deterministic, non-authoritative
line-estimate prefilter avoids wasting later verification calls on
obviously overlong wording). Phase 5 (added 2026-07-22): proposal
synthesis and verification with typed repair (`tailoring.verification` -
`synthesize_proposal` makes ONE bounded LLM call to turn a core claim plus
its expansion decision into fluent `AnnotatedProposal.proposal_text`;
`verify_proposal` runs a deterministic protected-fact-reuse check first,
then up to 4 narrow single-purpose classifiers in a fixed order that
doubles as failure-type priority (fact_support -> `hallucination`,
same_claim_integrity -> `bad_flow`, semantic_duplication/project_relevance
-> `bad_wording`); `repair_proposal` attempts one bounded, typed repair per
distinct failure type ever encountered, reverifying after each, discarding
on a repair that doesn't resolve its own target failure or immediately on
`unresolvable`). No LangGraph
orchestration exists yet in this package - later phases must not be
inferred from this module's presence.
"""

from __future__ import annotations

from tailoring.models import (
    AnnotatedProposal,
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
    "AnnotatedProposal",
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
