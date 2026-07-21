"""Typed artifact contracts for the fact-grounded bullet-tailoring pipeline.

Schema definitions only, per docs/agent/BULLET_TAILORING_DEV_PLAN.md (the
authoritative source for sequencing/contracts). Dataclasses (not pydantic)
are used to match this repository's existing convention (see
`matcher.models.SkillRecord`); no new dependency is introduced. Every
artifact type listed in the dev plan's Phase 0 "Tasks" item 4 is
represented here, even for later-phase artifacts (`ProjectFactMatch`
through `BulletPdfFitDiagnostic`), so the full artifact surface is
reviewable before any stage's behavior is implemented. Fields on
later-phase artifacts are deliberately conservative/minimal - they exist to
pin down artifact *shape*, not to lock in unreviewed design decisions about
each stage's internal logic. `JobRequirements` was added in Phase 1 (not
part of the original Phase 0 list) as that phase's first deliverable.

Frozen dataclasses are used throughout: every one of these artifact types is
either durable, human-authored source data (`FactAtom`, `BaselineBullet`) or an
immutable run-artifact record (everything else) - nothing here is meant to be
mutated in place after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

# Bullet display-position, relative to its project entry. This is an ordering
# signal only, per the dev plan: "start often anchors an entry, middle points
# are generally order-agnostic, end is only the final displayed point, not
# necessarily a conclusion."
Position = Literal["start", "middle", "end"]

# Slot triage labels (Phase 1). Advisory only - never assigns a generated
# claim to a particular slot.
TriageLabel = Literal["keep", "candidate_for_replacement", "deprioritize", "idk"]

# Match tiers reused conceptually from `src/matcher/` (exact/alias/semantic);
# Phase 2 project-fact retrieval may also use a "grounded" tier per the dev
# plan's "matching only where needed" language.
MatchTier = Literal["exact", "alias", "semantic", "grounded"]

# Verification status (Phase 5).
VerificationStatus = Literal["pass", "idk", "fail"]

# Typed repair sequence (Phase 5) - fixed order, one repair attempt per type.
RepairType = Literal["hallucination", "bad_flow", "bad_wording", "unresolvable"]

# Final human-selection source kind (Phase 7).
SelectionSource = Literal["original", "alternative", "manual"]


@dataclass(frozen=True)
class FactAtom:
    """A durable, human-authored, genuinely atomic fact about one project.

    Source of truth: `data/experience/<project_id>/<project_id>_fact_atoms.yaml`.
    Facts are NOT generated pipeline output - they are curated source data,
    same tier as `data/skills.yaml`. A fact atom that is a draft awaiting
    human review is marked at the file level (`_review_status` in the YAML),
    not per-atom, since an entire project's decomposition is reviewed as a
    unit.
    """

    id: str
    fact: str
    skill_tags: Tuple[str, ...] = ()
    rationale: Optional[str] = None


@dataclass(frozen=True)
class BaselineBullet:
    """One bullet currently rendered in `data/template.tex` for a project.

    Source of truth: `data/experience/<project_id>/<project_id>_bullets.yaml`,
    manually prepared from the current `data/template.tex` snapshot (no
    preprocessing pipeline module exists yet - see `tailoring.loaders` for
    the read-side YAML mapping). NOT a candidate library (no generated
    alternatives live here; see `SlotCandidateSet` for those).
    """

    id: str
    project_id: str
    order: int
    text: str
    position: Position
    fact_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectBaseline:
    """One resume project entry (an EXPERIENCE/PROJECTS/EDUCATION block)."""

    project_id: str
    project_title: str
    role_context: str
    dates: str
    resume_section: str
    bullets: Tuple[BaselineBullet, ...] = ()


@dataclass(frozen=True)
class ResumeManifest:
    """The whole preprocessed resume: durable input to every tailoring run.

    Source of truth: `data/experience/resume_manifest.yaml` plus each
    project's own bullets file. `projects` preserves resume display order.
    """

    source_template_path: str
    projects: Tuple[ProjectBaseline, ...] = ()


@dataclass(frozen=True)
class ProtectionState:
    """Derived (not stored) protection status for one baseline bullet.

    `keep`/`idk` triage labels protect a bullet: its facts are reserved and
    generated claims may not restate its primary accomplishment.
    `candidate_for_replacement`/`deprioritize` bullets are eligible and do
    not reserve their linked facts.
    """

    bullet_id: str
    project_id: str
    protected: bool
    reserved_fact_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class JobRequirements:
    """Phase 1: extracted job-posting requirements, reusing the existing
    production parser pipeline (`parser.factory.parse_posting`) rather than
    duplicating extraction logic. Source of truth for one run's
    `requirements.json`.

    `role_title`/`seniority`/`industry_domain`/`core_requirements`/
    `nice_to_have`/`summary_paragraph` come straight from the parser's own
    Stage 0 `PostingSummary`. `matched_skills`/`missing_skills` retain the
    parser's cache-matched terms and unmatched-but-grounded terms ("matched
    terms, relevant source context" per the dev plan) for reuse by later
    phases (e.g. Phase 2 fact retrieval) without a second parse. `raw_terms`
    intentionally excludes discarded/excluded candidates - triage only needs
    what the posting is actually asking for, not the parser's full internal
    debug trace (still available separately via the parser's own
    `extraction_debug_samples` if ever needed).
    """

    role_title: str
    seniority: str
    industry_domain: str
    core_requirements: Tuple[str, ...]
    nice_to_have: Tuple[str, ...]
    summary_paragraph: str
    matched_skills: Tuple[Dict[str, Any], ...] = ()
    missing_skills: Tuple[str, ...] = ()
    parser_provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotTriageResult:
    """Phase 1: advisory triage of one baseline bullet against requirements.

    Identifies which points are eligible for replacement; does not assign
    any particular generated claim to this bullet.
    """

    bullet_id: str
    project_id: str
    label: TriageLabel
    job_relevance: Optional[str] = None
    narrative_value: Optional[str] = None
    replacement_opportunity: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class ProjectFactMatch:
    """Phase 2: one candidate fact retrieved for a project's fact pool."""

    fact_id: str
    project_id: str
    match_tier: MatchTier
    matched_target_skill: str
    score: float
    included: bool = True
    exclusion_reason: Optional[str] = None


@dataclass(frozen=True)
class CoreClaimMolecule:
    """Phase 3: one grouped, fact-cited claim proposal for a project.

    Represents both un-ranked and ranked claims - `rank`/`non_advancement_reason`
    are populated only after the ranking step runs.
    """

    id: str
    project_id: str
    claim_text: str
    supporting_fact_ids: Tuple[str, ...]
    target_skills: Tuple[str, ...]
    primary_proof: str
    rationale: str
    rank: Optional[int] = None
    non_advancement_reason: Optional[str] = None


@dataclass(frozen=True)
class ExpandedClaimMolecule:
    """Phase 4: a core claim plus bounded support-fact expansion decisions."""

    core_claim_id: str
    project_id: str
    added_support_fact_ids: Tuple[str, ...] = ()
    excluded_fact_ids: Tuple[str, ...] = ()
    exclusion_reasons: Tuple[str, ...] = ()
    stop_reason: str = ""


@dataclass(frozen=True)
class AnnotatedProposal:
    """Phase 5 (DRAFT, needs human review - new schema per AGENTS.md Human
    Review Gates): one synthesized candidate bullet, ready for
    verification.

    Neither `CoreClaimMolecule` nor `ExpandedClaimMolecule` carries actual
    bullet text (Phase 4 deliberately deferred text-authoring - see
    `ExpandedClaimMolecule`'s docstring). This is the first artifact where
    a core claim plus its expansion decision are turned into ONE fluent
    candidate bullet (`proposal_text`), via a single bounded LLM call
    (`tailoring.verification.synthesize_proposal`) immediately followed by
    fact-support verification - not free-form, uncontrolled generation.
    `supporting_fact_ids` is the union of the core claim's own cited facts
    and any facts the expansion step added.
    """

    id: str
    project_id: str
    core_claim_id: str
    proposal_text: str
    supporting_fact_ids: Tuple[str, ...]
    target_skills: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairStep:
    """Phase 5: one bounded typed-repair attempt in the fixed repair sequence."""

    repair_type: RepairType
    before_text: str
    after_text: Optional[str]
    reverified_status: Optional[VerificationStatus] = None


@dataclass(frozen=True)
class VerificationResult:
    """Phase 5: verification outcome (plus lineage) for one expanded proposal."""

    proposal_id: str
    project_id: str
    status: VerificationStatus
    failure_type: Optional[str] = None
    repair_steps: Tuple[RepairStep, ...] = ()
    final_text: Optional[str] = None


@dataclass(frozen=True)
class SlotCandidateSet:
    """Phase 6: every eligible original plus verified alternative for a project."""

    project_id: str
    eligible_original_bullet_ids: Tuple[str, ...] = ()
    verified_proposal_ids: Tuple[str, ...] = ()
    ranking_rationale: str = ""
    recommended_proposal_id: Optional[str] = None
    recommendation_reason: Optional[str] = None


@dataclass(frozen=True)
class SelectedBullet:
    """Phase 7: one final, human-confirmed selection for a display slot."""

    bullet_id: str
    project_id: str
    selection_source: SelectionSource
    final_text: str
    source_proposal_id: Optional[str] = None
    source_fact_ids: Tuple[str, ...] = ()
    user_authored: bool = False


@dataclass(frozen=True)
class BulletPdfFitDiagnostic:
    """Bullet-level PDF-fit diagnostics (page-constraint policy is deferred;
    this shape only records the observation, not any resolution decision).
    """

    bullet_id: str
    rendered_line_count: int
    fits: bool
    template_profile: str = "default"
