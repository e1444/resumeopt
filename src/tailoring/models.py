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

# Repair resolution path (Phase 5.1, DRAFT - needs human review). Which of
# the 2-stage resolvability gate's paths a repair attempt actually took:
# edit_only rewords without dropping any currently-cited fact, remove_facts
# drops one or more currently-cited facts before rewording. Absent (None)
# on a RepairStep means the gate decided neither was viable and no rewrite
# was ever attempted for that step.
RepairResolution = Literal["edit_only", "remove_facts"]

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
    """One resume project entry (an EXPERIENCE/PROJECTS/EDUCATION block).

    `project_summary` (additive, DRAFT - needs human review per AGENTS.md
    Human Review Gates): a short, durable, human-authored description of
    what this project actually IS, written for a reader with no prior
    context - `tailoring.verification.synthesize_proposal` passes this to
    the synthesis LLM so it can write a self-contained bullet without
    assuming the reader already knows the project's domain/purpose.
    Empty string means no summary is available yet; callers must treat
    it as optional, not assume it is always populated.
    """

    project_id: str
    project_title: str
    role_context: str
    dates: str
    resume_section: str
    bullets: Tuple[BaselineBullet, ...] = ()
    project_summary: str = ""


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
class RequirementSentenceMatch:
    """Phase 3.9 (DRAFT, needs human review - new schema per AGENTS.md
    Human Review Gates): one posting requirement/responsibility sentence
    plus the skill terms the parser's own chunking/extraction stage
    attributed to it (`chunk_verdicts[raw_term]["chunk"]`), restricted to
    terms it actually kept (`included=True` - excludes discarded/
    redundant/miscategorized terms). This is an EXACT, parser-derived
    attribution, not a new fuzzy match - it exists so later phases (Phase
    2 retrieval, Phase 3 claim generation) can scope their own work to one
    requirement sentence at a time instead of one flattened whole-posting
    skill list.
    """

    sentence: str
    skill_terms: Tuple[str, ...]


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

    `requirement_sentences` (Phase 3.9, additive): the posting's own
    requirement/responsibility sentences, each with its own attributed
    skill terms, reusing the parser's existing `chunk_verdicts` byproduct
    rather than any new sentence-splitting/matching logic. Empty for a
    posting where this attribution wasn't available (e.g. loaded from an
    older persisted `requirements.json`) - callers must treat this as
    optional, not assume it is always populated.
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
    requirement_sentences: Tuple[RequirementSentenceMatch, ...] = ()


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

    Phase 3.8 (DRAFT, needs human review - new schema fields per AGENTS.md
    Human Review Gates): `why`/`result` are the claim's "nucleus" - an
    abstract motivation/theme (`why`) that this claim's own `claim_text`
    should be framed around, plus an optional concrete, SEPARATE payoff
    (`result`) when the underlying facts genuinely support one distinct
    from the why itself. `why` is always expected to be populated;
    `result` is `""` (never a fabricated placeholder) when `why` and
    `result` collapse into the same idea or no separable result exists -
    both are legitimate, common outcomes, not failure states. These are
    advisory/explanatory fields informing how `claim_text` itself should
    read (a clear center of gravity, not a flat fact enumeration) - they
    do not replace `claim_text`, `primary_proof`, or `rationale`, and are
    not yet consumed by any later phase.

    `source_requirement_sentence` (Phase 3.9, additive): the ONE posting
    requirement sentence this claim was seeded from, when generation was
    scoped to that sentence's own candidate fact pool. `None`/empty means
    this claim instead came from the RESIDUAL whole-pool generation pass
    (covering facts no posting sentence's own retrieval captured) - the
    two are deliberately distinguishable in every persisted artifact
    rather than silently merged into one undifferentiated claim list.
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
    why: str = ""
    result: str = ""
    source_requirement_sentence: Optional[str] = None


@dataclass(frozen=True)
class PostingNucleusClaim:
    """Phase 3 replacement (DRAFT, needs human review - new schema per
    AGENTS.md Human Review Gates): one why/result nucleus proposed for an
    ENTIRE posting (all requirement sentences considered together), via
    `tailoring.nucleus_pipeline`'s adaptation of the validated
    `scratch/phase3_9_spike19_*` nucleus prompt.

    Deliberately narrower than `CoreClaimMolecule`: there is no
    `claim_text` (the why/result nucleus IS the claim, synthesized
    directly into bullet text by `tailoring.verification.synthesize_proposal`)
    and no `primary_proof` (no equivalent field exists in this schema; a
    caller needing a proof string for Phase 6's overlap check should use
    `result` when present, falling back to a cited fact's own text
    otherwise). `target_skills` is NOT LLM-generated - it is derived
    deterministically as the union of `skill_tags` across every fact in
    `supporting_fact_ids`, per explicit decision. This is intentionally
    OVERINCLUSIVE (a fact's own skill_tags may cover more than what a
    particular why/result actually leans on), accepted as an acceptable
    tradeoff for avoiding a 5th LLM-judged field. `rationale` is populated
    from the nucleus prompt's own `strength_rationale` (why this theme is
    a compelling angle for THIS posting) - a posting-relevance
    justification, not `CoreClaimMolecule.rationale`'s fact-grouping-
    coherence one.

    Unlike the first (superseded) per-sentence design's
    `SentenceNucleusClaim`, there is no `source_requirement_sentence`
    field at all - a claim seeded from the WHOLE posting in one call has
    no single sentence to attribute itself to. That per-sentence design
    was replaced after a live e2e run showed heavy cross-sentence
    duplication (a generically-applicable fact independently matched many
    different sentences' own retrieval queries, producing near-identical
    nuclei with no cross-call awareness) - seeding from the whole posting
    in ONE call, asking for exactly 3 MUTUALLY DISTINCT themes, fixes both
    that and the per-sentence design's call-count cost problem at once.
    """

    id: str
    project_id: str
    supporting_fact_ids: Tuple[str, ...]
    target_skills: Tuple[str, ...]
    rationale: str
    why: str = ""
    result: str = ""


@dataclass(frozen=True)
class AnnotatedProposal:
    """Phase 5 (DRAFT, needs human review - new schema per AGENTS.md Human
    Review Gates): one synthesized candidate bullet, ready for
    verification.

    `CoreClaimMolecule` (Phase 3) does not carry actual bullet text. This
    is the first artifact where a core claim's why/result nucleus and its
    cited facts are turned into ONE fluent candidate bullet
    (`proposal_text`), via a single bounded LLM call
    (`tailoring.verification.synthesize_proposal`) immediately followed by
    fact-support verification - not free-form, uncontrolled generation.
    `supporting_fact_ids` is the core claim's own cited fact ids (Phase 4
    bounded support expansion, which used to be able to add further facts
    here, is deprecated and removed - see the dev plan's Phase 4 note).
    """

    id: str
    project_id: str
    core_claim_id: str
    proposal_text: str
    supporting_fact_ids: Tuple[str, ...]
    target_skills: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairStep:
    """Phase 5: one bounded typed-repair attempt in the fixed repair sequence.

    Phase 5.1 (DRAFT, needs human review - schema change per AGENTS.md
    Human Review Gates) adds resolution/removed_fact_ids: BEFORE any
    rewrite is attempted, a 2-stage resolvability gate decides whether the
    failure is fixable by editing alone (resolution=edit_only,
    removed_fact_ids empty) or by dropping specific currently-cited facts
    first (resolution=remove_facts, removed_fact_ids names exactly which).
    These fields make repair's fact-dropping decisions explicit and
    auditable, and let the repaired proposal's own supporting_fact_ids be
    pruned deterministically instead of silently going stale (a gap Phase
    5's live benchmark documented and left open). resolution is None if
    and only if the gate was reached but determined the failure is
    genuinely unresolvable (neither edit-only nor remove-facts was
    viable) - in that case after_text/reverified_status are also None,
    since no rewrite was ever attempted.
    """

    repair_type: RepairType
    before_text: str
    after_text: Optional[str]
    reverified_status: Optional[VerificationStatus] = None
    resolution: Optional[RepairResolution] = None
    removed_fact_ids: Tuple[str, ...] = ()


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
    """Phase 6: every eligible original plus verified alternative for a project.

    `verified_proposal_ids` is ordered by `rank_local_candidates`'s
    deterministic scorer (relevance/support/specificity/primary-proof
    distinctness) - never pruned, per the dev plan's Task 2 ("containing
    EVERY verified project-level alternative"). `recommended_proposal_id`/
    `recommendation_reason` are the advisory GLOBAL (cross-project) greedy
    diversity filter's single pick for this project, if any survived
    without a detected primary-proof overlap against a higher-priority
    pick elsewhere; both are None if no candidate was recommended (e.g. no
    verified proposals exist, or every one lost to a conflict) - the
    original bullet(s) remain the only choice in that case. Advisory only:
    never mutates `eligible_original_bullet_ids` or removes anything from
    `verified_proposal_ids`.
    """

    project_id: str
    eligible_original_bullet_ids: Tuple[str, ...] = ()
    verified_proposal_ids: Tuple[str, ...] = ()
    ranking_rationale: str = ""
    recommended_proposal_id: Optional[str] = None
    recommendation_reason: Optional[str] = None


@dataclass(frozen=True)
class ProofOverlapDecision:
    """Phase 6 (DRAFT, needs human review - new schema per AGENTS.md Human
    Review Gates): one pairwise judgment from the advisory global
    diversity filter's overlap validator (dev plan Task 5), auditable
    independent of which candidate the greedy filter ultimately dropped.

    `verdict` is `yes` (same real accomplishment, a genuine overlap),
    `no` (genuinely different accomplishments despite any surface
    similarity), or `idk` (genuinely unclear - treated as non-overlapping/
    inclusion-biased by the greedy filter, matching this codebase's
    established "if in doubt, do NOT exclude" convention, but still
    recorded here for human review rather than silently defaulted).
    `primary_dimension` is one of `system_boundary`/`responsibility`/
    `constraint`/`outcome`/`evidence_type` - the SINGLE dimension the
    verdict was primarily based on, not a exhaustive comparison across all
    5. `proposal_id_a`/`proposal_id_b` order is not meaningful (the
    judgment is symmetric).
    """

    proposal_id_a: str
    proposal_id_b: str
    verdict: Literal["yes", "no", "idk"]
    primary_dimension: Optional[str] = None
    reasoning: str = ""


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
