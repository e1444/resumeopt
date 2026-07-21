# Fact-Grounded Bullet Tailoring Development Plan

## Purpose

Implement the fact-grounded bullet-tailoring design in small, inspectable increments. [The architecture diagram](../proposals/BULLET_TAILORING_ARCHITECTURE.md) is a visual reference; this development plan is the source of truth for implementation sequencing, contracts, and validation.

The first release tailors only an existing default resume. It creates fact-grounded alternatives for explicitly triaged slots, preserves each original point as a selectable candidate, and requires user selection before rendering. It does not attempt autonomous resume rewriting.

## Scope and Non-Goals

### In scope

- Durable per-project atomic fact caches and a snapshot of the current default-resume bullets.
- Job requirement extraction using the existing parser pipeline.
- Preprocessed resume resources, slot triage, project-level fact retrieval, claim proposal, bounded support expansion, verification, human selection, and render-backed validation.
- Run artifacts that preserve every material decision from baseline through human selection.
- A web review workflow that permits selecting an original, selecting an approved alternative, or entering a manual free-text revision.

### Explicitly deferred

- Cross-project fact retrieval for a slot.
- Automatic replacement of a user-visible bullet.
- Insertion slots and automatic page filling.
- AI-assisted free-text adjustment.
- Global LLM optimization across the complete resume.
- Choosing among page-constraint resolution strategies. This has several interacting axes and is deferred until the selection workflow is stable.

## Architecture Constraints

- `src/llm/` remains the only provider layer. Every LLM classifier or repair call asks one narrow question.
- Reuse `src/parser/` for job requirement extraction and `src/matcher/` for deterministic/semantic fact-tag matching; do not create parallel extraction or matching stacks.
- Facts are durable, human-authored source data. Claim molecules, bullet proposals, rankings, verification reports, and selections are generated run artifacts.
- Triage identifies which existing points are replaceable; it does not prescribe what claim replaces any particular point. Middle/support points are largely position-agnostic, and generated claims remain project-level alternatives until the user selects a final set.
- Stored bullets are a snapshot of what is currently in `data/template.tex`; generated alternatives do not become durable source data. A cache of human-approved generated bullets is a separate future extension.
- Each stored bullet records its current structural position as `start`, `middle`, or `end`. This is an ordering signal, not a rhetorical classification: a `start` point often anchors an entry, `middle` points are generally order-agnostic supporting evidence, and an `end` point is only the final displayed point, not necessarily a conclusion.
- Resume preprocessing belongs to the future resume-upload/onboarding workflow, not the tailoring graph. This repository does not yet support uploading resumes; create the durable resume-manifest and per-project baseline YAML resources for the current template, then treat them as required inputs to every tailoring step.
- Non-triaged baseline points are protected. Their linked facts may not be used for generated alternatives, and generated claims may not restate their primary accomplishments. Repeated technologies are allowed; repeated primary accomplishment proofs are only an advisory global concern.
- New schemas and evaluation ground truth require human review before becoming contracts.
- Use LangGraph as the initial orchestration baseline. Establish the typed graph, state, edges, and artifact ownership for every stage before implementing individual stage behavior.

## Active Task List

Keep this list current during implementation.

- [ ] Review and approve Phase 0 schemas, baseline-resource contracts, and fixtures.
- [ ] Create the complete LangGraph boilerplate from the approved Phase 0 contracts.
- [ ] Build and benchmark the local offline vertical slice through Phase 3.
- [ ] Add support expansion and verification only after local proposal quality passes its gate.
- [ ] Add candidate competition, human selection, and rendering after the offline artifacts are stable.
- [ ] Validate the complete workflow with fixed fixtures and focused tests; decide page-constraint handling separately.

## Mandatory Phase Protocol

Every phase follows this order:

1. Create and human-review a fixture package before writing production behavior. It contains representative inputs, expected outputs, permitted ambiguous outcomes, and the rationale requested from an evaluator.
2. Add deterministic contract tests for the fixture shape and assertions that do not require an LLM.
3. For LLM generation or judgment, measure each requirement through at least four small, specific, single-question LLM classifiers. Preserve every verdict and disagreement, then inspect results at the term or claim level; agreement informs review but is not ground truth.
4. Record prompt/model versions, calls, latency, tokens, and estimated cost with the quality result.
5. Implement or promote the phase only after fixture review and empirical validation are complete.

## Bootstrap: Full LangGraph Boilerplate

Phase 0 is a prerequisite for this bootstrap. The graph is the first implementation step, but it must be built from reviewed models, artifact names, fixture shapes, and protection semantics rather than inventing them.

### Goal

Establish the complete orchestration contract before implementing any individual pipeline phase.

### First Deliverable

Create a graph-level fixture describing a successful no-op traversal, a no-candidate traversal, a verification-reject traversal, and a human-review pause/resume traversal. Each expected state transition includes its rationale and emitted artifact names.

### Tasks

1. Add `langgraph` to the dependency manifest and lock file.
2. Create `src/tailoring/` with typed state, artifact identifiers, and one node interface for every tailoring stage: requirements, triage, project retrieval, claim proposal, ranking, expansion, verification, slot competition, global advice, human selection, and deferred page-fit handling. Resume preprocessing is an external prerequisite, not a graph node.
3. Define the LangGraph edges, conditional routes, checkpoint boundary, and explicit terminal states without embedding stage behavior in the graph layer.
4. Make every placeholder node emit a structured `not_implemented` result or raise a domain-specific exception; no node may silently skip its owned artifact.
5. Add graph construction, route, state-serialization, and checkpoint/resume tests using the bootstrap fixtures.

### Validation Gate

- The graph can traverse each bootstrap fixture without provider calls.
- Every planned artifact has exactly one owning node and every route is explicit.
- Individual phase implementation begins only after the scaffold contract is reviewed.

## Phase 0: Contracts and Fixture Baseline

### Goal

Define the smallest reviewable source-data and artifact contracts before adding pipeline behavior.

### Tasks

1. Create and review Phase 0 fixtures: valid baseline data, duplicate IDs, unknown fact references, invalid positions, non-atomic fact warnings, and triage-derived protection states, each with expected outcomes and rationale.
2. Define the future resume-upload/onboarding preprocessing contract. It extracts the whole resume into a resume manifest plus per-project baseline bullet files containing project ownership, display order, current bullet text, and structural position. It does not generate alternatives. For the current no-upload implementation, create these resources from `data/template.tex` and treat them as existing inputs.
3. Confirm the existing `data/experience/<project>/<project>_fact_atoms.yaml` and `<project>_bullets.yaml` convention as the initial source format. Bullet files contain only points currently rendered in `data/template.tex`, with `position: start|middle|end`; they do not contain candidate libraries.
4. Define typed Python models or validation helpers for:
   - preprocessed resume snapshot
   - slot triage
   - project fact-match result
   - core and expanded claim molecule
   - verification and repair trace
   - slot candidate set
   - selected bullet set
   - bullet-level PDF-fit diagnostics
   - protection state derived from triage
5. Place schema models in the bootstrapped `src/tailoring/` package, which imports the existing parser, matcher, renderer, and LLM layers without duplicating them.
6. Select/generate a small fixed job-posting fixture set spanning at least frontend/product, backend/platform, and LLM/ML-infrastructure emphasis.
7. Create human-reviewed expected outcomes for only the first measurable stages: relevant slots, acceptable project fact pools, and forbidden cross-project/protected-fact reuse.

### Deliverables

- `src/tailoring/` package boundary and public model definitions.
- Template preprocessor plus source-data validation for duplicate IDs, missing references, atom shape, and structural position.
- Fixed fixtures under `tests/evals/` and corresponding expected review data.
- Documented artifact filenames under `build/<run_name>/logs/`.

### Validation Gate

- Human approves every new schema and fixture ground truth.
- A deterministic test resolves every `fact_id` referenced by every baseline bullet.
- A deterministic test derives protection consistently: `keep` and `idk` baseline points are protected; `candidate_for_replacement` and `deprioritize` points are eligible for replacement and do not reserve their linked facts.
- Invalid source data fails with actionable errors: duplicate IDs, unknown references, invalid project ownership, and non-atomic fact-shape warnings.
- No LLM calls are needed for this phase.

## Phase 1: Requirements and Slot Triage

### Goal

Use the existing preprocessed baseline resources and extracted job requirements to create inspectable, non-mutating slot recommendations.

### Tasks

1. Create and review triage fixtures. Include a frontend posting paired with a research/ML bullet expected to be `candidate_for_replacement`, strong aligned pairs expected to be `keep`, and ambiguous pairs with permitted labels plus requested rationale.
2. Reuse the production parser pipeline to write job requirements in `requirements.json`; retain matched terms, relevant source context, and parser provenance.
3. Implement slot triage as a narrow judgment over one existing bullet and the extracted requirements. Its output is `keep`, `candidate_for_replacement`, `deprioritize`, or `idk`, with job-relevance, narrative value, replacement opportunity, and reason fields.
4. Keep triage advisory. It identifies which baseline points are eligible for replacement, but it does not assign generated claims to particular points. Every original remains available for final human selection.
5. Write `slot_triage.json`, including every slot and an explicit explanation for non-actionable slots.

### Deliverables

- Requirements and triage artifacts that consume the pre-existing baseline resources.
- Triage implementation with a deterministic fallback for unavailable LLM providers.
- Focused unit tests using a fake provider that mirrors the real prompt/schema.

### Benchmark and Validation Gate

- On the fixed fixtures, inspect per-slot precision and recall for `candidate_for_replacement`; do not promote a default model/prompt based only on aggregate scores.
- Inspect explanations for position/context violations and false triage caused by repeated generic skills.
- Confirm that every baseline bullet can proceed to a candidate set regardless of triage label.
- Record model, call count, token usage, and latency alongside quality results before choosing defaults.

## Phase 2: Project-Level Fact Retrieval

### Goal

For each project with at least one triaged slot, retrieve one auditable pool of job-relevant facts before composing claims or assigning alternatives to slots.

### Tasks

1. Create and review retrieval fixtures with relevant project facts, facts already supporting protected baseline bullets, cross-project facts, and near matches. Specify expected inclusion/exclusion plus the rationale.
2. Derive protected facts from non-triaged baseline points: `keep` and `idk` points reserve their linked facts; `candidate_for_replacement` and `deprioritize` points do not. Load each eligible project's fact atoms, baseline bullets, protected fact IDs, and the extracted job requirements.
3. Match job-derived skills to flat fact `skill_tags` using `ExactAliasMatcher`, then semantic matching and grounded matching only where needed.
4. Exclude facts from another project and facts reserved by protected baseline bullets unless explicitly classified as broad context.
5. Record every candidate fact's match tier, matched target skill, score, project ID, and exclusion reason in `project_fact_matches.json`.
6. Cap each project pool so claim generation remains inspectable, but do not divide it by triaged slot.

### Deliverables

- Project-level retrieval service in `src/tailoring/` that calls the matcher package rather than duplicating match logic.
- `project_fact_matches.json` artifact.
- Deterministic retrieval tests and fixture-based exclusion tests.

### Validation Gate

- Verify recall of human-approved relevant facts and precision of returned facts for each fixture.
- Assert no returned fact belongs to a different project or is reserved by a protected baseline bullet.
- Compare exact-only and semantic/grounded matching by quality, call count, latency, and cost before making escalation defaults.

## Phase 3: Local Claim Proposal and Ranking

### Goal

Generate narrow, fact-cited claim molecules by grouping each project-level fact pool, then rank claims as project-level alternatives for later human selection.

### Tasks

1. Create and review proposal fixtures only for separable grouping constraints: a project fact pool with distinct frontend and backend work must not merge those into one claim; injected noise must remain unused; and a pool with no coherent grouping must permit no claim. Record the required rationale and any allowed ambiguity, not an exhaustive expected claim set.
2. Ask the LLM to discover and generate three to six coherent claim molecules from the bounded project-level fact pool. It groups facts and proposes the corresponding claim in one structured generation step; it does not test pre-enumerated fact pools against slots.
3. Require every generated claim to cite core fact IDs, name target skills, identify its primary proof, and return no claim when no coherent grouping exists.
4. Persist all outputs, including non-advancing candidates, in `unranked_core_claim_molecules.json`.
5. Rank each core molecule in a separate single-purpose step or deterministic scorer using direct skill coverage, fact support, narrowness, local novelty, likely expansion value, and broad relevance to the project's replaceable points. Ranking is advisory; it only needs to reliably promote one or two viable candidates to bounded expansion, not create a perfect total order.
6. Keep the top one or two candidates per project selection round; reserve primary facts after an accepted candidate and repeat only while non-overlapping facts remain.
7. Write `core_claim_molecules.json` with ranking rationale and non-advancement reasons.

### Claim-Generation Prompt Contract

The exact wording may evolve through fixture-backed benchmarks, but the initial generation prompt must require three to six coherent accomplishment claims and the following constraints:

- Direct support from one to ten fact atoms, close factual wording, and minimal inference.
- A plausible resume-bullet claim, with specific measurable outcomes where the supplied facts support them.
- No fact reused across claims unless it is explicitly broad contextual support.
- Prefer smaller, narrower, verifiable groupings; do not merge unrelated actions, tools, or outcomes.
- Return a smaller claim or no claim rather than forcing a broad synthesis.
- For each claim, return claim text, supporting fact IDs, target skills, a primary proof, and a brief rationale for why the facts belong together.

### Deliverables

- Claim proposal/ranking service and strict structured-output schemas.
- Fake-provider tests for no-claim behavior, fact-ID validation, coherent fact grouping, and candidate ranking.
- A reusable benchmark runner that records term-level/output-quality inspection plus call and cost metrics.

### Benchmark and Validation Gate

- Use at least four small, single-purpose LLM classifiers as the primary evaluation for fact support, single-accomplishment coherence, local novelty, and project relevance. Record all verdicts and disagreements, then inspect the promoted candidates rather than expecting perfect rankings.
- Validate separable fixture constraints directly; for open-ended claim generation, measure classifier verdict rates, duplicate-primary-proof rate, and unsupported-fact-ID rate rather than precision/recall against a supposedly complete expected claim set.
- Do not enable support expansion until the selected top one or two molecules are routinely more useful than arbitrary unranked candidates and manual inspection shows no systematic claim blending.

## Phase 4: Bounded Support Expansion and Verbosity Prefilter

### Goal

Strengthen a selected core claim without turning it into a second accomplishment or wasting verification calls on obviously overlong wording.

### Tasks

1. Create and review support-pool fixtures with obvious relevant near matches, subtle same-project support, and irrelevant/cross-project near matches. State expected inclusion/exclusion and rationale.
2. Build a support pool of at most four unused local facts, including near matches that do not directly overlap target skills.
3. Ask a single-purpose marginal-value question for each iteration: does this fact strengthen the same accomplishment?
4. Allow only `add_support`, `keep_out`, and `stop`; cap additions at three facts.
5. Persist each decision, rationale, and stop condition in `expanded_claim_molecules.json`.
6. Add a conservative template-specific line-estimate prefilter. On failure, remove the lowest-value support first, then narrow core wording; never silently replace the core claim.

### Deliverables

- Expansion controller with deterministic state transitions and bounded LLM judgments.
- Verbosity prefilter calibrated from the active template, clearly labeled non-authoritative.
- Tests for support caps, no-new-accomplishment behavior, stop conditions, and support-fact removal.

### Benchmark and Validation Gate

- Use at least four small, single-purpose LLM classifiers to compare core-only and expanded candidates for clarity, target-skill coverage, and same-claim integrity. Expansion should improve coverage or clarity without introducing a new accomplishment or context drift.
- Keep page-constraint strategy deferred. Any temporary prefilter must be deterministic, inspectable, and clearly non-authoritative; evaluate candidate rendering only once the page-fit phase is designed.
- The fixture set must cover: an obvious adjacent-frontend fact that should support a frontend claim; a subtle same-project frontend fact that may support a backend claim only when needed to establish one implementation boundary; and an irrelevant or cross-project frontend fact that must remain excluded from a backend claim.
- Keep expansion disabled by default if its quality gain does not justify additional calls or it materially raises claim-blending failures.

## Phase 5: Verification and Typed Repair

### Goal

Admit only grounded, distinct project-level alternatives into human review, with bounded repairs rather than uncontrolled regeneration.

### Tasks

1. Create and review verification fixtures for every failure type, including allowed `idk` cases and expected repair boundaries with their rationale.
2. Verify each expanded proposal for phrase-level fact support, unsupported ownership/causality/time claims, protected-fact reuse, semantic duplication of protected baseline points, project relevance, and same-claim expansion integrity. Do not infer a target replacement slot.
3. Return `pass`, `idk`, or a typed failure report.
4. Implement a repair controller with the fixed sequence `hallucination`, `bad_flow`, `bad_wording`, `unresolvable`.
5. Permit one constrained repair per required failure type, reverify after each repair, and discard on failed repair or `unresolvable`.
6. Disallow repair from retrieving facts, changing project context, replacing the core molecule, or introducing a new accomplishment.
7. Persist the full lineage in `annotated_proposal_set.json` and a summary in `verification_report.json`.

### Deliverables

- Verification and typed-repair services with structured schemas.
- Tests for every failure type, repair ordering, repair bounds, and discard behavior.
- Fixture cases with approved pass, `idk`, and reject outcomes.

### Benchmark and Validation Gate

- Report per-failure-type classifier precision/recall and inspect actual repair output, not merely verdict agreement.
- Measure whether repair improves supported, readable candidates without increasing unsupported assertions or position/context drift.
- Confirm each repair artifact has stable references to the input proposal and exact changed fact IDs/text.
- Keep `idk` visible below passed candidates rather than coercing it into acceptance or rejection.

## Phase 6: Slot Competition and Advisory Global Diversity

### Goal

Construct a selectable project-level alternative pool beside replaceable originals, then optionally recommend a resume-wide default without hiding alternatives.

### Tasks

1. Create and review competition fixtures that preserve originals, rank generated project-level alternatives, and mark overlap between primary accomplishment proofs while allowing repeated technologies. Do not require a fixture to identify one generated claim as the replacement for one specific triaged point.
2. Build `project_candidate_sets.json`, containing every verified project-level alternative and each eligible original point. Rank on relevance, support, specificity, and primary-proof distinctness; do not use page cost until a page-constraint policy exists.
3. Include every original point, including triaged points, as an available final-selection option.
4. Implement an advisory greedy global filter over the selected project-level candidates that penalizes repeated primary accomplishment proofs, not repeated skills, frameworks, or facts in general.
5. Follow the greedy recommendation with a narrow overlap validator that explains any conflict through system boundary, responsibility, constraint, outcome, or evidence type.
6. Write `default_resume_recommendation.json` with decisions and reasons; never feed it into automatic resume mutation.

### Deliverables

- Project-level candidate-pool service and advisory global recommender.
- Deterministic tests proving originals are preserved and repeated technologies do not trigger a diversity penalty.

### Validation Gate

- Evaluate local candidate rank quality before activating global advice.
- Human-review all global penalties for false overlap, especially semantically distinct projects sharing LLM, Python, or web technologies.
- Treat this stage as low priority: ship a usable local selection flow first if global advice remains inconclusive.

## Phase 7: Human Selection and Rendering Integration

### Goal

Expose candidate alternatives in the existing review workflow, persist explicit user choices, and render only the selected set.

### Tasks

1. Create and review review-workflow fixtures for keeping originals, selecting alternatives, manual text edits, restart/resume, and explicit omissions.
2. Extend the backend run state with a resumable bullet-review checkpoint after candidates are available.
3. Add read-only endpoints for bullet artifacts and a selection endpoint that persists `selected_bullet_set.json`.
4. Extend the React UI to show all original points, the verified project-level alternative pool, verification status, fact provenance, and advisory recommendation reason. Do not imply an automatic replacement pairing.
5. Provide a manual free-text field per selected point; preserve source candidate ID and fact provenance when applicable, and mark manual text as user-authored.
6. Preserve the existing skills review behavior and avoid coupling skill-cache mutation to bullet selection.
7. Resume rendering from the persisted selected set, so a server restart cannot lose review state.

### Deliverables

- Backend endpoint/run-manager support and persisted run artifacts.
- Frontend selection/review view consistent with the existing webapp patterns.
- API tests, frontend unit tests, and one manual browser workflow check.

### Validation Gate

- A user can keep every original, select project-level alternatives for the replaceable portion of an entry, make a manual edit, return after a restart, and re-render without losing choices.
- The renderer consumes the persisted selection artifact, not transient UI state.
- No review action silently changes the fact cache, baseline resume resources, or template.

## Deferred: Page-Constraint Policy

### Goal

Choose and validate the page-constraint policy after candidate selection is stable. The policy may include compiled-PDF diagnostics, user-directed omission/replacement, and approved template profiles, but this is not part of the first implementation sequence.

### Tasks

1. Create and review policy fixtures for one-page overflow, line overflow, whitespace, multiple viable omission choices, and approved layout profiles.
2. Compare policy alternatives against compiled PDFs and user-review outcomes before selecting a default.
3. Do not auto-shorten content or change typography/margins. Any layout profiles must be explicit and human-approved.

### Deliverables

- A human-reviewed page-constraint policy and artifact contract.
- A follow-up implementation plan for the selected policy.

### Validation Gate

- Do not implement page-fit behavior until the fixtures distinguish its competing resolution strategies.

## Initial Vertical Slice Order

Implement one project and a small set of existing resume slots before generalizing:

1. Load the existing ResumeOpt fact atoms and preprocessed baseline bullet resources.
2. Extract requirements for one fixed posting.
3. Triage replaceable ResumeOpt points and derive protected facts.
4. Retrieve local facts and produce/rank core molecules.
5. Verify a core-only proposal without expansion or global advice.
6. Show all originals plus verified project-level alternatives in a minimal selection artifact.

Only after this path is stable should expansion, repairs, richer UI behavior, and global advice be added.

## Artifact Contract

The preprocessed resume manifest and per-project baseline YAML are durable inputs, not per-run logs. Each tailoring run consumes that baseline and writes its artifacts under `build/<run_name>/logs/` in this dependency order:

1. `requirements.json`
2. `slot_triage.json`
3. `project_fact_matches.json`
4. `unranked_core_claim_molecules.json`
5. `core_claim_molecules.json`
6. `expanded_claim_molecules.json`
7. `annotated_proposal_set.json`
8. `verification_report.json`
9. `project_candidate_sets.json`
10. `default_resume_recommendation.json`
11. `selected_bullet_set.json`

An omitted stage must write an explicit status or omission reason rather than leave later stages unable to distinguish "not run" from "no candidates." Page-constraint artifacts are added only after a policy is selected and reviewed.

## Definition of Done for the First Release

The first release is complete when a user can submit a supported job posting, receive an inspectable set of project-level alternatives beside replaceable default-resume points, keep all originals or select alternatives, make a manual wording adjustment, and resume review after a restart.

The release must also demonstrate, on reviewed fixed fixtures, that alternatives are fact-grounded, position/context-preserving, locally non-redundant, and no worse than the originals on the intended job-relevance decision. Page-constraint handling, global diversity, insertion slots, and AI-assisted manual adjustment remain separate follow-up decisions.