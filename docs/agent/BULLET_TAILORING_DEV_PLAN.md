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

## Phase 3.5 (concluded, no result): Claim Role Metadata

### Goal (as attempted)

Give expansion (and later verification) a stable, explicitly declared scope to evaluate new facts against, instead of only the claim's current literal wording. `claim_text` can be narrower than what the claim is actually FOR (for example a claim worded only around one metric may still be intended as a general model-summary point). This subphase was triggered by a live Phase 4 benchmark run surfacing a concrete real example of that gap: the core claim "Developed a flow-based generative classifier that achieved state-of-the-art generative quality (0.88 bits/dim)" is, by its own current wording, narrowly about generative quality only, so classifiers correctly flagged adding a classification-accuracy metric (a real, same-model result) as introducing a second accomplishment. The hypothesis was that an explicit declared `role` (for example "presenting model summary and model metrics") on `CoreClaimMolecule` would let expansion widen what counts as "the same accomplishment" without weakening the boundary against genuinely different accomplishments.

### What was tried

An experimental `role: Optional[str]` field was added to `CoreClaimMolecule`, wired through claim generation (`tailoring.claims`) and into expansion's marginal-value prompt (`tailoring.expansion`), on branch `bullet-tailoring-claim-role-experiment` (deprecated, never merged).

### Result

Role metadata did NOT resolve the disagreement it targeted and was NOT promoted, per this subphase's own validation gate ("if the comparison is inconclusive... revert to wording-only evaluation"). Findings from 3 independent live runs of the exact real example (adding "Maintained 98.5% classification accuracy on MNIST" to the generative-quality claim above):

- A dedicated single-purpose "same-claim-integrity" classifier returned False in 12/12 calls whenever the fact was actually added (2 of 3 runs; see below) - IDENTICAL whether or not a declared role was supplied. Declaring a broader role made no measurable difference.
- Root cause: a declared `role` is invisible in the rendered claim text. A reader - or a classifier judging only the rendered `claim_text` - has no access to hidden intended-scope metadata; `claim_text` still narrowly asserts generative quality only. Widening the declared role does not widen what the bullet actually SAYS. Only rewriting `claim_text` itself would fix this, and that is explicitly out of Phase 4's bounded scope (no text-authoring field exists on `ExpandedClaimMolecule` by design; see Phase 4 below).
- A secondary, orthogonal finding: the CURRENT single monolithic marginal-value prompt's own add/keep_out decision for this exact candidate was itself unstable across the 3 runs (added in 2/3, excluded in 1/3), while the dedicated single-purpose integrity classifier was perfectly consistent (12/12) on the same underlying judgment whenever it got to evaluate an added fact. This instability, and the fact that a narrow single-purpose classifier was measurably more consistent than the bundled prompt on the identical judgment, is the direct motivating evidence for Phase 3.6 below.

The `role` field, its generation/expansion wiring, and their tests were fully reverted (not merged) rather than kept as unused dead code. This section is retained as a historical record so the same mechanism is not re-attempted without first solving "a declared role is invisible in rendered text" (which would require a text-rewriting step, a larger, separate change).

## Phase 3.6 (experimental): Classifier + Judge Expansion Decisions

### Goal

Replace `expand_claim_molecule`'s single monolithic marginal-value prompt - one call bundling "does this fact add relevant evidence" AND "does it preserve the same accomplishment" AND "should we stop checking more candidates" into one 3-way `add_support`/`keep_out`/`stop` decision - with a small set of narrow, single-purpose classifiers plus a deterministic judge/decision rule, per AGENTS.md's "ask exactly one question per call" principle and the pattern already used by this project's own benchmark scripts (fact support / same-claim integrity / target-skill coverage / clarity as separate classifiers).

### Rationale

This is not a hypothesis - Phase 3.5's live data is direct empirical evidence for it. The CURRENT bundled marginal-value decision was unstable (2/3 add, 1/3 exclude) across 3 identical live runs of the same real candidate/claim pair, while a dedicated single-purpose classifier asking only "does this preserve the same accomplishment" was perfectly consistent (12/12) on the same underlying judgment whenever it was evaluated. Splitting the bundled decision into narrow, single-purpose calls measurably improved consistency on this exact reproducer.

### Tasks

1. Add the flow-based-generative-classifier-plus-classification-accuracy case (the real, reproducible finding from Phase 3.5) as a new, permanent, human-reviewable fixture in `tests/evals/tailoring/expansion/` alongside the existing backend-claim fixture - it deserves a durable regression fixture rather than only living in throwaway experiment scripts. Hard constraint: given the claim's CURRENT wording (generative-quality only), a different-measurement-axis candidate (classification accuracy) must be excluded; rationale references the Phase 3.5 finding.
2. Replace `expand_claim_molecule`'s decision logic with 2 narrow single-purpose classifiers per candidate fact, called in short-circuit sequence (skip the 2nd call once the 1st already rejects, to bound the added cost):
   - `evidences_specific_claim`: does this fact add genuine evidence for the SPECIFIC assertion in `claim_text` (not merely "same project/phase")? Boolean + reasoning.
   - `preserves_same_accomplishment`: if added, would the expanded fact set (core + already-added + this candidate) still read as ONE single accomplishment, not a second, different one? Boolean + reasoning.
   A deterministic judge rule combines them: `add_support` only if BOTH are true; otherwise `keep_out`, always recording the deciding classifier's own reasoning as the exclusion reason (never a generic one).
3. Remove the model-decided `stop` outcome entirely. Since `MAX_SUPPORT_POOL_SIZE` is already small (4), evaluate every candidate in the already-ranked, already-capped pool up to `max_additions`, rather than relying on an LLM's own judgment of when to stop early - trading a small, bounded increase in call count for removing another bundled judgment from a single call.
4. Deterministic tests (fake provider): evidence-fails short-circuits without calling the 2nd classifier; evidence-passes-but-integrity-fails still excludes; both-pass adds; max-additions cap unaffected; the new fixture case behaves correctly under a fake provider forcing the expected verdicts.
5. Live benchmark: rerun the existing Phase 4 fixture case, the new fixture case, and the real project's claims, with repeated trials to directly compare decision consistency against Phase 3.5's recorded baseline (2/3 vs 1/3 add-rate instability on the reproducer).

### Validation Gate

- This is a bounded, cost-visible design change (roughly 2x the LLM calls per candidate versus today, partly offset by removing the `stop` early-exit path) - record call count, latency, and cost alongside the consistency finding, per AGENTS.md's Optimization & Validation Rules.
- Promote out of "experimental" only if repeated live runs show measurably improved decision consistency on the Phase 3.5 reproducer case versus its recorded baseline, without regressing the existing Phase 4 fixture's 3 hard constraints (adjacent-frontend exclusion, subtle-case handling, irrelevant-fact exclusion).

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

## Phase 3.7 (experimental): Claim Mergeability Reframe

### Goal

Phase 3.6's `preserves_same_accomplishment` classifier was anchored to whether an expanded claim still matched the claim's CURRENT literal wording. Live inspection of a real example (a claim narrowly worded around one measured axis of a model, from a project whose own other facts establish that axis and a second one are part of ONE jointly-optimized method) showed this is too rigid: a claim describing one system/method is allowed to BROADEN into a still-single, still-coherent claim covering more than one of that system's own measured dimensions - it does not have to remain the exact original claim to remain ONE claim. Reframe the second classifier around MERGEABILITY (can this fact be folded into a broadened single claim about the same underlying deliverable) rather than literal-wording preservation.

### Hygiene Rule (applies to every classifier prompt already written, not just new ones)

Production LLM prompts must NEVER contain wording copied or closely paraphrased from this repository's test fixtures or benchmark scripts. Phase 3.6's shipped prompts violated this - both classifiers' anchor examples used the real project's own facts (the flow-based-generative-classifier/bits-dim/classification-accuracy example, and the FastAPI/pagination/React-dashboard example that mirrors the Phase 4 fixture almost verbatim). This is corrected as part of this subphase: every anchor example is replaced with a fully invented scenario in a domain that does not appear in any fixture or real project data, and this rule applies retroactively to prompts written before this subphase, not only to new ones.

### Tasks

1. Replace `_PRESERVES_SAME_ACCOMPLISHMENT_SYSTEM_PROMPT` with a mergeability-framed classifier: given a candidate fact, would folding it into a broadened (not merely appended) restatement of the claim still read as ONE coherent accomplishment about the same underlying system/method/deliverable - allowing the claim's own scope to widen to cover an additional measured dimension of that same thing, while still rejecting a fact that describes a genuinely different deliverable or an incoherent narrative combination.
2. Reframe `_EVIDENCES_SPECIFIC_CLAIM_SYSTEM_PROMPT` as a same-underlying-deliverable check: is the candidate fact a result/capability of the SAME system/method/deliverable as the claim (a different measured dimension of that same thing still counts), as opposed to a different deliverable entirely, or a tool/process used to build it rather than a result of it.
3. Keep the classifier + judge structure and short-circuit call sequence from Phase 3.6 (deliverable-identity check first, mergeability check second, AND judge) - this subphase changes the QUESTIONS, not the architecture Phase 3.6 validated.
4. Remove every fixture-derived anchor example from both prompts (see Hygiene Rule) and replace with invented, non-fixture scenarios.
5. Reuse the existing Phase 4 fixture set (adjacent-frontend/subtle/irrelevant) and the Phase 3.6 reproducer fixture (generative-quality-vs-classification-accuracy) to check this reframe does not regress the boundary against genuinely different deliverables, while re-examining whether the reproducer fixture's hard constraint should be loosened now that the underlying claim is expected to legitimately broaden.

### Validation Gate

- Live comparison against the Phase 3.6 reproducer case: the classification-accuracy candidate (same underlying model, a second measured dimension of one jointly-optimized method) should now be judged mergeable, while the existing Phase 4 fixture's genuinely-different-deliverable cases (adjacent-frontend, irrelevant/cross-project) must still be rejected.
- Record repeated-run consistency (per Phase 3.6's own validation gate) rather than a single sample.
- No production prompt may contain wording traceable to a specific fixture or benchmark script after this subphase.

### Result

Implemented and live-validated (3 independent runs). The hygiene rule and the architecture change both held: the existing Phase 4 boundary (adjacent-frontend, irrelevant/cross-project) still rejected consistently 3/3, and no production prompt now contains fixture-derived wording. The reproducer case, however, did NOT flip to mergeable - it was excluded 3/3 runs, but the reasoning is now directly self-diagnosing rather than merely asserting a boundary: every run's `not_mergeable_into_one_claim` verdict explicitly named the actual gap, for example "combining them would... conflate generation and classification tasks **unless the claim specifically framed a joint generative-classification model**; as stated, broadening would produce a conflation of distinct achievements."

This confirms the reframe's mechanism works as designed - it genuinely asks "could this credibly be one broadened claim," not "does it match the literal current wording" - but the reproducer fixture's own core claim (built from only 2 facts: "developed a flow-based generative model" + "achieved state-of-the-art generative quality") never states or implies a joint/hybrid objective anywhere the classifier can see it. Nothing in `expand_claim_molecule`'s inputs gives it access to information beyond the claim's own currently-cited facts, so a classifier judging in good faith correctly has no basis to credit a broader framing that was never asserted.

### Follow-up: `EXPANSION_REASONING_EFFORT` fix (same subphase, same branch)

Further investigation reframed the cause: the reproducer fixture's own CANDIDATE fact text already explicitly states "using the same model" - the classifier did not need to infer or bridge anything; the answer was already stated in its input. Yet the reject-heavy behavior persisted at the project-wide default `reasoning_effort="minimal"`. Live comparison (`tests/tailoring/project_context_experiment_benchmark.py`, exploratory, not merged) isolated the true variable: raising `expand_claim_molecule`'s reasoning effort from `"minimal"` to `"low"` made BOTH classifiers consistently credit the already-stated "same model" fact (3/3 runs, including the no-extra-context baseline), while the existing boundary fixture (adjacent-frontend/irrelevant-fact exclusion) still correctly rejected 3/3 runs at the same effort level - i.e. `"minimal"` was intermittently failing to fully credit information already present in its own input, not failing to bridge a genuinely missing inference; `"low"` fixed that reliability gap without loosening the boundary.

This is PROMOTED to production: `EXPANSION_REASONING_EFFORT = "low"` is now `expand_claim_molecule`'s default (overriding the project-wide `DEFAULT_REASONING_EFFORT = "minimal"` for this module specifically). Live-validated end-to-end 3/3 runs: the reproducer fixture's classification-accuracy candidate now consistently merges (`add_support`), its training-technique candidate still correctly excludes as tooling/process, and the existing Phase 4 boundary fixture still passes all hard constraints. Cost impact is modest (~1100-1200 reasoning tokens per full benchmark run of ~40 calls, versus 0 at `"minimal"`) and recorded alongside the quality result per AGENTS.md's Optimization & Validation Rules.

The original diagnosis above (Phase 3's claim generation sometimes under-scoping claims) remains true and is still an open, separate concern for the reproducer's OWN core claim wording - but it is no longer the reason THIS classifier pair fails on the reproducer's already-explicit candidate fact; that reliability gap is now closed.

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

### Result

Implemented (`tailoring.verification`: `synthesize_proposal`, `verify_proposal`, `repair_proposal`) with a 6-case fixture package (`tests/evals/tailoring/verification/`) and 13 deterministic fake-provider tests. Full suite: 230/230 passing.

Live-validated against all 6 fixture cases (3 repeated trials each) plus the real project's real Phase 3/4 pipeline output (`constrained_optimization_for_generative_classification`, 2 real selected claims, 3 synthesis trials each):

- `clean_pass`, `protected_fact_reuse_unresolvable` (0 LLM calls, deterministic short-circuit confirmed), and `idk_relevance` (idk kept visible, never coerced) all behaved exactly as expected, 3/3 trials.
- `hallucination_repaired`: failed as `hallucination` 3/3, repaired to a fully-supported text, and reverified `pass` in every observed run.
- `bad_flow_unresolvable`: failed as `bad_flow` 3/3. The one permitted repair attempt resolved it by narrowing the claim to just the rate-limiting fact (dropping the React-UI fact entirely) rather than fabricating a connection between the two deliverables - exactly the `allowed_ambiguity` the fixture anticipated, not a violation of it. This surfaced a real, documented gap: `repair_proposal` does not currently prune `AnnotatedProposal.supporting_fact_ids` when a repair narrows scope by dropping a fact, so the reverified/passing proposal's fact-id lineage can become stale (still listing a fact the final text no longer discusses). Left as a known follow-up rather than fixed here, since no cheap, reliable way to detect "fact no longer discussed in text" exists without another LLM call, and the dev plan's own hard rule (no fabricated connection, no new accomplishment) was not violated.
- `bad_wording_repaired`: failed as `bad_wording` 3/3 (correctly caught the fact-id-level protection gap - a different fact ID restating the same protected accomplishment). The repair prompt initially (0/3 in the full benchmark run) only said to "remove or de-emphasize" the duplicated content, which let the model get away with reordering instead of removing it. Fixed in two steps: (1) made the repair instruction explicit that the restated portion must be removed entirely, not reordered/shortened; (2) passed the actual protected baseline bullet text into the repair prompt (previously `_repair_text` never received it), so the model knows precisely what content to remove instead of guessing from the failure-type label alone. Targeted re-testing after both fixes: 3/5 trials reached `pass`, up from 0/3 before. The remaining 2/5 failures are genuine classifier-verdict variance on this specific near-duplicate-paraphrase judgment (observed via `semantic_duplication`/`fact_support` disagreeing on structurally similar reworded text across runs), not a repeat of the same root cause - a candidate follow-up (not applied here) would be raising this classifier's `reasoning_effort` specifically, re-validated the same way `EXPANSION_REASONING_EFFORT` was in Phase 3.7, rather than assumed.
- Real project Part 2: both real claims verified `pass` on all 3 synthesis trials with no repair needed - `synthesize_proposal`'s output read naturally and stayed fully grounded in its cited facts every time.

### Follow-up (open, not applied in this phase)

Two items noted above are left as documented follow-ups rather than fixed now, per the "don't over-claim/don't over-engineer without validation" rule: (1) stale `supporting_fact_ids` after a scope-narrowing repair, and (2) `bad_wording` repair's residual ~40% failure rate on a genuine near-duplicate paraphrase, which may need a dedicated `reasoning_effort` bump for that one classifier once validated the same way Phase 3.7 validated `EXPANSION_REASONING_EFFORT`.

## Phase 5.1: Two-Stage Repair Resolution

### Goal

Replace `repair_proposal`'s single implicit "just rewrite it" call per failure type with an explicit, auditable two-stage resolvability decision made BEFORE any rewrite is attempted:

1. Is this failure resolvable by editing alone - reword the existing text without dropping any currently-cited fact?
2. If not, is it resolvable by removing one or more currently-cited facts - drop them, then reword using only the remaining facts?
3. If neither, the failure is `unresolvable` - discard immediately, no rewrite ever attempted.

This directly closes 2 gaps the Phase 5 live benchmark surfaced and documented as open follow-ups: (a) `bad_flow_unresolvable`'s repair implicitly decided to drop a fact INSIDE the same rewrite call, with no auditable record of that decision and no corresponding update to `supporting_fact_ids` (a stale-lineage bug); (b) `bad_wording_repaired`'s repair prompt had no explicit signal for whether editing alone was even viable versus requiring content removal, plausibly contributing to its residual ~40% failure rate.

### Design

- Two new single-purpose classifiers, generic across all 3 repairable failure types (parameterized by the failure type and its own verification reasoning, not 3 separate prompt families):
  - `resolvable_by_editing_alone` (yes/no/idk + reasoning): can this specific failure be fixed by only rewording, keeping every currently-cited fact?
  - `resolvable_by_removing_facts` (yes/no/idk + reasoning + `fact_ids_to_remove`, only asked if the first is not `yes`): can dropping specific already-cited fact(s) - naming which ones - resolve it, rewording using only the rest? Naming the fact IDs to remove is part of answering "yes" to this one question (a closed selection over an already-fixed, small list of currently-cited facts), not a second independent judgment - consistent with AGENTS.md's guidance that closed, classification-style selections over an already-fixed list are safer to bundle than open-ended extraction.
- Deterministic dispatch based on those two verdicts:
  - `editing_alone = yes` -> edit-only repair prompt, explicitly forbidden from dropping any currently-cited fact.
  - `editing_alone != yes` and `removing_facts = yes` -> remove-facts repair prompt, given exactly which fact IDs to drop; the repaired `AnnotatedProposal.supporting_fact_ids` is deterministically updated (facts removed, not left stale) rather than inferred after the fact from the rewritten text.
  - Neither -> immediately `unresolvable`; no rewrite attempted, no further repair attempts of any kind for this proposal.
  - An `idk` from either gate classifier is treated as `no` for dispatch purposes (never assume repairability from uncertainty), but its reasoning is preserved in the repair step for human review.
- `RepairStep` (schema change, human review required per AGENTS.md Human Review Gates) gains 2 new fields: `resolution: Optional[Literal["edit_only", "remove_facts"]]` and `removed_fact_ids: Tuple[str, ...] = ()`, so lineage records WHICH path was taken and WHAT was dropped, closing the stale-fact-id gap directly instead of leaving it open.
- The fixed repair sequence (`hallucination` -> `bad_flow` -> `bad_wording`) and the one-attempt-per-type bound are unchanged; only the INSIDE of each attempt changes (2-stage resolvability gate first, then dispatch), replacing the single monolithic `_repair_text` rewrite call.

### Fixture Taxonomy Adjustment

Phase 5's fixture case IDs fused two DIFFERENT axes into one compound name - the verification `failure_type` (what `verify_proposal` returns: `hallucination`/`bad_flow`/`bad_wording`/`unresolvable`) and the eventual repair outcome (`_repaired` vs `_unresolvable`). Phase 5.1 makes the repair outcome itself a 3-way, explicitly-decided axis (`edit_only` / `remove_facts` / gate-`unresolvable`), independent of which failure type triggered it, so a single compound case name can no longer stand in for both. Concretely, this reclassifies 2 of Phase 5's existing cases based on what its own live benchmark already showed happening under the hood:

- `hallucination_repaired` stays a clean `edit_only` case: its single cited fact (`fact_002`) is never a candidate for removal (there is nothing else to fall back to), so its fix must be a same-facts reword. No change needed.
- `bad_flow_unresolvable`'s name and hard constraint claimed the case should end up discarded as `unresolvable` - but Phase 5's own live run showed the (then-implicit) repair actually resolved it by dropping the conflicting fact (`fact_004`) and reverifying `pass`. Under Phase 5.1's formal model this is exactly a sanctioned `remove_facts` resolution, not a violation - the case is renamed to reflect an expected `remove_facts` resolution (dropping `fact_004`) ending in `pass`, and a NEW, separate case must be added to cover a genuinely neither-resolvable failure (see Tasks item 1), since this case no longer serves that purpose.
- `bad_wording_repaired`'s successful repairs also, on inspection, effectively dropped `fact_001` from the rendered text (foregrounding only `fact_003`'s latency content) rather than reword while keeping both - so this case is reclassified as `remove_facts` (dropping `fact_001`), not `edit_only`, going forward.
- `protected_fact_reuse_unresolvable` is unaffected - it is the deterministic, non-repairable short-circuit and was never part of this axis.

### Tasks

1. Extend the Phase 5 fixture package (`tests/evals/tailoring/verification/`) per the taxonomy adjustment above: reclassify `bad_flow_unresolvable`'s expected resolution to `remove_facts` (dropping `fact_004`, ending `pass`) and `bad_wording_repaired`'s to `remove_facts` (dropping `fact_001`), add a NEW genuinely neither-resolvable case (expected to short-circuit straight to gate-`unresolvable` without ever calling a rewrite prompt), and record each repairable case's expected `resolution` and (where applicable) `removed_fact_ids` as separate fixture fields rather than folding them into the case name. Update `expected_outcomes.yaml` with these cases' hard constraints and rationale (draft, needs human review).
2. Add `resolution`/`removed_fact_ids` fields to `RepairStep` in `tailoring.models` (draft, needs human review - new schema fields).
3. Implement `_classify_resolvable_by_editing` and `_classify_resolvable_by_removing_facts` in `tailoring.verification`, with fully invented (non-fixture) anchor examples per the Phase 3.7 hygiene rule.
4. Rework `repair_proposal`'s per-attempt body to call the 2-stage gate before any rewrite, dispatch to the appropriate repair prompt, and deterministically update `supporting_fact_ids` on a `remove_facts` resolution.
5. Update deterministic tests (`tests/tailoring/test_verification.py`) to cover: edit-only dispatch, remove-facts dispatch (asserting `supporting_fact_ids` is pruned correctly), immediate-unresolvable dispatch (asserting the rewrite prompt is never called), and idk-treated-as-no for both gate classifiers.
6. Re-run the live benchmark (`tests/tailoring/verification_benchmark.py`) against all fixture cases (repeated trials) plus the real project's real claims, specifically re-testing the renamed `bad_flow`/`bad_wording` cases and the new neither-resolvable case to check whether the explicit resolvability gate improves reliability over Phase 5's baseline (0/3 -> 3/5 for the `bad_wording` case after Phase 5's own prompt fix).

### Validation Gate

- Every `remove_facts` resolution's final `AnnotatedProposal.supporting_fact_ids` must exclude every fact ID the classifier named for removal - mechanically checked, not just inspected.
- The immediate-`unresolvable` path must make zero rewrite/reverify calls beyond the 2 gate classifiers - mechanically checked (call-count assertion), matching the existing zero-call precedent set by the deterministic protected-fact check.
- Compare the renamed `bad_wording` case's repair success rate against Phase 5's baseline (0/3 raw, 3/5 after the first prompt fix) over repeated live trials; report the new rate honestly even if it does not fully resolve the remaining variance.
- No production prompt may contain wording copied from this subphase's own fixtures (hygiene rule, still in force).

### Result

The implementation (models schema change, 2-stage gate, `repair_proposal` dispatch rework, deterministic tests) landed clean (17/17 targeted, 234/234 full suite), but the first live benchmark run against all 4 targeted fixture cases surfaced 2 distinct root-cause bugs, each found by inspecting raw classifier `reasoning` text directly rather than trusting aggregate pass/fail:

1. **Bare failure-type string, no definition.** Both gate-classifier prompts passed the raw `failure_type` string (`"bad_flow"`, `"bad_wording"`) with no accompanying definition, letting the model substitute a plausible-but-wrong generic-English reading of the label - `bad_flow` read as "run-on sentence, fix by splitting into two sentences or adding 'also'/'additionally'" (still leaves two accomplishments, which would fail `same_claim_integrity` again), and `bad_wording` read as "implied causal linkage between two facts, fix by rephrasing to avoid causation" (a phrasing/grammar issue) instead of its actual codebase meaning (substantially restates an existing protected bullet - a semantic-duplication problem). Fixed by adding a module-level `_FAILURE_TYPE_DESCRIPTIONS` dict with precise per-failure_type definitions, interpolated directly into each gate-classifier prompt alongside the bare label, plus 1-2 new anchor examples in each system prompt specifically ruling out both wrong interpretations. After the fix, `bad_flow_remove_facts` and `bad_wording_remove_facts`/`bad_wording_gate_unresolvable`'s editing-gate calls verified 3/3 trials each with the correct `no` verdict and on-target reasoning.
2. **Removing-facts gate missing protected-bullet context.** Even after fix (1), `bad_wording_remove_facts` still chose the WRONG fact to remove (`fact_003`, the genuinely-new latency content) while keeping the duplicative one (`fact_001`) - exactly backwards. Root cause: `_classify_resolvable_by_removing_facts` never received the protected baseline bullet's text at all (unlike `_repair_text`, the actual rewrite call, which already did), so for `bad_wording` it had no way to know WHICH cited fact actually duplicated protected prior work and was guessing. Fixed by threading `protected_baseline_bullet_texts` through both gate classifiers (`_classify_resolvable_by_editing` too, for consistency) and `repair_proposal`'s call sites, referencing it explicitly in the `bad_wording` prompt guidance ("identify which cited fact's content IS one of these, and drop that one"). After the fix, the removing-facts gate verified 3/3 trials correctly naming `fact_001` for removal.

A full re-run of the live benchmark after both fixes showed all 4 targeted fixture resolution checks passing: `hallucination_edit_only` (`edit_only`), `bad_flow_remove_facts` (`remove_facts`, drops `fact_004`), `bad_wording_remove_facts` (`remove_facts`, drops `fact_001`), and `bad_wording_gate_unresolvable` (`unresolvable`). The full deterministic suite remained 234/234 throughout (prompt/context-only changes; `FakeLLMProvider`-based tests are insensitive to prompt content). Lesson generalized: never pass an internal enum/label into an LLM prompt without an accompanying plain-language definition, and never ask a classifier to select among options (here, which fact to drop) without giving it every piece of context a human reviewer would need to make that same selection correctly.

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

### Result

Implementation landed in `src/tailoring/competition.py`: `rank_local_candidates` (deterministic scorer: relevance/support/specificity/local-duplicate-distinctness, never prunes) and `build_global_recommendation` (round-robin global priority walk + pairwise `_classify_primary_proof_overlap` LLM classifier, inclusion-biased so `idk` never blocks a candidate). A new `ProofOverlapDecision` dataclass (`models.py`) records every pairwise judgment for audit. The human-approved fixture (`tests/evals/tailoring/competition/`) deliberately included an unambiguous duplicate pair (shared onboarding-time-reduction accomplishment across 2 projects), a genuinely ambiguous pair (same-shape Kubernetes migration, different systems/outcomes - fixture explicitly allows either verdict), and 2 negative controls (Kubernetes-tag-only overlap that must never be penalized). See the end of this section for the final deterministic-test and full-suite counts, which include several rounds of follow-up fixes below.

The first live benchmark run (3 trials per named pair, real `gpt-5-mini` calls) passed every check on the first attempt - no bugs found, unlike Phase 5.1's 2-round fix cycle:

- Local candidate-set membership: exact match on both projects' `eligible_original_bullet_ids` and `verified_proposal_ids`.
- All 4 overlap pairs verdicted consistently across all 3 trials each: the unambiguous onboarding pair reliably `yes`/`outcome`; both Kubernetes-tag-only negative controls reliably `no`/`responsibility`; the deliberately ambiguous Kubernetes-migration pair reliably `no`/`system_boundary` (a defensible, fixture-allowed outcome - the model consistently weighted differing system boundaries and outcomes over the shared migration shape).
- The full-fixture global recommendation walk passed all 3 mechanical hard constraints, but only exercised 1 real overlap LLM call - because each project's own top-ranked local candidate (`itp_proposal_k8s`, `cad_proposal_dashboard`) already succeeded on the first try, the two duplicate onboarding proposals were locally deprioritized (their `primary_proof` text is shorter/less detailed, scoring lower on the specificity heuristic) and never reached the greedy walk's conflict-resolution path at all.
- To directly exercise the conflict-resolution path against a real LLM call (rather than relying solely on the deterministic suite's canned-response test for this), a targeted Part 4 check was added to the benchmark: force each project's *only* candidate to be its onboarding proposal, bypassing local ranking entirely. This confirmed the greedy filter correctly resolves a genuine head-to-head duplicate live - exactly one of the two onboarding proposals was recommended, with the overlap decision correctly verdicting `yes`/`outcome`.

No prompt or scoring changes were needed after the first implementation. Lesson: a live benchmark passing on the fixture's natural ranking order doesn't guarantee the harder conflict-resolution branch was actually exercised - when a scoring heuristic happens to already deprioritize the conflicting candidates, add a forced/adversarial variant that bypasses ranking to validate that code path directly against real LLM output.

`build_global_recommendation` was subsequently hardened for the general n-projects-vs-m<n-unique-underlying-points case (n projects competing for a slot, but fewer than n genuinely distinct real-world accomplishments among them - e.g. the same achievement written up under 2 different projects). No special-case logic was needed for the core greedy walk itself: once every remaining candidate for an "excess" project is judged to overlap an already-accepted pick, that project simply ends up with no recommendation (`recommended_proposal_id=None`) and an explanatory `recommendation_reason`, never a crash or a forced/incorrect pick. On top of that, a small deterministic sanity net (`_find_duplicate_recommendation_warnings`) was added as a second, cheap line of defense: it flags (via a plain warning string, never blocking anything) any 2 different projects whose FINAL recommended proposals have exact-text-identical `primary_proof` after normalization - catching the case where the LLM overlap classifier itself returns an unexpected `no`/`idk` for text that is, on its face, the same string. `build_global_recommendation` now returns a 3rd value, `duplicate_warnings: List[str]`, and `write_default_resume_recommendation_json` includes it in the written artifact. 3 new deterministic tests cover this (2 for the sanity net, 1 for a live 3-projects/2-unique-points scenario), and the live benchmark gained a "Part 5" check reusing the fixture's duplicate onboarding pair alongside a 3rd, genuinely distinct synthetic project - passing on first try (exactly one onboarding duplicate recommended, the 3rd project's distinct recommendation untouched, zero duplicate warnings since the classifier already correctly excluded the loser).

PR #9 review comments (all fixed): (1) the overlap classifier's positive anchor example cited 2 dimensions (`responsibility` AND `outcome`) despite the prompt requiring exactly ONE primary dimension - reworded to cite only `responsibility`; (2) `build_global_recommendation` treated a `proposal_id` missing from `proposals_by_id` as an empty-proof, automatically-accepted candidate instead of a data-integrity problem - fixed to skip it (recording why) with zero wasted overlap-classifier calls, and simplified the now-guaranteed-safe `accepted` list lookup accordingly; (3)/(4) the `FakeLLMProvider` test helper allowed emitting `primary_dimension=None`, which violates `_OVERLAP_JSON_SCHEMA`'s always-required enum - the one call site relying on this (the `idk` test) was fixed to use a real dimension, and the helper's type hint was tightened to require one; (5) this Result section's test counts had already gone stale mid-paragraph (`12/12`/`246/246` vs. the later `249/249`) - replaced with a single final count below rather than pinning intermediate numbers that will drift again. 2 new deterministic tests were added for the missing-`proposal_id` fix (skip-and-explain; fall-through to the next real candidate). Full deterministic suite after all of the above: 251/251.

## End-to-End Integration Validation (post-Phase 6)

After Phase 6 merged, Phase 7 was pushed down to instead run the first genuine multi-phase integration test: chaining Phases 0-6 for one real project against one real posting and inspecting the actual generated output, per AGENTS.md's guidance to reserve end-to-end tests for validating the integrated workflow after its modules pass isolated tests.

`tests/tailoring/end_to_end_benchmark.py` chains every stage (fact/baseline load, requirements, triage, retrieval, claim generation/ranking, expansion, synthesis/verification/repair, local ranking, global recommendation) for project `benchmark_driven_llm_workflow_orchestration` against the real `llm_ml_infra` posting fixture, writing each stage's real production artifact (via each module's own writer, not a synthetic blob) to a timestamped `build/tailoring_e2e_runs/<project>__<posting>__<timestamp>/` directory.

The first live run completed end to end with no crash (34 reasoning-tier calls, ~138s): triage correctly kept the one FastAPI-relevant bullet and deprioritized the other 3 (LLM-workflow bullets only nice-to-have relevant to this posting); retrieval pooled 20/67 facts; 5 claims were generated and 2 selected; expansion added no support facts to either (both already at/over the 2-line verbosity-prefilter budget from claim text alone - an expected but noteworthy limitation given the prefilter runs before a page-constraint policy exists); one proposal passed verification after a wording repair, the other failed verification as `hallucination` and could not be repaired (the repair rewrite returned the input text unchanged and re-failed) - it was correctly discarded rather than surfaced, i.e. the pipeline's fail-closed design worked as intended. The single verified proposal became the recommended candidate with no overlap conflicts (degenerate single-project case).

**Open finding (not yet fixed - flagged for a follow-up decision, not patched unilaterally, since `verification.py` is shipped/merged Phase 5 production code and any prompt change needs the same fixture-first + human-review rigor Phase 5.1 followed):** the `hallucination` verdict above was re-run 4 more times in isolation (same proposal text, same cited facts, zero pipeline re-run cost) and came back `yes` 4/4 with near-identical reasoning every time - this is a real, reproducible classifier behavior, not single-sample noise. Root cause: the proposal used ordinary first-person resume-bullet phrasing ("Designed and implemented..."), but every cited fact atom is worded in third-person/passive, system-centric voice ("The parser pipeline extracts skills into structured records.") with no explicit sentence attributing the work to the candidate. `_FACT_SUPPORT_SYSTEM_PROMPT` reads this literally and flags the authorship framing itself as an unsupported claim. Since fact atoms across this entire codebase are consistently authored this way (describing what a system does, not "I built..."), this gap is not specific to this one bullet or project - it would likely recur for any synthesized proposal citing similarly-worded facts. Candidate fixes for a future pass: (a) make the fact-support classifier's prompt explicitly treat first-person authorship/action-verb framing as an implicit, always-supported convention of the resume-bullet genre rather than a factual claim needing its own citation, or (b) adjust fact-atom authoring guidance to include explicit ownership language. Left as an open, documented finding rather than an immediate patch.

Inspecting the real `verification_report.json` this run wrote surfaced a genuine bug: `verification_results_to_dicts` (`src/tailoring/verification.py`) never serialized `RepairStep.resolution`/`removed_fact_ids` - the Phase 5.1 fields added specifically to make a repair's fact-dropping decisions auditable - so every persisted verification report was silently missing them, even though the in-memory objects had the correct values. No prior deterministic test caught this because existing tests only asserted against in-memory dataclass fields, never round-tripped through the JSON writer. Fixed by adding both fields to the dict conversion; added `VerificationResultsToDictsTest` (2 tests) to close the coverage gap. Full deterministic suite: 253/253. The `verification_report.json` already on disk for this specific run predates the fix and is missing those 2 fields; the run's other artifacts and conclusions are unaffected.

### Open design finding: generated claim/proposal text reads as documentation, not resume bullets (NOT yet actioned)

This one real run is explicitly **not treated as a pass/fail evaluation fixture** - bullet quality is multifaceted/holistic, not a fixed template a mechanical check can score. It is, however, a genuine qualitative finding worth recording before any future prompt work on `tailoring.claims`/`tailoring.verification`'s generation/synthesis prompts.

Human review of `core_claim_molecules.json` and `annotated_proposal_set.json` from the real run above: the generated `claim_text`/`proposal_text` read like technical documentation (a flat, exhaustive enumeration of what a system does) rather than resume bullets, and do not surface the claim's own `target_skills` as recognizable keyword terms in the actual wording - e.g. claim_04's text ("Implemented observability and persistence for parser runs: stage- and batch-level progress reporting, writing run metrics as artifacts, persisting run history, and retrieving stored posting text for historical runs.") never mentions "FastAPI," "JSON," or "structured logging" even though all three are listed in that same claim's own `target_skills`.

Recorded framework (verbatim from human review) for what makes a resume bullet effective, to guide a future prompt redesign rather than a fixed template:

- Three ingredients, mixed in different proportions per bullet: **what it is** (the system/product/principle), **how it was done** (methods/tools/technical choices), **why it mattered** (outcome/benefit/capability).
- Two example archetypes are really just different weightings of those three ingredients, not two rigid templates:
  1. *Useful product/principle + methods/technologies* (best when the value is in the architecture/robustness/design itself - "why it mattered" is implicit in the principle). Example: "Built a robust staged LLM workflow using grounded extraction, validation, and human-in-the-loop review."
  2. *Outcome + support* (best when there's a clear measurable/qualitative payoff - the outcome leads and methods support it). Example: "Improved extraction F1 from X to Y using staged orchestration and benchmark-driven iteration."
- A strong bullet has a clear **center of gravity** - it proves ONE of: capability ("I built a hard system well"), impact ("I improved something important"), judgment ("I chose the right tradeoff"), or scale ("I handled a lot of data/users/complexity") - rather than trying to flatly state everything a system does.
- A good bullet is specific enough to sound real, broad enough to sound important, and compact enough to read fast.
- One-line summary rule: **good resume bullets = evidence-backed technical action, framed around either impact, robustness, or scope** - typically action + system + relevant keywords, plus either a measurable outcome or a clear product/engineering benefit.
- Implication for the generator: its job is not to invent an outcome for every claim, nor to enumerate every contributing fact - it is to DECIDE which of the three ingredients (what/how/why) should be the center of gravity for each individual claim, and to weight the others accordingly, while still surfacing the claim's own target-skill terms as literal keywords in the wording.

This is left as a documented, unactioned design note - no prompt in `tailoring.claims` or `tailoring.verification` has been changed based on it yet. A future pass to act on it would need to go through this project's normal fixture-first + human-review process (like Phase 5.1's own hallucination-classifier work), since it touches merged, production prompt design.

Phase 7 (below) remains the next planned phase but is deprioritized relative to this validation work per explicit instruction; it has not been started.

## Phase 7: Human Selection and Rendering Integration (deferred - not yet started)

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