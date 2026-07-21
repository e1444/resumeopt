# Agent Instructions for ResumeOpt

## Start Here
Any agent working in this repository must read this file before taking further action.

## Purpose
This repository builds a resume tailoring pipeline that parses job postings, matches skills against a canonical cache, and renders ATS-friendly LaTeX output.

## Working Rules
- Prefer small, incremental changes.
- Keep canonical data separate from generated output.
- Treat schemas, eval sets, and final deliverables as human-reviewed artifacts when they are vague or newly introduced.
- Validate each stage before moving to the next.
- Do not widen scope unless the current step is stable.

## Git Workflow
- Develop each feature or fix on a dedicated branch created from the current `main` branch; use a concise, task-specific branch name.
- After completing and validating a coherent change, inspect the diff and create a focused commit with a descriptive message. Do not include unrelated user changes in that commit.
- Create a pull request containing the branch and validation results for user review.
- Merge an approved pull request back into `main` only after the user explicitly approves it. Do not merge or commit when the user instructs otherwise.

## Implementation Boundaries
- Reuse existing modules instead of creating parallel copies of core infrastructure.
- `src/llm/` is the provider layer; parser work should import it, not reimplement it.
- Prefer established packages and libraries when they materially simplify the task or improve correctness.
- If a package is the right tool, install it and wire it in explicitly rather than recreating the behavior by hand.
- If a dependency is missing, add it to the proper dependency manifest before proceeding.
- If YAML needs to be read or written, use a YAML library such as `PyYAML` and add the dependency explicitly rather than hand-parsing line by line.
- Do not create a deterministic prototype that bypasses the documented architecture unless the task explicitly calls for a temporary test harness.
- LLM classifier calls should ask exactly one question per call, especially when using cheap/lightweight models - split multi-part judgments into separate single-purpose calls (e.g. boolean classifiers) rather than combining them into one call with multiple fields/decisions. Cheap models are less reliable at multi-task prompts, and single-purpose calls are easier to test, tune, and diagnose independently. If two questions are both needed, prefer running them as separate concurrent calls over merging them into one prompt.
- For the bullet-tailoring pipeline, use LangGraph as the initial orchestration baseline. Create and test the complete graph's typed node and artifact interfaces before implementing the behavior of any individual stage; node bodies may initially raise a clear not-implemented outcome, but graph routing and state ownership must be explicit.

## Optimization & Validation Rules
- Any cost/latency optimization (model downgrade, batching, caching, screening, reasoning-effort tuning) must be validated empirically before becoming a default: compare token/call counts AND inspect term-level/output-quality impact, not just aggregate count deltas. Token count is not a proxy for dollar cost - per-token pricing differs across models, so do not conflate "fewer tokens" with "cheaper" without checking real pricing.
- Default reasoning-tier LLM calls to the lowest `reasoning_effort` that empirically preserves output quality (this pipeline uses `"minimal"`) rather than leaving it unset, since unset/default effort can silently spend hidden reasoning tokens with no quality benefit for narrow, single-purpose judgments.
- Batching multiple items into one LLM call is not universally good or bad - its effect on recall/precision is input-dependent and must be benchmarked per use case. Closed, classification-style batches (e.g. yes/no over an already-fixed list) are safer to batch than open-ended, generative extraction.
- For outputs with a hard physical/rendering constraint (e.g. a LaTeX page or line budget), validate by producing and checking the real compiled artifact rather than estimating analytically - line-wrapping and layout are too fragile to predict from character counts alone.
- Prefer adaptive, input-tailored categorization (e.g. LLM-proposed section names) over a fixed, hardcoded taxonomy when the fixed taxonomy would need to generalize across varied inputs.
- Reuse already-computed context (e.g. an earlier stage's summary/classification) as a ranking or relevance signal instead of issuing a new LLM call for a redundant judgment.
- When a test stubs/fakes an LLM provider, keep the fake's trigger condition (prompt substring match) and response shape in lockstep with the real prompt/schema it mimics - a stale fake silently falls through to a fallback code path and can mask real regressions or produce misleading failures.
- Generation-quality testing uses a series of at least four small, single-purpose LLM classifiers, one requirement per call (for example: fact support, single-accomplishment coherence, local novelty, or position/context preservation). Record the model, prompt version, verdicts, disagreements, call count, latency, tokens, and estimated cost. Classifier agreement is evidence for review, not ground truth: preserve dissent and use human-reviewed fixtures to resolve ambiguity.

## Fixture-First Phase Method
- Before implementation work begins for every pipeline phase, create the phase's fixtures, inputs, expected outputs where they are separable, allowed ambiguous outcomes, and requested rationale. This fixture package is the phase's first deliverable.
- Each module must persist or fixture every input it consumes from an upstream module, then use those stored inputs in its own tests. A module must not regenerate an upstream artifact merely to exercise itself.
- Expected outputs must state the rationale an evaluator should use, so an agent can compare its assumptions rather than only matching labels. For example, a frontend posting paired with a research/ML bullet may expect `candidate_for_replacement`; ambiguous cases may list multiple permitted labels and explain why. For open-ended generation, fixtures define only separable hard constraints and allowed ambiguity (for example, frontend and backend facts must not be merged into one claim); they must not pretend to supply one exhaustive correct set of generated claims.
- Do not implement a phase's production behavior until its fixture package and expected outcomes have been reviewed. Update the fixtures and their rationales whenever the contract changes.
- LLM-generated outputs are evaluated against the separable fixture constraints and the multi-classifier protocol above, then inspected at the term/claim level before any default is promoted. For open-ended generation, the classifiers are the primary quality measure.
- Test each current module directly with its persisted or fixture inputs; never validate it by rerunning the same input through the full pipeline. Reserve end-to-end tests for validating the integrated, completed workflow after its modules have passed isolated tests.

## Task Tracking Policy
For any task that spans more than one step, maintain a task list for the full development cycle, not just isolated subtasks.

Minimum expectations:
- create the task list before the first substantive change
- keep task states current as work moves from `not started` to `in progress` to `done`
- use the task list to track planning, implementation, validation, and cleanup
- keep the list coarse enough to be useful, but detailed enough to show progress and blockers

The task list is required for multi-step work and should remain visible and up to date until the work is complete.

## Current Scope
The current production implementation focuses on the skills section. Experience-bullet tailoring is a separate, planned workflow described in `docs/proposals/` and `docs/agent/`.

## Required Behavior
- Preserve repository structure and keep files easy to inspect.
- Use deterministic logic where possible.
- Use LLMs for extraction and judgment, not uncontrolled generation.
- Keep intermediate artifacts around for debugging.
- Prefer explicit schemas for parsed outputs and validation results.

## Testing Expectations
- Add or update tests when behavior changes.
- Keep provider tests separate from parsing and evaluation tests.
- Use eval samples to catch regressions in matching and validation.

## Human Review Gates
Human review is required for:
- new or changed schemas
- ambiguous parsed-line output schemas
- eval ground truth
- final output format changes
- final resume deliverables

## Recommended Layout
- `docs/agent/` for agent-facing specification and plans
- `data/` for canonical caches and source data
- `tests/llm/` for provider tests
- `tests/evals/` for job posting samples and expected outputs
