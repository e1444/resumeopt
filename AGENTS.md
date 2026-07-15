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

## Implementation Boundaries
- Reuse existing modules instead of creating parallel copies of core infrastructure.
- `src/llm/` is the provider layer; parser work should import it, not reimplement it.
- Prefer established packages and libraries when they materially simplify the task or improve correctness.
- If a package is the right tool, install it and wire it in explicitly rather than recreating the behavior by hand.
- If a dependency is missing, add it to the proper dependency manifest before proceeding.
- If YAML needs to be read or written, use a YAML library such as `PyYAML` and add the dependency explicitly rather than hand-parsing line by line.
- Do not create a deterministic prototype that bypasses the documented architecture unless the task explicitly calls for a temporary test harness.

## Task Tracking Policy
For any task that spans more than one step, maintain a task list for the full development cycle, not just isolated subtasks.

Minimum expectations:
- create the task list before the first substantive change
- keep task states current as work moves from `not started` to `in progress` to `done`
- use the task list to track planning, implementation, validation, and cleanup
- keep the list coarse enough to be useful, but detailed enough to show progress and blockers

The task list is required for multi-step work and should remain visible and up to date until the work is complete.

## Current Scope
The first implementation focuses on the skills section only.

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
- `schemas/` for structured schema examples
- `data/` for canonical caches and source data
- `tests/llm/` for provider tests
- `tests/evals/` for job posting samples and expected outputs
