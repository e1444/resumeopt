---
name: "Start Bullet Tailoring"
description: "Use when: starting the fact-grounded bullet-tailoring project from its Phase 0 fixture and contract baseline."
argument-hint: "Optional Phase 0 constraint or question"
agent: "agent"
---

Start the fact-grounded experience-bullet tailoring project.

Read [the repository instructions](../../AGENTS.md), [the development plan](../../docs/agent/BULLET_TAILORING_DEV_PLAN.md), and [the architecture diagram](../../docs/proposals/BULLET_TAILORING_ARCHITECTURE.md). The development plan is authoritative.

Begin at Phase 0 only. Create the reviewable fixture package, source-data and artifact contracts, expected outcomes, permitted ambiguities, and evaluator rationale required by that phase. Do not implement production pipeline behavior, add LangGraph, or change dependencies until the Phase 0 schemas, fixtures, and ground truth have been reviewed and approved.

Work incrementally:

1. Inspect the current Git state and create a concise feature branch from current `main` before making substantive changes. Preserve unrelated user changes.
2. Create and maintain a coarse task list covering fixture design, contract work, validation, and review readiness.
3. Persist or fixture every upstream input that a Phase 0 module consumes. Test modules directly from those inputs; do not regenerate upstream artifacts or exercise the current module through the full pipeline.
4. Use existing parser, matcher, renderer, and LLM layers as integration boundaries; do not duplicate them.
5. Run focused validation after each substantive change. Stop for human review once the Phase 0 fixture package and schema/ground-truth contracts are ready; do not advance into production implementation without approval.
6. After approval and validation of a coherent change, inspect the diff, create a focused commit, and open a pull request that includes validation results. Merge to `main` only after explicit user approval.

User-provided focus for this run: ${input:Optional Phase 0 constraint or question}