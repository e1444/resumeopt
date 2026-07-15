# Resume Tailoring Agent Development Plan

## Goal
Implement a skills-only resume tailoring pipeline that can read a job posting, match it against the skills cache, generate a LaTeX skills section, and validate the result.

## Active Task List
- [done] Confirm the skill cache schema and record the contract in `SPEC.md`
- [done] Define the posting extraction output schema
- [done] Define the validation output schema
- [done] Review the draft schemas against sample fixtures and update examples if needed
- [done] Human review of the schema drafts before they become authoritative
- [done] Add deterministic tests for the LLM provider factory
- [done] Implement Phase 3 parser scaffold in `src/parse_posting.py` with deterministic default behavior and an LLM-backed path that reuses `src/llm/`
- [done] Add parser tests in `tests/parse_posting/` for deterministic extraction, cache matching, and schema-shaped outputs
- [done] Add a Python dependency manifest and include YAML support explicitly for structured cache and fixture loading
- [done] Run parser and existing provider tests, then record validation outcomes in this task list
- [done] Refactor LLM parser flow to pass chunk plus cache context and return cache-constrained matched skills directly
- [done] Add parser tests for LLM cache-constrained matching and rejection of non-cache canonical names
- [done] Remove separate Phase 4 and fold matching responsibilities into parsing + validation flow
- [done] Add Phase 5 validation functions for selected skills (unsupported, weak, grounding, size constraints)
- [done] Add Phase 5 tests for validation pass/fail behavior
- [done] Add Phase 5 LLM-assisted grounding validation for edge cases (for example ipynb -> jupyter)
- [not started, low priority] Convert `data/template.tex` from an example into the final working template near the end of the project

## Guiding Strategy
Build in small, inspectable layers:
1. data definitions
2. LLM wrapper
3. posting parsing
4. ranking and validation
5. LaTeX rendering
6. PDF validation

## Phase 1: Data and Contracts
### Tasks
- Confirm the skill cache schema.
- Define the posting extraction output schema.
- Define the validation output schema.
- Decide how canonical names, aliases, and related terms are represented.
- Have a human review the proposed schemas before they become the project contract.

### Deliverables
- `data/skills.yaml`
- schema notes in `SPEC.md`
- example extraction outputs

### Validation
- Sample entries can be parsed without ambiguity.
- Canonical names are unique.

## Phase 2: LLM Abstraction
### Tasks
- Keep provider-specific logic isolated.
- Support OpenAI first.
- Preserve the option for Anthropic and Ollama later.
- Read API keys from environment variables.

### Deliverables
- `src/llm/`
- provider factory
- basic provider tests

### Validation
- OpenAI text and JSON calls succeed.
- Provider-specific code is not hard-coded into higher layers.

## Phase 3: Posting Parsing
### Tasks
- Use a class-based parser design with a shared interface.
- Support at least two implementations: deterministic and LLM-backed.
- Keep the parser responsible for chunking, filtering, extraction, normalization, and validation orchestration.
- Import and reuse the existing `src/llm/` package; do not create a second LLM implementation under the parser layer.\
- Split job postings into chunks.
- Filter out useless information and keep only chunks that are likely to contain skill-relevant content.
    - Use an LLM to split the posting into chunks.
    - Verify that the chunks combine to reproduce the original posting. This may require normalization of whitespace and punctuation.
    - Use an LLM to filter out chunks that are unlikely to contain skill-relevant content.
- Extract matched skills per chunk using cache-aware prompts.
    - Provide each chunk and the canonical skills cache to the LLM during matching.
    - Require LLM outputs to return matched skills using canonical names from the provided cache.
    - Reject LLM matches that do not map to canonical cache entries during deterministic post-processing.
    - Follow the schema for parsed posting output, including extracted terms, matched skills, and optional validation artifacts.
- Normalize the extracted text.

### Schema Checkpoint
- Define the expected schema for one parsed line or chunk.
- Have a human verify and iterate on that schema before using it downstream.

### LLM Call Conventions
- Use `call_json` for chunk splitting, chunk filtering, and skill extraction when the response should be structured.
- Use `call` only when the output is intentionally freeform or the provider cannot return a reliable structured payload.
- Keep `temperature` low for extraction and filtering prompts.
- Include a `system_prompt` that tells the model to return only the expected format.
- Keep deterministic post-processing steps outside the prompt whenever possible.

### Deliverables
- `src/parse_posting.py`
- parsing output structure

### Validation
- Relevant lines are extracted consistently.
- Obvious noise is excluded.
- The parser does not recreate `src/llm/` or other shared infrastructure.
- YAML is parsed with a library, not ad hoc line processing.

## Phase 5: Validation Layer
### Tasks
- Reject duplicates.
- Reject unsupported or weak matches.
- Verify that selected skills appear in the posting.
- Use an LLM grounding check for edge cases where deterministic string checks are insufficient (for example ipynb indicating jupyter).
- Enforce size/shape constraints on the skills section.

### Deliverables
- validation functions
- sample validation reports

### Validation
- Invalid outputs fail loudly.
- Valid outputs pass without manual intervention.
- Edge-case grounding can be confirmed via a constrained `call_json` validation step.

## Phase 6: LaTeX Rendering
### Tasks
- Insert selected skills into the template.
- Keep the template ATS-friendly.
- Render the `.tex` file to PDF.

### Deliverables
- template injection step
- render command or script

### Validation
- Generated LaTeX compiles cleanly.
- Output fits the expected page layout.

## Phase 7: PDF Validation
### Tasks
- Confirm page count.
- Confirm no rendering failures.
- Spot-check layout and readability.

### Deliverables
- PDF validation step
- regression samples

### Validation
- One-page target is respected.
- Final PDF is usable as a resume.

## Recommended File Layout
- `docs/agent/SPEC.md`
- `docs/agent/DEV_PLAN.md`
- `src/llm/`
- `src/parse_posting.py`
- `data/skills.yaml`
- `schemas/`
- `tests/llm/`
- `tests/parse_posting/`
- `tests/evals/`

## Naming Conventions
- Use `schemas/` for schema examples and draft contracts that need human review.
- Use `tests/llm/` for provider-level tests.
- Use `tests/parse_posting/` for parser and matching tests.
- Use `tests/evals/` for job posting fixtures and expected outputs.
- Prefer descriptive filenames that reflect the contract or behavior being tested, for example:
	- `parsed_posting_line.example.yaml`
	- `sample_job_posting.txt`
	- `sample_expected_skills.yaml`

## Task Tracking Policy
For the duration of any multi-step development effort, maintain a task list that covers the full cycle of work.

Minimum expectations:
- create the task list before the first substantive implementation step
- keep task states current as work moves from `not started` to `in progress` to `done`
- use the task list to track planning, implementation, validation, and cleanup
- keep the list coarse enough to be readable, but detailed enough to expose blockers and progress

The task list remains active until the work is complete and reviewed.

## Definition of Done for This Phase
This phase is done when:
- a job posting can be ingested
- relevant skills can be extracted and matched
- the skills section can be generated in LaTeX
- the PDF renders successfully
- validation catches obvious errors before output
- the main schemas and test expectations have been reviewed by a human and iterated if necessary