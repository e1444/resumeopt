# Resume Tailoring Agent Development Plan

## Goal
Implement a skills-only resume tailoring pipeline that can read a job posting, match it against the skills cache, generate a LaTeX skills section, and validate the result.

## Completed Milestones
The detailed, chronological build history (every iteration, benchmark, and bug fix) has grown too long to usefully keep inline here and has been condensed into this summary; full narrative detail lives in repo memory (`/memories/repo/parsing.md`, `layout.md`, `testing.md`) and git history. The current end-to-end architecture is documented authoritatively in `README.md`, not in the historical notes below.

- **Phase 1-2 (data contracts, LLM abstraction)**: skill cache schema, parsed-line/validation schemas, and the `src/llm/` provider abstraction (OpenAI/Anthropic/Ollama, structured JSON outputs) were defined and human-reviewed.
- **Parser architecture went through several full generations before settling**: a first deterministic-only parser, then LLM-backed multishot/single-shot variants, then a self-consistency-voted multi-classifier extraction pipeline (`orchestra_single_shot` + `parallel_extraction`), each benchmarked against the others and promoted/deprecated based on measured precision/recall. **All of these were eventually superseded** by the current architecture: a staged pipeline (Stage 0 posting summary -> chunking -> Stage 1 extraction -> Stage 2 categorization -> Stage 3a/3b atomicity+redundancy) living in `src/parser/` (`summary.py`, `extraction.py`, `categorization.py`, `keyword_atomicity.py`, `redundancy.py`, `pipeline.py`, `factory.py`) and `src/chunker/` (LLM-based chunking with a deterministic regex fallback). See `/memories/repo/layout.md`'s "Final architecture after the parser/matcher/chunker refactor" entry for the full rationale.
- **Matching was split into its own top-level package**, `src/matcher/` (`ExactAliasMatcher`, `SemanticMatcher`, `LLMGroundingMatcher`, `EmbeddingCache`), decoupled from parsing/extraction, operating only on in-memory `SkillRecord` sequences.
- **Rendering pipeline (Phases 6-8)**: LaTeX template injection, `pdflatex` rendering, PDF validation (`validate_pdf`), and full run telemetry/observability (`run_metrics.json`, per-call LLM usage logging) were implemented and are stable.
- **Token/cost optimization pass**: `reasoning_effort="minimal"` tuning, Stage 0.5 chunk screening (cheap-model pre-filter), and batched multi-chunk extraction/categorization calls cut per-run token usage by roughly 83% on the benchmark posting with an accepted, reviewed recall trade-off. Full detail and the exact validated numbers are in `/memories/repo/parsing.md`.
- **Skill cache schema simplified**: the `related` tier (which conflated true synonymy with specialization and fuzzy evidence) was removed entirely; `aliases` is now reserved strictly for true spelling/naming variants, and distinct technologies (e.g. `sql` vs. `relational databases`) get their own canonical cache entries instead of being folded together.
- **Dynamic skill-section grouping + fit-to-budget rendering**: the skills section is no longer grouped into a fixed 3-category taxonomy - an LLM proposes 2-4 posting-tailored section names, and an iterative render-compile-validate loop trims the lowest-ranked skill and retries until the rendered PDF fits its line budget, replacing the old fixed top-N truncation approach.

## Currently Active
- [in progress] Frontend interface - see `docs/agent/FRONTEND_DEV_PLAN.md` for the full plan. Phases 1-6.5 (backend API scaffold, MVP UI, run inspection/QOL, config picker + cache search, UI/UX polish, real per-stage and per-batch progress) plus a Phase 7 bug-fix/QOL pass (skill-name capitalization, always-include skills, inline alias editing, a progress-bar substage-weighting fix, and the first frontend unit tests) are done. See `/memories/repo/layout.md` for the full narrative detail.

## Known Bugs
None currently open. The previously tracked acronym-mangling bug (`sql` rendering as `Sql`) was fixed - see the Phase 7 entry in `docs/agent/FRONTEND_DEV_PLAN.md` (`render_resume.capitalize_skill_name` replaced the old naive `_display_skill_name`).

## Guiding Strategy
Build in small, inspectable layers:
1. data definitions
2. LLM wrapper
3. posting parsing
4. ranking and validation
5. LaTeX rendering
6. PDF validation

*Note: the phase sections below (1-8) describe the ORIGINAL implementation plan and are kept for historical/design rationale. Some file paths they mention (`src/parse_posting.py`, `tests/parse_posting/`) no longer exist - see "Completed Milestones" above and `README.md` for the current, authoritative module layout.*

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
 - Extract raw candidate skills from the posting in one batch with the LLM.
    - Match extracted raw skills to canonical cache entries deterministically after extraction.
    - Preserve unmatched extracted terms in a `missing_skills` intermediate artifact for cache curation.
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
- Sort selected skills into the section set (`Languages`, `ML & Data`, `Tools`) using an LLM with deterministic fallback.
- Allow the LLM to omit non-relevant sections and render only active sections for the target role.
- Keep the template ATS-friendly.
- Render the `.tex` file to PDF via a bash `pdflatex` command.
- For each run, write artifacts to `build/[run_name]/aux` and write logs to `build/[run_name]/logs`.

### Deliverables
- template injection step
- render command or script
- section-grouped skills formatter for template insertion

### Validation
- Generated LaTeX compiles cleanly.
- Output fits the expected page layout.
- Render path is covered by automated tests, including placeholder replacement and PDF generation.
- Run logs include pipeline stage artifacts and pdflatex command/stdout/stderr/engine logs.

## Phase 7: PDF Validation
### Tasks
- Confirm page count.
- Confirm no rendering failures.
- Ensure that the Skills section is at most 3 lines long.

### Deliverables
- PDF validation step: `validate_pdf(pdf_path, max_pages=1, max_skills_section_lines=3)` in `src/render_resume.py`, using `pypdf` to read the rendered PDF's page count and extract page-1 text between the `SKILLS`/`EXPERIENCE` section headers to count rendered skills-section lines.
- Regression samples: `tests/render/test_render_resume.py` covers a passing case, an over-page-count case, an over-line-count case, and a missing-file case.

### Validation
- One-page target is respected.
- Final PDF is usable as a resume.
- `validate_pdf` output is written to `build/[run_name]/logs/pdf_validation.json` on every run, and a non-pass status fails the pipeline run.

## Phase 8: Observability and Run Telemetry
### Tasks
- Define and log stage timing metrics across the full pipeline.
- Log parse and validation counts (record counts, selected skill counts, issue counts).
- Log run artifact metadata (file counts and output sizes).
- Log estimated token usage metrics where authoritative provider token usage is not available.
- Persist telemetry as inspectable run artifacts under `build/[run_name]/logs`.

### Deliverables
- `run_metrics.json` in run logs
- additional `pipeline.log` timing and token-estimate entries
- helper tests for telemetry estimation utilities

### Validation
- Each run emits `run_metrics.json` with timings, counts, artifact stats, and estimated token usage fields.
- Telemetry values are deterministic and reproducible from pipeline outputs.

## Recommended File Layout
- `docs/agent/SPEC.md`
- `docs/agent/DEV_PLAN.md`
- `src/llm/`
- `src/parse_posting.py`
- `data/skills.yaml`
- `tests/llm/`
- `tests/parse_posting/`
- `tests/evals/`

## Naming Conventions
- Schema contracts live inline as JSON schemas next to the code that uses them (e.g. `_CATEGORY_JSON_SCHEMA` in `src/parser/categorization.py`) rather than as standalone example files under a `schemas/` directory - more convenient for structured-output API calls and keeps the contract in lockstep with its consumer.
- Use `tests/llm/` for provider-level tests.
- Use `tests/parse_posting/` for parser and matching tests.
- Use `tests/evals/` for job posting fixtures and expected outputs.
- Prefer descriptive filenames that reflect the contract or behavior being tested, for example:
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