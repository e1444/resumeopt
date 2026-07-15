# Resume Tailoring Agent Spec

## Handoff Rule
Any agent taking over this repository should read `AGENTS.md` before proceeding with this spec or the development plan.

## Purpose
Build a resume tailoring pipeline that takes a job posting, extracts role-relevant skills, matches them against a curated skill cache, and produces an ATS-friendly LaTeX resume.

## Primary Goal
Given a job posting, generate a tailored resume that is:
- faithful to the source data
- readable by humans
- structurally simple enough for ATS parsing
- easy to validate and iterate on

## Scope
This first version focuses on the skills section only.

In scope:
- job posting ingestion
- splitting the posting into useful chunks and filtering out useless information
- skill extraction from posting text
- skill normalization and matching against `skills.yaml`
- relevance scoring and ranking
- generation of a skills section in LaTeX
- output validation

Out of scope for now:
- experience/project rewriting
- automatic bullet rewriting
- cover letter generation
- application submission automation

## Inputs
### Job Posting
Raw plain-text posting content.

### Skill Cache
YAML file containing canonical skills and metadata.

Expected fields:
- `name`
- `aliases`
- optional `related`
- optional metadata for ranking

#### Skill Cache Schema Draft
- The cache is a YAML sequence of skill records.
- Each record MUST include a unique canonical `name` string.
- `aliases` MUST be a YAML sequence of strings when present.
- `related` MUST be a YAML sequence of strings when present.
- Additional metadata fields MAY be added later for ranking or validation, but they must not change the canonical `name` for a record.
- Canonical names are treated as the stable identifiers for matching and output.
- Aliases and related terms are normalized as lowercased comparison terms during matching, but the source YAML should remain human-readable.
- Duplicate canonical names are invalid.

### Parsed Posting Output Schema
Each parsed posting line or chunk should produce a structured object with a documented schema.

Implementation note for the first iteration:
- Prefer a class-based parser surface so deterministic and LLM-backed parsing can live side by side.
- Keep the deterministic path as the default for local validation and regression testing.
- Add an LLM-backed method that uses the same reviewable schema when structured extraction is available.

Expected fields:

#### Parsed Posting Schema Draft
- One record represents one source posting line or chunk.
- `posting_line` MUST preserve the original text fragment used for extraction.
- `extracted_raw_terms` MUST be a YAML sequence of strings describing the terms surfaced from the fragment.
- `matched_skills` MUST be a YAML sequence of structured skill matches.
- Each matched skill record MUST include `raw_term`, `canonical_name`, `match_type`, `confidence`, `relevance_score`, and `evidence`.
- `match_type` SHOULD be one of `exact`, `alias`, or `related`.
- `confidence` SHOULD be a numeric value between 0.0 and 1.0.
- `relevance_score` SHOULD be a numeric ranking value with higher meaning more relevant.
- A top-level `validation` block MAY be included for review artifacts, but downstream code should treat validation as a separate contract once it is finalized.
- The sample artifact in `schemas/parsed_posting_line.example.yaml` is the draft reference for this schema and should be human-reviewed before it becomes authoritative.
- Future schema drafts should also ship with a concrete example artifact in `schemas/parsed_posting_line.example.yaml` or a similarly named reviewable example file so the contract stays inspectable.

This schema is intentionally reviewable and should be human-verified before it becomes the contract.

### Validation Output Schema
Validation output should be a small, explicit report that explains whether a parsed line or generated skills selection should be accepted.

Expected fields:
- `status`
- `notes`
- optional `issues`

#### Validation Schema Draft
- `status` MUST be one of `pass`, `fail`, or `flag`.
- `notes` MUST be a YAML sequence of human-readable strings.
- `issues` MAY be included as a YAML sequence of structured problem records when the validator needs to explain specific failures.
- Validation output should remain separate from the extraction result once the contract is finalized, even if draft review artifacts embed a validation block for convenience.
- The validation output must make it easy to distinguish accepted outputs from those that require manual review or rejection.

### LLM Call Contract
Use the provider abstraction in `src/llm/` for all model calls.

Expected behavior:
- `call_json(...)` is the default for structured extraction, filtering, and matching tasks.
- `call(...)` is reserved for freeform text generation or cases where a structured response is not practical.
- All prompts should include a role-specific `system_prompt` when available.
- Structured prompts must explicitly name the expected output fields and schema shape.
- Temperature should stay low for extraction and validation tasks so repeated runs remain reproducible.
- `max_tokens` should be sized to the smallest practical response for the task.

Phase 3 uses LLM prompts for these steps:
- splitting a posting into useful chunks
- filtering chunks that are unlikely to contain skill-relevant content
- extracting candidate skill mentions from each chunk

Phase 3 should keep these steps deterministic where possible:
- reconstructing the posting from its chunks for validation
- normalizing extracted text
- validating duplicates and weak matches against the cache

### Resume Template
LaTeX template with placeholders for generated content.

## Outputs
### Intermediate Outputs
- extracted skills per posting line or chunk
- normalized canonical skill matches
- scoring and confidence data
- validation results

### Final Output
- updated LaTeX resume source
- compiled PDF

## Core Workflow
1. Read the job posting.
2. Split it into useful chunks by separating informative text from useless text.
3. Filter out non-skill content.
4. Extract skills from each useful chunk.
5. Match extracted terms to the skill cache.
6. Assign relevance and confidence scores.
7. Validate that the selected skills are grounded in the posting and in the cache.
8. Format the skills section into the LaTeX template.
9. Render to PDF.
10. Validate the rendered PDF.

## Matching Rules
- Exact canonical matches are strongest.
- Aliases are strong matches.
- Related terms are weaker and should be used carefully.
- Ambiguous matches must be flagged for validation.
- The system should prefer canonical skill names in output.

## Ranking Rules
Skills should be ranked using a combination of:
- direct mention strength
- alias vs related-term quality
- frequency in the posting
- inferred role relevance
- cache priority / baseline importance

## Validation Rules
The system should reject or flag outputs when:
- a skill is not present in the cache and cannot be justified
- a match is too weak or ambiguous
- duplicate skills appear in the final list
- the skills section becomes too long
- the LaTeX output fails to compile
- the PDF exceeds the target format

Any schema used by the agent, including per-line parse outputs, should be human-reviewed before it is treated as authoritative.

## Acceptance Criteria
The first implementation is acceptable when:
- a posting can be processed end-to-end
- skills are selected from the cache and ranked consistently
- output is reproducible across repeated runs
- invalid or weak matches are surfaced clearly
- the generated PDF is valid and fits the expected layout
- the expected schemas have been reviewed and iterated by a human where needed

## Design Principles
- Prefer deterministic logic where possible.
- Use the LLM for extraction and judgment, not for uncontrolled generation.
- Keep canonical data separate from generated output.
- Make intermediate artifacts inspectable.
- Optimize for maintainability and debuggability.