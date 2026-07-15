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
- [done] Refactor LLM parser flow to extract all raw skills in one batch, then deterministically match them to the cache
- [done] Add parser tests for LLM cache-constrained matching and rejection of non-cache canonical names
- [done] Add missing-skills intermediate output for unmatched extracted terms to support future skill-cache expansion
- [done] Add structured extraction candidates with include/discard flags and reasons to reduce generic missing-skill noise
- [done] Split missing-skill outputs into candidate terms and discarded terms for cache curation auditability
- [done] Remove separate Phase 4 and fold matching responsibilities into parsing + validation flow
- [done] Add Phase 5 validation functions for selected skills (unsupported, weak, grounding, size constraints)
- [done] Add Phase 5 tests for validation pass/fail behavior
- [done] Add Phase 5 LLM-assisted grounding validation for edge cases (for example ipynb -> jupyter)
- [done] Implement Phase 6 rendering module with template injection and bash-based pdflatex rendering
- [done] Add LLM-based grouping of selected skills into Languages / ML & Data / Tools sections before template injection
- [done] Allow LLM grouping to include only role-relevant sections (for example omit ML & Data when irrelevant)
- [done] Convert `data/template.tex` into an injectable template by replacing static skills with [INSERT SKILLS HERE]
- [done] Add render tests for template injection, skill section formatting, and pdflatex PDF generation
- [done] Implement run-scoped output folders under `build/[run_name]/` with `aux/` for TeX/PDF artifacts and `logs/` for pipeline and pdflatex logs
- [done] Keep final resume PDF at `build/[run_name]/tailored_resume.pdf` while retaining TeX and engine artifacts in `aux/`
- [done] Capitalize displayed skill names in rendered LaTeX output for readability (for example `python` -> `Python`)
- [done] Add run telemetry logs for stage timings, estimated token usage, parse/validation counts, and artifact sizes
- [done] Add a dedicated observability phase documenting loggable pipeline information and persisted log artifacts
- [done] Add a canonical big-section sentence benchmark and record the current parser scores
- [done] Add a gpt-4o integration test that compares missing-skills output to the canonical big-section skill set with explicit margin thresholds
- [done] Replace broad big-section OpenAI coverage with sentence-level one-skill chunk cases
- [done] Add a gpt-4o integration test that asserts each pre-split chunk returns exactly one expected skill and no extras
- [done] Rename the current LLM parser to `MultiShotPostingParser` and add `SingleShotPostingParser` for comparison
- [done] Diagnose multishot's chunk-splitting bug (LLM re-split of already-atomic input duplicated/fragmented results) and fix via `resolve_chunks`
- [done] Add experimental parser variants (`OrchestraSingleShotParser`, `MultiShotPostingParserV1Loose`) and benchmark them against v1/v2/single-shot on atomic and combined-posting scenarios
- [done] Promote `OrchestraSingleShotParser` (deterministic-only chunking, per-chunk single-shot-style calls) to the default in `parse_posting()` and the `--parser-mode` CLI flag, based on benchmark evidence that it matches-or-beats multishot_v1 while being simpler/cheaper
- [done] Deprecate and delete the multishot family (`MultiShotPostingParserV1`, `MultiShotPostingParserV2`, `MultiShotPostingParserV1Loose`) and their now-dead LLM-based re-chunking helpers now that `OrchestraSingleShotParser` is the default; consolidated the retained extraction/matching logic directly into `orchestra_single_shot.py`
- [done] Add strict JSON Schema structured outputs (OpenAI `response_format: json_schema`) for extraction, grounding validation, and skill-section grouping calls, replacing prompt-instructed JSON shape hints
- [done] Add self-consistency voting (`num_votes`, default 3) to `OrchestraSingleShotParser`: repeats each chunk's extraction call and keeps only majority-agreed terms
- [done] Split extraction vs. validation/sectioning onto separate models (`--extraction-model` default gpt-4o-mini, `--model` default gpt-4o for grounding fallback and skill grouping)
- [done] Add real, provider-reported token usage tracking (`LLMProvider.usage_totals`, including cached-prompt-token counts) to `run_metrics.json`, replacing the estimate-only placeholder when available
- [not started] TODO: Further review `_llm_group_skills` behavior with the user (section omission policy, fallback assignment policy, and prompt contract stability)
- [done] Audit repo files against `SPEC.md`/`DEV_PLAN.md` and record findings (test discovery gaps, dead CLI flags, missing Phase 7 implementation)
- [done] Add missing `__init__.py` files to `tests/evals/`, `tests/parse_posting/`, `tests/render/` so `unittest discover` finds all test modules
- [done] Rewrite `tests/llm/test_openai.py` as a skip-guarded `unittest.TestCase` (previously pytest-style bare functions with no API-key guard, unlike the rest of the live-integration tests)
- [done] Remove dead `PipelineConfig.output_tex_path`/`output_pdf_path` fields and `--output-tex`/`--output-pdf` CLI flags in `src/main.py` (run-scoped `build/[run_name]/` paths were already the only paths actually used)
- [done] Implement Phase 7 PDF validation (`validate_pdf` in `src/render_resume.py`): page count via `pypdf`, PDF readability, and skills-section rendered-line-count check (adjusted scope: at most 3 lines instead of general layout spot-check)
- [done] Wire `validate_pdf` into the pipeline (`src/main.py`), writing `pdf_validation.json` and failing the run on a non-pass status
- [done] Add `validate_pdf` regression tests in `tests/render/test_render_resume.py` (pass, page-count-exceeded, skills-section-too-long, missing-file cases)
- [done] Fix a shell-injection risk in `render_pdf_with_pdflatex` (previously built a `bash -lc "..."` string by interpolating user-influenced paths; now invokes `pdflatex` directly as an argv list, no shell)
- [done] Analyze recall gaps in `OrchestraSingleShotParser` against `build/benchmarks/experimental_parser_variants.json`: precision is 1.0 (no noise) but recall ~0.78-0.84; 100% of misses are (a) dropped "headline domain term before parenthetical examples" phrases and (b) inconsistent splitting of conjoined "X and Y Z" phrases
- [done] Fix a confirmed discard-list bug contributing to (a): removed `"usage-based insurance (ubi)"` from `_DISCARD_TERMS` in `src/parser/orchestra_single_shot.py`, which was unconditionally discarding that expected term every run
- [done] Add an explicit "extract the headline category term AND its parenthetical examples, they are not redundant" instruction + a concrete few-shot example to the extraction prompt (`_extract_terms_llm_batch`), and soften the refine step's "do not output fragments" instruction (`_refine_extraction_candidates_llm`) to explicitly carve out the headline/examples case as non-fragmentary. Deferred regex-based and conjoined-phrase-merging fixes per user direction (regex doesn't generalize across careers; conjoined-phrase splitting is a symptom of a larger issue to tackle separately).
- [done] Remove the hardcoded, posting-specific `_DISCARD_TERMS` noise-keyword set entirely (including its duplicate regex check for the same 5 phrases) per user direction that a fixed keyword list doesn't generalize across careers. Generic/noise filtering now relies solely on the extraction/refine/classification prompts' own instructions and each candidate's LLM-assigned `category`/`include_for_resume_skills`/`include_for_cache_candidate` flags. `_DEGREE_FIELD_NOISE`, `SOFT_SKILL_TERM_HINTS`, and the job-title regex were left in place (not flagged for removal).
- [done] Ran the live sentence-case benchmark after removing `_DISCARD_TERMS`/`_DEGREE_FIELD_NOISE`/`SOFT_SKILL_TERM_HINTS`/job-title-regex checks: F1 dropped 85.76% -> 75.76% (precision 1.0 -> 0.74), with concrete new false positives (degree-major words, "explaining results and limitations") no longer filtered. Confirmed this is a real precision cost of full generalization, not sampling noise; motivated the semantic-matching plan below rather than re-adding keyword lists.
- [done] Diagnosed two concrete, non-metric root causes behind remaining benchmark gaps: (1) the "Degree in a relevant discipline (e.g., mathematics, ..., statistics, ...)" atomic sentence case has leaky/arbitrary ground truth (expects only "statistics", using knowledge from a *different* sentence elsewhere in the full posting that isn't available when this chunk is scored in isolation); (2) "GLM/GBM" not matching "glm"+"gbm" was a concrete ordering bug in `_candidate_display_terms` (its verbatim-in-posting early return fired before the abbreviation-slash-split heuristic ever ran), not a fundamental limit of deterministic matching.
- [done] **Phase B: implemented embedding-based semantic matching**, per user direction to make embeddings the primary path for alias/variant resolution instead of continuing to hand-enumerate every phrasing edge case in `skills.yaml` (which doesn't scale/generalize). Refactored matching into `src/parser/matching.py` with three independently-testable classes:
  - `ExactAliasMatcher`: the existing deterministic exact/alias/related lookup, moved out of `base.py` unchanged in behavior (kept as the free, instant, fully-deterministic first tier).
  - `SemanticMatcher`: new embedding-based cosine-similarity matcher. Added `LLMProvider.embed()` (default raises `NotImplementedError`; implemented in `OpenAIProvider` via `text-embedding-3-small`, no new heavy dependency since `openai` was already a dependency). Returns `match_type: "semantic"` candidates, and can return *multiple* canonical matches for one raw term (needed for "GLM/GBM" -> both `glm` and `gbm`, not just a single best match).
  - `LLMGroundingMatcher`: the existing Phase 5 LLM grounding-validation fallback (`_llm_validate_skill_grounding`), moved out of `selection.py` into its own class with the same `confirm_grounding(...)` behavior, for independent testability.
  - `MATCH_PRIORITY`/`BASE_CONFIDENCE`/`BASE_RELEVANCE` in `models.py` gained a `"semantic"` entry; `_finalize_match` in `base.py` now accepts a precomputed `base_confidence` (used for semantic matches, whose confidence comes from similarity score, not a fixed per-match_type constant).
  - `OrchestraSingleShotParser` (and `SingleShotPostingParser` via inheritance) now try exact/alias first, then fall back to `SemanticMatcher` before giving up to `missing_skills`; constructor takes `use_semantic_matching: bool = True` and gracefully disables semantic matching if the LLM provider doesn't support `embed()` (e.g. Anthropic/Ollama today). `parse_posting()` exposes the same toggle.
  - Added `numpy` as a dependency for cosine similarity computation.
  - **Empirical threshold calibration finding (important):** bare short terms embedded in isolation don't carry enough signal - real `text-embedding-3-small` similarity for bare "ipynb" was *lower* against "jupyter" (0.44) than against "python" (0.47), a wrong answer, at an initially-assumed 0.80 threshold. Embedding the raw term *together with its surrounding posting-chunk context* fixed this (0.51 for jupyter, python no longer even close) and gave a clean separation from unrelated terms (~0.32 ceiling for unrelated pairs with equivalent context vs. ~0.50-0.56 for true matches, measured live against real OpenAI embeddings for ipynb/jupyter, GLM-GBM/glm+gbm, BSc/bachelor's degree, kubernetes-negative-control). `SemanticMatcher.match()`/`match_batch()` therefore accept an optional `context` string (the posting chunk text) to embed alongside the raw term, and `DEFAULT_SIMILARITY_THRESHOLD` was recalibrated from an initial guess of 0.80 down to 0.45 based on this measurement.
  - Added `tests/parse_posting/test_matching.py` (15 tests) covering all three matcher classes independently: `ExactAliasMatcher` against real jupyter/ipynb-style cache fixtures; `SemanticMatcher` against a deterministic fake-embedding provider (hand-crafted vectors, no API cost) covering single-match, multi-match (GLM/GBM -> glm+gbm), below-threshold, batch, and provider-without-embeddings-support cases; `LLMGroundingMatcher` against a fake grounding-confirmation provider. Verified separately (ad hoc, not a committed test) against real OpenAI embeddings that ipynb/GLM-GBM/BSc resolve correctly and a kubernetes negative control returns no match.
- [done] **Persistent embedding cache** (closes the follow-up flagged above): added `EmbeddingCache` in `src/parser/embedding_cache.py` - a JSON-backed cache of `{embedding_model: {text: vector}}`, keyed by individual reference text (not a whole-cache hash), so adding or changing one skill only re-embeds that one text. Default location `build/cache/skill_embeddings_cache.json` (gitignored, persists across runs, distinct from per-run `build/<run_name>/` folders). `SemanticMatcher` takes an optional `embedding_cache` param and looks up each reference text before calling `embed()`, only embedding cache misses and persisting new entries back. `OrchestraSingleShotParser`/`SingleShotPostingParser`/`parse_posting()` gained `embedding_cache_path` (default enabled; pass `None` to disable). Verified end-to-end: a cold run populated 35 cached entries; a second real pipeline run against the same cache added zero new entries (no cache growth), confirming reuse. Added `EmbeddingCache` round-trip/model-isolation/corrupt-file tests plus a `SemanticMatcher` test asserting a warm cache makes zero additional `embed()` calls.
- [not started] TODO: propose and implement a dedicated matching-quality measurement plan (distinct from the existing extraction-focused sentence-case benchmark) - see plan drafted 2026-07-15 for fixture design, precision/recall-by-tier metrics, and threshold-governance process.
- [done] **Architecture review (2026-07-15)**: confirmed with the user that the existing 3-stage design (deterministic chunking -> LLM extraction-with-judgment -> separate cache matching) already matches their intended architecture, including that extraction never needs to know canonical cache names (`EXTRACTION_CANDIDATES_JSON_SCHEMA` never included one). Identified two concrete gaps: (1) chunking is not robust to copy-pasted PDFs/websites where line breaks land mid-sentence (deferred - flagged as needing its own design, not implemented); (2) the extraction prompt has no explicit "credential/degree-mention vs. demonstrated-responsibility" instruction (e.g. "engineering" as a degree major is not a skill, but "data engineering" as a responsibility is) - explicitly deferred by the user until an evaluation method exists to validate the change.
- [done] **Implemented the two-step, two-tier LLM-judge quality-evaluation framework** requested to satisfy the above validation gate, as new evaluation infrastructure (not a production pipeline feature) under `tests/evals/`:
  - `tests/evals/quality_judges.py`: `judge_extraction_quality(llm_provider, chunk, extracted_terms)` (tier 1: free deterministic substring-grounding filter catches hallucinated/ungrounded extractions instantly; tier 2: strong-LLM judge checks each grounded term is a genuine, specific, demonstrable skill in context - not a soft skill/degree-mention/job-title/responsibility-without-specifics - and separately scans for missed skills) and `judge_match_relevance(llm_provider, raw_term, canonical_name, context, match_type)` (tier 1: exact/alias matches auto-pass, since they're human-curated cache facts, not LLM-judged; tier 2: semantic/related matches get an LLM relevance check, e.g. does "ipynb" in context really imply Jupyter).
  - `tests/evals/test_quality_judges.py`: live, `OPENAI_API_KEY`-gated tests. `ExtractionQualityJudgeTest` runs the real `OrchestraSingleShotParser` over `sample_big_section_sentence_cases.yaml` and judges the actual extracted terms with gpt-4o, writing a report to `build/benchmarks/quality_judges_benchmark.json` (reporting-only, no hard gate yet - no multi-run baseline to calibrate against). `MatchRelevanceJudgeTest` has 4 hard-asserted cases with unambiguous correct answers: a semantic-tier plural variant ("Jupyter notebooks"), a semantic-tier compound abbreviation ("GLM/GBM" -> both glm and gbm), an exact-match auto-pass (confirms tier 1 skips the LLM), and a deliberately-wrong injected pairing (kubernetes/jupiter) that the judge correctly rejects. All 4 passed on a real run.
  - **First real run already produced concrete, actionable signal**: the extraction-quality judge, run against the *current* (not-yet-updated) extraction prompt, independently flagged all 6 degree-major words in the "Degree in a relevant discipline (e.g., mathematics, ..., statistics, ...)" chunk as invalid - including "statistics", which `sample_big_section_sentence_cases.yaml`'s own ground truth currently claims IS a valid skill for that chunk. This is independent confirmation (from a completely separate evaluation mechanism, not just human judgment) that both the extraction behavior *and* that specific fixture's ground truth are inconsistent, matching the earlier diagnosis in this session. This is exactly the kind of evidence the user asked to gather before touching the extraction prompt.
  - **Also revealed a real tension worth flagging, not yet resolved**: the judge disagreed with the project's own established ground truth on domain/headline-level terms - it called "insurance pricing", "call center optimization", "usage-based insurance", "telematics", and "segmentation" too broad/abstract to be skills, even though `sample_big_section_sentence_cases.yaml` (and the extraction prompt's own explicit headline-term instruction, added earlier this session) treat these as legitimate expected skills. The LLM judge's own notion of "specific enough to be a skill" doesn't fully match this project's domain-expert-curated ground truth. Needs a decision: is the fixture right and the judge's prompt needs tightening (e.g. explicitly allow domain/industry-area terms), or is the fixture itself too permissive?
  - **Known minor limitation**: the extraction judge's tier-1 grounding filter is a plain substring check and can false-flag a legitimately-extracted paraphrase-split term as "ungrounded" (observed once: "LLM-assisted workflows" extracted from "LLM-assisted and Agentic workflows" was flagged both invalid *and* missed in the same report - a contradiction). The production parser already handles this exact pattern via `_candidate_display_terms`'s splitting heuristic; the evaluator's simpler check doesn't replicate it. Not fixed - flagged as a known rough edge in the evaluation tool itself, not the production pipeline.
- [done] **Deprecated `sample_big_section_sentence_cases.yaml` as ground truth (2026-07-15, per explicit user decision)**: "the fixture is an artifact, which we will no longer consider truth. we will work on refining the judge, and then we will consider that the new truth." The old F1 hard-gate in `tests/evals/test_big_section_skill_coverage.py` (`MIN_ORCHESTRA_F1_SCORE = 0.90`) was softened to reporting-only (prints the F1 instead of asserting on it), since hard-gating against ground truth we've explicitly said isn't trustworthy anymore would be internally inconsistent. The fixture file itself and its F1 measurement are kept as informational signal, not deleted.
- [done] **Refined the judge based on two concerning outputs the user flagged directly from the first report:**
  - Tried the user's proposed fix for the grounding-filter contradiction ("LLM-assisted workflows" flagged both invalid-ungrounded and missed): an embedding-only workflow (embed the candidate term and the whole chunk, compare cosine similarity). **Empirically rejected** - measured live, a genuinely grounded short term ("peer review", similarity 0.27) scored *lower* than a genuinely hallucinated term ("continuous integration", similarity 0.30) against the same chunk: the true/false ordering was inverted, making it less reliable than the check it would replace, not more. Reported this negative result rather than proceeding with it.
  - Implemented a **deterministic token-subset grounding check** instead: a term is grounded if every one of its significant words (stopwords excluded) appears somewhere in the chunk, not necessarily contiguously - fixes the exact paraphrase-split case (`"LLM-assisted workflows"` <- `"...LLM-assisted and Agentic workflows..."`) without embedding cost/noise. Verified live: the contradiction is gone (term is no longer flagged invalid or missed). Trade-off noted: this can false-positive if a chunk happens to scatter all of a term's words across unrelated parts of the same short sentence - accepted given chunks here are single sentences.
  - Fixed the tier-2 prompt's self-contradictory "Git" reasoning (it had called "Git" invalid because a broader phrase "Git-based development practices" was "broader and more specific" - a contradiction in the model's own words) and the related tendency to invent overly-broad "missed skill" phrases restating a whole clause instead of atomic skill terms. Added explicit prompt guidance: a term that is itself a valid, specific tool/technology name is valid on its own regardless of a broader phrase also being present, and missed_skills must be atomic phrases at genuine-resume-skill granularity, not clause paraphrases. Verified live: "Git" no longer flagged invalid; missed_skills for that chunk is now the correctly atomic `["peer review", "writing tests", "technical documentation"]` instead of inventing "Git-based development practices".
  - Added `tests/evals/test_quality_judges_grounding.py` (4 deterministic, no-API-key-needed tests) protecting the token-subset grounding fix specifically (exact substring, paraphrase-split, hallucinated-term rejection, empty-term rejection).
  - The domain/headline-term tension flagged in the previous entry (judge vs. fixture disagreeing on "insurance pricing"/"intelligent document processing" etc.) is **still open** - not resolved by this round of fixes, and reproduced again on the "intelligent document processing" case during verification. Still needs a decision from the user.
- [done] **Resolved the domain-knowledge tension (2026-07-15, user decision):** for the current skills-only scope, the skills section is technical-only; business/industry domain knowledge (insurance pricing, telematics, call center optimization) is explicitly excluded from it, though it remains legitimate and should be preserved once this project later tailors experience/project bullets rather than just the skills section.
- [done] **Iterated an embedding-based grounding-check idea further per user request (context-in-text templating), then abandoned it** based on new evidence: tested two context templates (short "Skill term: X / Job posting sentence: Y" and a richer "Context/Item/Metadata" template modeled on the user's suggested pattern). Both showed the same structural failure - a genuinely grounded generic term ("peer review") scored lower than a genuinely hallucinated one ("continuous integration") under both templates, with the richer template making the inversion worse (0.6363 vs 0.6464), not better. Confirmed: whole-chunk embedding similarity is unreliable for grounding verification regardless of context-templating sophistication; kept the deterministic token-subset check as-is.
- [done] **Delivered a written analysis + hybrid-cascade pipeline proposal** for the lexical-vs-semantic trade-off (token matching = precise identity check, blind to aliasing; embeddings = bridges aliasing, blind to fine-grained identity among topically-similar short phrases). Conclusion: grounding and matching are different sub-problems, each already served by the right tool (token-based grounding; embedding-based cache matching with LLM-judge escalation for ambiguity) - user explicitly declined the proposed ambiguity-escalation safeguard for `SemanticMatcher` ("i don't see the necessity... don't implement").
- [done] **5-round extraction-prompt iteration campaign against the LLM judge (2026-07-15)**, per user instruction to treat the refined judge as ground truth going forward. Added `tests/evals/extraction_prompt_benchmark.py` (standalone, not gated) to run the real parser over all 7 sentence-case chunks and judge the actual extraction each round, writing `build/benchmarks/extraction_prompt_iteration_{1..5}.json` (each includes the literal prompt source used, for inspectability). Results (total invalid_terms / total_missed_skills across 7 chunks):
  - iteration_1 (applied "use exact chunk tokens" + technical-vs-domain rule to `_extract_terms_llm_batch`/`_refine_extraction_candidates_llm`): 19 / 6
  - iteration_2 (added degree-major exclusion + "preserve complete phrase, don't truncate to bare noun"): 21 / 4 - regression: introduced 2 new false extractions ("education", "experience"); degree-major words still not suppressed at all
  - iteration_3 (strengthened degree-major exclusion with an exact worked example matching the chunk almost verbatim; added named-technical-field examples to the domain-exemption): 22 / 2 - degree-major instruction STILL had zero effect on extraction despite the explicit worked example; judge started incorrectly flagging "machine learning" as invalid despite explicit instruction it's valid
  - iteration_4 (front-loaded/reordered both prompts so the degree-major rule and the recognized-technical-field exemption are the FIRST instruction in each, not buried mid/end-of-prompt): 13 / 5 - big improvement (22->13 invalid), confirming instruction *position* mattered more than instruction *content* for these two models; side effect: the front-loaded "recognized field" exemption over-generalized and also stopped the judge from flagging bare "mathematics"/"statistics"/"engineering"/etc. as invalid (only "education"/"experience" still flagged), and the extraction under-extracted the Python/Git chunk (dropped version control/writing tests/technical documentation entirely)
  - iteration_5 (added an explicit precedence sentence to the extraction prompt only - degree-major rule wins over recognized-field rule when a word appears only as a listed major - plus "this strict rule applies only to degree/qualification enumerations, don't become more conservative elsewhere"): 15 / 2 - fixed the Python/Git under-extraction (all terms extracted again); degree-major-vs-recognized-field precedence conflict is NOT fixed in the JUDGE prompt (only added to the extraction prompt) - "mathematics"/"statistics"/etc. remain unflagged there, an open gap
  - **Net result across 5 iterations**: invalid_terms+missed_skills went from 25 (iteration 1) to 17 (iteration 5), with the clearest, most reliable win being domain-only headline terms (insurance pricing, usage-based insurance, telematics, call center optimization) now consistently excluded from extraction entirely.
  - **Known remaining issues, not fixed in this campaign:** (a) the "process vs. technical skill" boundary for terms like "peer review", "writing tests", "technical documentation" is genuinely unstable in the judge - it flip-flopped between valid/invalid/missing across iterations with no clear resolution, independent of extraction-prompt changes; (b) the degree-major-vs-recognized-field precedence fix was only applied to the production extraction prompt, not mirrored into the judge's own prompt in `quality_judges.py`, so the judge still under-flags "mathematics"/"statistics"/"engineering"/"operations research"/"geomatics"/"AI" in the degree-list chunk; (c) LLM instruction *position* within a long prompt appears to matter as much as instruction *content* for both gpt-4o-mini (extraction) and gpt-4o (judge) - a front-loaded "CRITICAL RULE" measurably outperformed the identical rule stated later in the same prompt (iteration 3 -> 4).
- [done] **Corrected two methodological problems the user identified in the 5-iteration campaign above (2026-07-15, same day) - the iteration_1..5 numbers should be treated as informal directional signal only, not a valid controlled comparison:**
  1. **Prompt contamination**: both the extraction prompt and the judge prompt had accumulated verbatim or near-verbatim text copied directly from `sample_big_section_sentence_cases.yaml` (the entire "Degree in a relevant discipline (e.g., mathematics, engineering, operations research, statistics, geomatics, AI)..." sentence, plus "insurance pricing", "call center optimization", "usage-based insurance", "telematics", "GLM", "demand forecasting", "time-series feature engineering", "Git-based development practices"). This is a train/test leakage problem - a prompt that has memorized the eval fixture's own sentences isn't being validated against unseen input, so passing scores don't demonstrate generalization. Rewrote all three prompts (`_extract_terms_llm_batch`, `_refine_extraction_candidates_llm`, `judge_extraction_quality`) to use abstract, structural placeholder examples (`<Business Domain>`, `X, Y, or Z`) instead of any fixture-derived wording, while preserving the same underlying rules. This also substantially shortened all three prompts (the user's separate complaint about verbosity/inefficiency) since the previous versions had accumulated long, specific, patched-together worked examples across the 5 iterations.
  2. **Non-frozen judge**: the judge's own prompt was edited mid-campaign (iterations 3 and 4 both touched `quality_judges.py`), which confounds the iteration-over-iteration extraction-prompt comparison - a change in invalid/missed counts could be due to the extraction prompt improving OR the judge's criteria shifting, and there's no way to separate the two from the iteration_1..5 artifacts alone. Added an explicit "FROZEN PROMPT" policy as a code comment at the top of `quality_judges.py`: the judge prompt must not change while iterating on extraction; if it genuinely needs to change, that must be its own separate, deliberately-labeled step.
  - Ran one clean validation benchmark with the decontaminated, now-frozen judge and decontaminated extraction prompt: `build/benchmarks/extraction_prompt_decontaminated_v2.json` - **12 invalid_terms / 9 missed_skills** across the 7 sentence-case chunks. This number is NOT directly comparable to the iteration_1..5 numbers (different judge version, decontaminated prompts) - treat it as the new valid starting baseline going forward, not as a regression from iteration_5's 15/2.
- [done] **Implemented and benchmarked an experimental parallel multi-classifier extraction pipeline**, per user analysis request + decision to implement it (gap analysis showed a single monolithic extraction prompt suffers real instruction interference between its degree-context/domain-vs-technical/soft-skill/decomposition rules). New experimental module `tests/evals/parallel_classifier_pipeline.py` (not wired into production `OrchestraSingleShotParser` yet):
  - Stage 1 (`decompose_candidates`): broad, recall-only decomposition into atomic candidate phrases, no classification judgment.
  - Stage 2 (`run_classifier_voted`, x3 concurrently via `ThreadPoolExecutor`): three independent, narrowly-scoped classifiers - `degree_context`, `domain_vs_technical`, `soft_skill` - each judging the SAME full candidate list without seeing each other's verdicts. A candidate is excluded from the final result if ANY classifier flags it. Each classifier path supports optional self-consistency voting across `n` samples (reuses `src/parser/voting.py`'s `majority_threshold`).
  - Benchmark runner `tests/evals/parallel_classifier_benchmark.py`, run against the same frozen judge and same 7 sentence-case chunks as the single-prompt baseline for direct comparability.
  - **Results**: n=1 (`build/benchmarks/parallel_classifier_n1.json`): 6 invalid_terms / 8 missed_skills (14 total) vs. the single-prompt baseline's 12/9 (21 total) - a clear improvement, consistent with the gap analysis (isolating the degree-context rule into its own call finally let it work reliably, matching the earlier finding that this rule had a 0% success rate for 5+ iterations when bundled with other rules).
  - n=3 (`build/benchmarks/parallel_classifier_n3.json`): 6 invalid_terms / 10 missed_skills (16 total), with only 2 recorded disagreements across ~150+ individual classifier votes. **`extracted_terms` were identical between the n=1 and n=3 runs for every single chunk** - the small total-count difference (14 vs 16) is attributable to the judge's own call-to-call variance on a re-run with identical input, not to anything the voting changed. **Conclusion: for this pipeline and dataset, n=3 self-consistency voting provides no meaningful benefit over n=1** - each classifier is already highly consistent on its own at temperature 0.1, so voting adds ~3x classifier-call cost without adding accuracy. Recommend n=1 if/when this pipeline is promoted.
  - Not yet done: this experimental pipeline has not been promoted to replace the production single-prompt extraction in `OrchestraSingleShotParser` - that would need its own deliberate decision given the added latency/cost (1 decomposition + 3 classifier calls per chunk vs. 1-2 calls today), consistent with this project's practice of benchmarking before promoting a parser strategy change.
- [done] **Promoted the parallel multi-classifier extraction pipeline to production at n=1 (2026-07-15, per explicit user instruction).** Ported the validated logic from `tests/evals/parallel_classifier_pipeline.py` into a new production module `src/parser/parallel_extraction.py` (`decompose_candidates`, `run_classifier_voted`, `extract_with_parallel_classifiers`), adapting its output from a flat `extracted_terms: List[str]` shape to the standard candidate-dict shape (`raw_term`/`category`/`include_for_resume_skills`/`include_for_cache_candidate`/`reason`/`evidence_quote`) so the rest of the pipeline (`_normalize_extraction_candidates`, `_match_extracted_terms_to_cache`, cache matching) needed no changes. Excluded candidates are still returned (with include flags `False`) rather than dropped, so they continue to surface in `missing_skills_discarded` for auditability, same as before.
  - `OrchestraSingleShotParser._extract_terms_llm_batch` now calls `extract_with_parallel_classifiers(self.llm, posting_text, n=self.classifier_votes)`. Removed the old single monolithic extraction prompt, `_refine_extraction_candidates_llm`, `_classify_legacy_terms_llm`, `_LEGACY_TERMS_JSON_SCHEMA`, and the now-unused `EXTRACTION_CANDIDATES_JSON_SCHEMA` import (the schema itself is left in `src/llm/schemas.py`, unused for now, in case it's wanted again later).
  - Added a new `classifier_votes: int = 1` constructor parameter (to `OrchestraSingleShotParser`, `SingleShotPostingParser`, and `parse_posting()`) controlling self-consistency voting *within* each of the 3 parallel classifiers, defaulting to 1 per the benchmarked n=1-vs-n=3 finding (no measurable difference, see above). This is separate from the pre-existing outer `num_votes` (default 3), which still governs whole-chunk-extraction self-consistency voting one level up; the two compose (default settings: 3 outer votes x (1 decomposition + 3 classifiers) = up to 12 LLM calls per chunk - flagged here since it's a real cost/latency increase over the old single-prompt approach's 1-2 calls per chunk).
  - **Found and fixed a real measurement bug this surfaced in `tests/evals/test_quality_judges.py`**: the extraction-quality judge test judged `record["extracted_raw_terms"]`, which the old single-prompt architecture already filtered inline (rules were applied *during* generation, so invalid candidates were rarely proposed at all) - but the new decompose-then-classify architecture deliberately over-proposes atomic candidates in Stage 1 and relies on Stage 2's classifiers' `include_for_resume_skills` flag to exclude bad ones, so `extracted_raw_terms` (the full pre-filter list) is no longer a fair thing to judge; a candidate correctly excluded by a classifier is not extraction noise. First run under this bug showed 27 invalid_terms / 1 missed_skills (much worse than expected, because "call center optimization"-style domain terms and degree-major words, though correctly excluded via `include_for_resume_skills=False`, still counted against the judge since they still appeared in the raw list). Fixed by judging the post-classification-filtered candidate list instead (`extraction_candidates` filtered to `include_for_resume_skills=True`). Re-run: 6-7 invalid_terms / 8-11 missed_skills (~14 total, run-to-run judge variance), matching the standalone experimental benchmark's validated 14-total result almost exactly - confirming the production port is behaviorally equivalent to the validated experimental pipeline.
  - Updated mocked tests in `tests/parse_posting/test_parse_posting.py` (`FakeLLMProvider`, `AssetListFakeLLMProvider`, `SingleShotFakeLLMProvider`-dependent assertion) to respond to the new pipeline's 4 distinct prompts (1 decomposition + 3 classifiers) instead of the old single fallback-prompt text pattern. All 16 parser tests plus the full non-stale-fixture suite (41 tests total across parse_posting/render/llm/evals) pass.
  - `tests.evals.test_big_section_skill_coverage_openai`'s per-chunk exact-set assertions remain a known, pre-existing, ignorable failure (see `/memories/repo/testing.md`) - its informational F1 dropped further (85.76% -> 54.13%) because the new decomposition-first architecture correctly excludes business-domain headline terms and splits compound listed items differently than that stale fixture (predating the domain-knowledge policy decision) expects. Not something to "fix" by editing the fixture without human review per `AGENTS.md`'s human-review gate on eval ground truth.
- [done] **Returned to F1 scoring as ground truth, replacing the LLM-judge approach (2026-07-15, same day, per explicit user decision)**: the judge added unneeded noise for what the user considers a static/deterministic problem. `tests/evals/sample_big_section_sentence_cases.yaml`'s `expected_terms` are once again **totalic** (exhaustive) ground truth - exactly these terms, no more/no less - reversing the earlier deprecation decision, now reviewed against the current extraction policy rather than the stale pre-policy version:
  - Degree-list chunk now expects `[]` (all 6 majors + education/experience excluded - the old fixture's "statistics" exception was itself flagged earlier as leaky/arbitrary ground truth, since it depended on knowledge from a different sentence in the full posting not available when this chunk is scored alone).
  - Pure business-domain headline labels dropped from expectations (`insurance pricing`, `call center optimization`, `usage-based insurance`/`ubi`, `staffing`), consistent with the already-decided technical-only skills-section policy; recognized technical/scientific fields (`machine learning`, `intelligent document processing and information retrieval`) remain expected even though broad-sounding.
  - `"/"`-joined lists split into separate atomic terms (`speech analytics`+`text analytics`, `monitoring drift`+`seasonality`), matching the existing GLM/GBM precedent. `complex decision making` dropped as an abstract/soft-responsibility phrase. `classification and summarization` split into 2 atomic terms.
  - Cleared all previous reporting-only benchmark artifacts under `build/benchmarks/` (`quality_judges_benchmark.json`, `extraction_prompt_iteration_{1..5}.json`, `extraction_prompt_decontaminated_{baseline,v2}.json`, `parallel_classifier_n{1,3}.json`, the old `big_section_parser_benchmark.json`), since none reflected the new methodology and none were hard test scores.
  - **Fresh F1 measurement**: 72.62% mean F1 (precision 0.8714, recall 0.6517) for `orchestra_single_shot` against the revised fixture - not yet re-gated as a hard pass/fail threshold (`MIN_ORCHESTRA_F1_SCORE = 0.90` remains reporting-only pending a decision on what threshold is realistic against the new fixture).
  - **Categorization/gap-coverage review (per user's step 2)**: every included/excluded term in the revised fixture fits conceptually into exactly one of the 3 existing extraction classifiers' jurisdiction (`degree_context`, `domain_vs_technical`, `soft_skill`) or needs none - no term was found that structurally falls outside all 3, so there's no evidence a 4th classifier is needed. The real gaps are calibration problems *within* existing branches (see below), plus two issues entirely outside the 3-classifier framing.
  - **Concrete failure patterns found from the fresh run** (not sampling noise - each is a repeatable, structural pattern): (1) the `soft_skill` classifier/Stage-1 decomposition dropped `peer review`/`technical documentation`/`version control`/`writing tests` entirely from the Python/Git chunk (recall 0.33 on that chunk alone) - the same "process vs. technical skill" instability flagged repeatedly during the judge era, now confirmed as a real recall hit under hard F1 too; (2) Stage-1 decomposition's `"/"`-list splitting is inconsistent - GLM/GBM and monitoring-drift/seasonality split correctly, but `speech/text analytics` and `staffing/queueing concepts` did not, letting the excludable "staffing" half ride along attached to valid "queueing concepts" as one bundled candidate; (3) `domain_vs_technical` mis-scored two recognized-field-vs-domain-label edge cases (`telematics`, `portfolio impact measurement` both excluded, contradicting the intended recognized-technical-field exemption); (4) a `SemanticMatcher` false positive, unrelated to any extraction classifier, resolved an ungrounded candidate (likely `classification`) to the `machine learning` canonical entry in the Intelligent Document Processing chunk - a matching-layer precision issue, not an extraction-classification gap.
- [done] **Switched F1 scoring from exact-string matching to matcher-based matching (2026-07-15, same day)**: exact string equality between `expected_terms` and `observed_terms` was brittle by construction - exactly the wording-variance problem `ExactAliasMatcher`/`SemanticMatcher` already exist to solve. Rewrote the F1 test's `_score_parser` to build a tiny ad hoc skill cache per case (`SkillRecord` per expected term) and resolve each observed term via the same tiered `ExactAliasMatcher` -> `SemanticMatcher` path production uses. **Result: mean F1 jumped 72.62% -> 85.32%** - precision became a perfect 1.0 across all 7 chunks, since every apparent "false positive" was actually just a wording/splitting difference (e.g. `speech/text analytics` now correctly multi-matches both `speech analytics` and `text analytics`). Remaining recall gaps became a much cleaner signal of genuine misses, no longer confounded by measurement artifacts.
- [done] **Fixture editorial refinement + F1 improved to 89.60% (2026-07-15, same day)**: per user's reasoning that `git` (a specific version-control tool) already implies `version control` in a tight resume skills section, and that `peer review`/`technical documentation` read more like experience-bullet material than skills-section keywords (unlike `writing tests`, commonly a standalone skills-section item like "Unit Testing"), dropped `peer review`/`technical documentation`/`version control` from the Python/Git chunk's `expected_terms` (kept `writing tests`). New mean F1: **89.60%**.
- [done] **Deprecated the LLM-judge evaluation framework entirely (2026-07-15, same day, per explicit user decision)**: `sample_big_section_sentence_cases.yaml` + matcher-based F1 scoring is now the primary/authoritative extraction-quality test, superseding the judge. Deleted `tests/evals/quality_judges.py`, `test_quality_judges.py`, `test_quality_judges_grounding.py`, `extraction_prompt_benchmark.py`, `parallel_classifier_benchmark.py`, `parallel_classifier_pipeline.py` (fully superseded by production `src/parser/parallel_extraction.py`), and the unused `benchmark_scoring.py` helper (0 references anywhere). Updated the stale `quality_judges.py` docstring reference in `src/parser/parallel_extraction.py` to not point at a deleted file.
- [done] **Refactored matching into its own top-level package, `src/matcher/` (2026-07-15, same day, per explicit user decision that "matcher is its own responsibility")**: split `src/parser/matching.py` into `src/matcher/base.py` (`MatchCandidate`, `Matcher` ABC), `exact_alias.py` (`ExactAliasMatcher`), `semantic.py` (`SemanticMatcher`), `grounding.py` (`LLMGroundingMatcher`); moved `src/parser/models.py` -> `src/matcher/models.py` and `src/parser/embedding_cache.py` -> `src/matcher/embedding_cache.py` unchanged. `src/matcher/` only operates on in-memory `SkillRecord` sequences passed in by the caller - no YAML/file I/O, no chunking/extraction knowledge. `src/parser/__init__.py` does **NOT** re-export matcher classes anymore (clean break, per explicit user decision) - `base.py`/`orchestra_single_shot.py`/`selection.py` now import matching classes directly `from matcher import ...`. Skill-cache YAML *loading* stays in `parser/base.py` (parser's I/O responsibility); only the `SkillRecord` data contract moved.
- [done] **Reorganized tests so `tests/evals/` is data-only (2026-07-15, same day, per explicit user decision that evals was originally meant for test data, not test cases)**: new `tests/matcher/` (mirrors `src/matcher/`, holds the moved `test_matching.py`); `tests/parse_posting/test_big_section_skill_coverage.py` moved from `tests/evals/` (it's a parser-quality test); new `tests/main/` (mirrors `src/main.py`) holds the moved `test_run_metrics_logging.py`. Deleted `tests/evals/test_big_section_skill_coverage_openai.py` (per user decision - duplicated the F1 test with a stricter, permanently-failing exact-set assertion) and the confirmed-unused `sample_big_section_expected_skills.yaml` fixture; kept the also-unused `sample_job_posting_big.txt` per explicit user decision to retain it for future use. Remaining `tests/evals/` fixtures: `sample_big_section_sentence_cases.yaml`, `sample_expected_skills.yaml`, `sample_job_posting.txt`, `sample_job_posting_big.txt`. Full suite (55 tests) passes; end-to-end pipeline smoke-tested via `src/main.py`.

## Benchmark Snapshot
- Multishot exact-match rate on `tests/evals/sample_big_section_sentence_cases.yaml`: 85.71%
- Single-shot exact-match rate on `tests/evals/sample_big_section_sentence_cases.yaml`: 42.86%
- Benchmark artifact: `build/benchmarks/big_section_parser_benchmark.json`

## Parser Strategy Decision (current)
- Default parser strategy is now `orchestra_single_shot` (deterministic-only chunk splitting; each chunk gets its own independent, self-contained extraction+cache-match call, run concurrently, with self-consistency voting across `num_votes` samples).
- Rationale, from `build/benchmarks/experimental_parser_variants.json` (measured before the multishot family was retired):
  - Atomic per-sentence F1: orchestra_single_shot 0.904 > single_shot 0.888 > multishot_v1 0.851 > multishot_v2 0.752.
  - Combined multi-bullet posting F1: multishot_v1 and orchestra_single_shot tied at 0.877; single_shot collapsed to 0.205 (extracts coarse per-bullet phrases instead of decomposing them when forced to process a whole multi-bullet posting in one call). single_shot is therefore only used for already-atomic input.
  - Loosening multishot v1's discard/grounding filters (`MultiShotPostingParserV1Loose`) was tested as a worst-case ablation: precision dropped sharply (1.0 -> 0.771 on the combined posting) for only a marginal recall gain, and let through exactly the noise the filters exist to catch (degree-major words like "mathematics"/"engineering"/"AI"). This is why `OrchestraSingleShotParser` keeps the same discard/grounding filtering rather than relaxing it.
- Given orchestra_single_shot matched-or-beat multishot_v1 everywhere tested and is simpler/cheaper (no chunk-splitting LLM call), the multishot family (`MultiShotPostingParserV1`, `MultiShotPostingParserV2`, `MultiShotPostingParserV1Loose`) and their LLM-based re-chunking helpers were deleted rather than kept as unused comparison code. `DeterministicPostingParser` remains as shared cache-loading/matching infrastructure and the offline (no-LLM) fallback (`use_llm=False`); it was not deleted.
- These were single-run measurements with known LLM sampling variance; repeat runs would strengthen confidence further, but the direction of each finding was large and consistent.

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