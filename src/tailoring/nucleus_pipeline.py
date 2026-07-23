"""Phase 3 replacement: whole-posting-seeded nucleus generation, synthesized
directly into bullet text. Replaces `tailoring.claims`/`tailoring.claim_discovery`
for the real generation pipeline (those modules are kept, unmodified, as the
legacy group-then-narrate design still exercised by their own benchmark
scripts - see docs/agent/BULLET_TAILORING_DEV_PLAN.md's Phase 3 superseded
note).

Design history (2026-07-22):

1. First landing: nucleus generation was scoped PER POSTING SENTENCE (one
   LLM call per sentence, word-for-word porting
   `scratch/phase3_9_spike19_result_guardrail.py`'s prompt), replacing the
   earlier "additive guardrail graft onto the old prompt" attempt (which
   regressed atomicity - see the dev plan's Phase 3.9 continued section).
2. THIS revision: a live e2e run against the real project/posting (19
   sentences, 26 claims, 970.7s, 9 nucleus calls at ~38k reasoning tokens)
   surfaced two problems, diagnosed together rather than patched
   separately: (a) cost/latency - one gpt-5/"high" call per relevant
   sentence does not scale; (b) heavy cross-sentence duplication - a
   generically-applicable fact (e.g. progress-reporting facts) matches
   MANY different sentences' own independent retrieval queries, and since
   each sentence's nucleus call has zero visibility into any other
   sentence's output, the same underlying fact(s) independently produced
   near-identical nuclei under several different sentences (observed: 4
   exact-same-fact-set clusters of 3 near-duplicate claims each, ~46% of
   all generated claims). Per explicit decision, both problems are fixed
   by the SAME change: seed nucleus generation from the ENTIRE posting (all
   requirement sentences together) in ONE call, retrieving ONE whole-
   posting candidate fact pool, and asking for exactly 3 MUTUALLY DISTINCT
   themes - the model can now see everything at once and deliberately
   avoid redundancy, rather than each of many isolated calls guessing at a
   theme with no cross-call awareness. This collapses N calls into 1 and
   removes the structural cause of the duplication, rather than adding a
   dedup pass after the fact.

- `_NUCLEUS_SYSTEM_PROMPT`'s validated GUARDRAIL language (achieved-outcome
  fabrication, invented-mechanism fabrication, the `result` definition) is
  carried over UNCHANGED from `scratch/phase3_9_spike19_result_guardrail.py`
  - only the FRAMING (one sentence -> the whole posting) and the
  candidate-count instruction (1-3 -> exactly 3, with an explicit mutual-
  distinctness requirement) were adapted, since those are the parts that
  necessarily depend on the sentence-vs-posting scoping decision.
- Synthesis (turning a nucleus into actual bullet text) reuses
  `tailoring.verification.synthesize_proposal` UNCHANGED - its
  `_SYNTHESIS_SYSTEM_PROMPT` is already the spike22-validated design (per
  the dev plan's "Result: production integration landed" correction), so
  there is nothing to re-port there. `synthesize_proposal` was given one
  small, backward-compatible plumbing change (tolerating a claim object
  with no `claim_text`) rather than duplicated here.
- `PostingNucleusClaim` (in `tailoring.models`) is a NEW, narrower
  dataclass, not a variant of `CoreClaimMolecule` - it has no `claim_text`
  (the why/result nucleus IS the claim) and no `primary_proof` (a caller
  needing a proof string for Phase 6's overlap check should use `result`
  when present, falling back to a cited fact's own text otherwise, same
  fallback the orchestrating script already used for the per-sentence
  design). Unlike the first revision's `SentenceNucleusClaim`, there is no
  `source_requirement_sentence` field at all - a claim seeded from the
  WHOLE posting has no single seeding sentence to attribute. `target_skills`
  is derived DETERMINISTICALLY (no LLM call) as the union of `skill_tags`
  across a claim's own cited facts - explicitly accepted as overinclusive.
- No ranking/selection step (unlike `tailoring.claims.rank_core_claim_molecules`):
  every nucleus this module generates is synthesized into a proposal and
  handed to the caller; downstream verification/Phase 6 competition (both
  unchanged, called separately by the orchestrating script) are the only
  filters.

NOT yet decided: which model/reasoning-effort tier to run nucleus
generation at in production. `scratch/phase3_9_spike19_result_guardrail.py`
validated its ORIGINAL per-sentence prompt on `gpt-5` at
`reasoning_effort="high"` (`NUCLEUS_REASONING_EFFORT` here matches that
default), but this module's adapted whole-posting prompt/schema has not
itself been independently spike-validated the same rigorous way (single
real-project e2e run only, not a dedicated fixture package or repeated-
trial consistency check).

Live-validated (2026-07-22, real project/posting, full e2e chain via
`tests/tailoring/end_to_end_benchmark.py`): 1 nucleus call, 3 genuinely
distinct claims with zero fact overlap between them, 6,336 reasoning
tokens - versus the per-sentence design's 9 calls/38,464 reasoning tokens
for the same posting. Full run: 119.0s (versus 970.7s, ~8x faster), 18
verification-tier calls (versus 124). 2/3 proposals passed verification
cleanly; the third correctly failed as `bad_wording` for substantially
restating the protected baseline bullet's own content - a genuine catch,
not a regression. Zero duplicate clusters (versus 4 clusters/46%
redundancy before). See the dev plan's Phase 3 replacement Result section
for the full write-up.

3. THIS revision (2026-07-23): the "exactly 3, mutually distinct" count
   requirement was replaced with a relaxed 1-20 range (`MIN_NUCLEUS_CANDIDATES`/
   `MAX_NUCLEUS_CANDIDATES`), "preferring fewer, stronger ones" rather than
   a hard mutual-distinctness rule - live spiking (`scratch/nucleus_breadth_spike.py`)
   found the mutual-distinctness requirement was pushing nucleus generation
   toward artificially broad, multi-subsystem umbrellas to hit exactly 3
   themes from a large fact pool; relaxing the count let coherent, narrow
   themes emerge naturally instead (facts-per-nucleus dropped from an
   observed max of 9 to typically 1-4). The prompt was also reframed
   explicitly as a RESUME BULLET generator (not an abstract "why" generator)
   with two new instructions: `why` is scaffolded BY the facts but need not
   be derivable from them alone (may require reasonable inference), and
   `result` must be a QUANTIFIABLE, already-achieved outcome specifically
   (not just "directly evidenced" - a qualitative claim like "more robust"
   belongs in `why`, never forced into `result`). `synthesize_proposal`
   (`tailoring.verification`) was independently tuned in the same round:
   an explicit instruction that facts are unfit for verbatim inclusion (the
   bullet must be written in the synthesizer's own words), an instruction
   to infer high-level motivations rather than list facts, and a final
   self-review pass ("validate the flow... reword if needed"). It also now
   accepts an optional `project_summary` (from `ProjectBaseline.project_summary`)
   so a bullet can be self-contained for a reader with no project context.
   Live-validated on the real project against 2 different real postings
   (the original AI-heavy `llm_ml_infra` posting and a generic, non-AI
   `backend_platform` entry-level posting) - both produced well-formed,
   individually coherent nuclei with no crashes; the non-AI posting
   surfaced a (lower-severity) residual duplication risk where the single
   strongest/most-quantified fact gets reused as the primary basis for
   multiple different why-framings when few other facts are strong
   candidates - accepted as a known, lower-severity trade-off of dropping
   the mutual-distinctness requirement, deferred to Phase 6's competition/
   local-distinctness scoring to resolve downstream rather than re-adding
   an explicit distinctness rule here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from llm import LLMProvider

from tailoring.models import (
    AnnotatedProposal,
    FactAtom,
    JobRequirements,
    PostingNucleusClaim,
    ProjectFactMatch,
)
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements
from tailoring.verification import VERIFICATION_REASONING_EFFORT, synthesize_proposal

# Matches scratch/phase3_9_spike19_result_guardrail.py's validated tier
# (STRONG_MODEL="gpt-5", STRONG_REASONING_EFFORT="high") for its ORIGINAL
# per-sentence prompt - not yet re-validated at any tier for this module's
# adapted whole-posting prompt specifically.
NUCLEUS_REASONING_EFFORT = "high"

# Relaxed range (was "exactly 3, mutually distinct") - live spiking found
# the hard count + distinctness requirement pushed generation toward
# artificially broad, multi-subsystem umbrella nuclei to hit exactly 3 from
# a large fact pool. The prompt itself now says "preferring fewer, stronger
# ones" instead of enforcing distinctness as a hard rule.
MIN_NUCLEUS_CANDIDATES = 1
MAX_NUCLEUS_CANDIDATES = 20

_NUCLEUS_JSON_SCHEMA = {
    "name": "resume_bullet_candidates",
    "schema": {
        "type": "object",
        "properties": {
            "posting_interpretation": {"type": "string"},
            "possible": {"type": "boolean"},
            "candidate_bullets": {
                "type": "array",
                "maxItems": MAX_NUCLEUS_CANDIDATES,
                "items": {
                    "type": "object",
                    "properties": {
                        "why": {"type": "string"},
                        "result": {"type": "string"},
                        "supporting_fact_ids": {"type": "array", "items": {"type": "string"}},
                        "strength_rationale": {"type": "string"},
                    },
                    "required": ["why", "result", "supporting_fact_ids", "strength_rationale"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["posting_interpretation", "possible", "candidate_bullets"],
        "additionalProperties": False,
    },
}

# Ported word-for-word from scratch/nucleus_breadth_spike.py (2026-07-23
# tuned version) - do not edit without re-validating against that spike's
# own real-data runs first (this repo's hygiene rule). Supersedes the
# earlier "exactly 3, mutually distinct" design adapted from
# scratch/phase3_9_spike19_result_guardrail.py; see this module's docstring
# for the full history of what changed and why.
_NUCLEUS_SYSTEM_PROMPT = (
    "You propose candidate `why` themes for an ENTIRE job posting (all of its requirement/responsibility "
    "sentences together, given below), drawing on a pool of candidate facts about a real project already "
    "matched as relevant to this posting.\n\n"
    "Your goal is to ultimately propose a set of ideas for resume bullets that would be compelling to a"
    " hiring manager reading this posting. Whys targetting intent and principle behind the job posting are usually more compelling than narrow, implementation-level details. A good resume bullet point will put most of the emphasis on the why, rather than the facts supporting the why.\n\n"
    "First, in `posting_interpretation`, state your own read on what this posting, taken as a whole, is really "
    "probing for, beyond its literal keyword list.\n\n"
    "- IMPORTANT: facts are ultimately scaffolding for the `why` theme, not the other way around. Whys themselves must be strong, and may require inference beyond the given facts. The posting interpretation is a good place to start, but do not rely on it exclusively.\n"
    "- A `why` is a general, ABSTRACT motivating principle or DESIGN INTENT - things like 'long-term "
    "maintainability', 'engineering rigor and quality discipline', or 'designing with future scalability in "
    "mind'. It is NOT a description of what was technically built, not a mini-summary of the facts, and must "
    "not name any specific tool, technology, or component - that level of detail belongs to a later stage, "
    "not here.\n"
    "- CRITICAL: `why` must describe INTENT or PRINCIPLE, never an achieved or observed behavioral outcome "
    "that isn't directly evidenced. Phrasing like 'to maintain responsiveness under load', 'to ensure X stays "
    "stable', or 'keeping Y reliable' asserts that X/Y was actually observed or measured - that is a `result`-"
    "shaped claim, not a `why`, and must NEVER appear in `why` unless a fact literally states that outcome was "
    "measured or observed.\n"
    "- CRITICAL, separately from the above: `why` must never assert a SPECIFIC MECHANISM, architecture, or "
    "implementation detail that isn't stated by the facts, even when phrased as forward-looking intent rather "
    "than an achieved outcome. The test: could this detail be swapped for a different, equally plausible "
    "implementation without contradicting the stated facts? If yes, it is an INVENTED mechanism, not something "
    "the facts establish - leave it out, regardless of how natural or common that mechanism would be for this "
    "kind of feature. A `why` MAY draw out the inherent purpose of an already-stated capability - something "
    "that capability is definitionally FOR, not a separate technical choice about it (e.g. a cache's inherent "
    "purpose is reducing redundant work; persisted historical records inherently support later retrieval/"
    "audit) - but it may NOT invent a new, separately-falsifiable mechanism the capability doesn't itself "
    "imply (e.g. 'can trigger a pipeline run' does not imply queueing, background workers, or any particular "
    "execution model - triggering a run could equally be synchronous or asynchronous, so asserting either one "
    "is invention, not inherent purpose).\n\n"
    "- A `result` is a concrete, quantifiable, ALREADY-ACHIEVED outcome. A result must be directly evidenced by a fact that literally states this outcome happened or was measured (a number, an explicit before/after comparison, an explicit test/benchmark result). If no such fact exists, leave `result` empty. It is normal and expected for a good `why` to stand alone with no separate result; do not force one.\n\n"
    "- A posting sentence's literal keywords are a weak signal for what is actually resume-strong for that "
    "theme. Infer the broader underlying intent behind the posting as a whole.\n\n"
    "IMPORTANT: if the given candidate facts do NOT genuinely, honestly support any coherent theme for this "
    "posting, set `possible` to false and return an EMPTY `candidate_bullets` list. Do not force a fabricated "
    "or strained connection just to produce an answer.\n\n"
    f"Propose between {MIN_NUCLEUS_CANDIDATES} and {MAX_NUCLEUS_CANDIDATES} mutually distinct candidate whys, preferring fewer, stronger ones.\n\n"
    "Whys should be motivated either by a specific sentence from the posting or by a coherent theme across multiple posting sentences."
    "Whys will be later screened and only a small subset will be selected. Don't prioritize distinctness over quality: ensure that each why reads as a whole, strong point.\n"
)


def generate_posting_nucleus_claims(
    project_id: str,
    requirements: JobRequirements,
    candidate_facts: Sequence[FactAtom],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = NUCLEUS_REASONING_EFFORT,
) -> Tuple[List[PostingNucleusClaim], str, bool]:
    """Propose 0-20 candidate why/result nuclei for the ENTIRE posting,
    preferring fewer, stronger ones over a fixed count, scoped to
    `candidate_facts` (the caller's own whole-posting retrieval result -
    this function does no retrieval itself).

    Returns `(claims, posting_interpretation, possible)`. `claims` uses
    TEMPORARY, per-call-local ids (`{project_id}_claim_01`, ...) that a
    caller MAY still need to renumber if combined with other claims from
    elsewhere - kept for interface consistency with the module's own
    orchestration function below, even though a single whole-posting call
    has no cross-call collision risk on its own.

    A candidate whose `supporting_fact_ids` is empty or cites ANY id not
    in `candidate_facts` is a generation-time validation failure and is
    dropped entirely (not partially filtered) - there is no
    `non_advancement_reason` field on `PostingNucleusClaim` to flag a
    retained-but-invalid claim, unlike `CoreClaimMolecule`.
    """

    if not candidate_facts:
        return [], "", False

    fact_atoms_by_id = {atom.id: atom for atom in candidate_facts}
    fact_lines = "\n".join(f"- {atom.id}: {atom.fact}" for atom in candidate_facts)
    sentences_block = (
        "\n".join(f"- {sentence_match.sentence}" for sentence_match in requirements.requirement_sentences)
        or requirements.summary_paragraph
        or "(no per-sentence breakdown available)"
    )
    prompt = (
        f"Job posting requirement/responsibility sentences:\n{sentences_block}\n\n"
        f"Candidate facts retrieved for this posting (fact_id: text):\n{fact_lines}\n\n"
        "Propose candidate themes, or honestly report none are possible."
    )
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_NUCLEUS_SYSTEM_PROMPT,
        json_schema=_NUCLEUS_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )

    posting_interpretation = response.get("posting_interpretation", "")
    possible = bool(response.get("possible", False))
    if not possible:
        return [], posting_interpretation, possible

    claims: List[PostingNucleusClaim] = []
    for index, candidate in enumerate(response.get("candidate_bullets", [])[:MAX_NUCLEUS_CANDIDATES], start=1):
        raw_fact_ids = candidate.get("supporting_fact_ids", [])
        if not raw_fact_ids or any(fact_id not in fact_atoms_by_id for fact_id in raw_fact_ids):
            continue
        supporting_fact_ids = tuple(dict.fromkeys(raw_fact_ids))

        # Deterministic, explicitly overinclusive - see PostingNucleusClaim's docstring.
        target_skills = tuple(
            dict.fromkeys(tag for fact_id in supporting_fact_ids for tag in fact_atoms_by_id[fact_id].skill_tags)
        )

        claims.append(
            PostingNucleusClaim(
                id=f"{project_id}_claim_{index:02d}",
                project_id=project_id,
                supporting_fact_ids=supporting_fact_ids,
                target_skills=target_skills,
                rationale=candidate.get("strength_rationale", ""),
                why=candidate.get("why", ""),
                result=candidate.get("result", ""),
            )
        )
    return claims, posting_interpretation, possible


def discover_and_synthesize_posting_nuclei(
    project_id: str,
    fact_atoms: Sequence[FactAtom],
    fact_atoms_by_project: Dict[str, Sequence[FactAtom]],
    protected_fact_ids: Set[str],
    requirements: JobRequirements,
    nucleus_llm_provider: LLMProvider,
    synthesis_llm_provider: LLMProvider,
    embedding_llm_provider: Optional[LLMProvider] = None,
    nucleus_reasoning_effort: Optional[str] = NUCLEUS_REASONING_EFFORT,
    synthesis_reasoning_effort: Optional[str] = VERIFICATION_REASONING_EFFORT,
    project_summary: str = "",
) -> Tuple[List[PostingNucleusClaim], List[AnnotatedProposal], List[ProjectFactMatch]]:
    """Full posting -> nucleus -> synthesis chain for one project against
    one posting. ONE retrieval call (the posting's whole flattened
    target-skill list, reusing `retrieve_project_fact_pool` unchanged) and
    ONE nucleus-generation call - not one per sentence. No ranking/
    selection - every synthesized proposal is returned; verification
    (unchanged, `tailoring.verification.verify_proposal`) and Phase 6
    competition are the caller's responsibility.

    `project_summary` (additive): passed straight through to every
    `synthesize_proposal` call (from `ProjectBaseline.project_summary`),
    so each synthesized bullet can be self-contained for a reader with no
    prior project context. Empty string (default) omits it, same as
    `synthesize_proposal`'s own default.
    """

    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    target_skills = target_skills_from_requirements(requirements)
    matches = retrieve_project_fact_pool(
        project_id, fact_atoms_by_project, protected_fact_ids, target_skills, llm_provider=embedding_llm_provider
    )

    candidate_atoms = [
        fact_atoms_by_id[match.fact_id] for match in matches if match.included and match.fact_id in fact_atoms_by_id
    ]
    if not candidate_atoms:
        return [], [], matches

    claims, _posting_interpretation, _possible = generate_posting_nucleus_claims(
        project_id, requirements, candidate_atoms, nucleus_llm_provider, reasoning_effort=nucleus_reasoning_effort
    )

    proposals = [
        synthesize_proposal(
            claim,
            fact_atoms_by_id,
            synthesis_llm_provider,
            reasoning_effort=synthesis_reasoning_effort,
            project_summary=project_summary,
        )
        for claim in claims
    ]

    return claims, proposals, matches


def posting_nucleus_claims_to_dicts(claims: Sequence[PostingNucleusClaim]) -> List[dict]:
    return [
        {
            "id": claim.id,
            "project_id": claim.project_id,
            "supporting_fact_ids": list(claim.supporting_fact_ids),
            "target_skills": list(claim.target_skills),
            "rationale": claim.rationale,
            "why": claim.why,
            "result": claim.result,
        }
        for claim in claims
    ]


def write_posting_nucleus_claims_json(claims: Sequence[PostingNucleusClaim], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(posting_nucleus_claims_to_dicts(claims), handle, indent=2)
