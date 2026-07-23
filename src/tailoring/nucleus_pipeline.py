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
itself been independently spike-validated the same way - it is a direct,
reasoned adaptation, not a byte-for-byte port, so its own behavior should
be re-inspected on real data (this module's docstring will be updated with
that live-validation result once run).
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

# Exactly 3 whenever the facts genuinely support 3 mutually distinct themes -
# see _NUCLEUS_SYSTEM_PROMPT's own honesty escape valve for fewer/zero.
MAX_NUCLEUS_CANDIDATES = 3

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

# Adapted from scratch/phase3_9_spike19_result_guardrail.py's
# _NUCLEUS_SYSTEM_PROMPT: guardrail language (achieved-outcome/invented-
# mechanism fabrication, the `result` definition) is UNCHANGED; only the
# framing (one sentence -> the whole posting) and the candidate-count
# instruction (1-3 -> exactly 3, mutually distinct) were adapted. Do not
# edit the guardrail paragraphs without re-validating against spike19's own
# reproduction cases first (this repo's hygiene rule).
_NUCLEUS_SYSTEM_PROMPT = (
    "You propose candidate `why` themes for an ENTIRE job posting (all of its requirement/responsibility "
    "sentences together, given below), drawing on a pool of candidate facts about a real project already "
    "matched as relevant to this posting.\n\n"
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
    "Example (BAD - achieved-outcome fabrication): facts state only 'the webapp includes a FastAPI backend' "
    "and 'the webapp can trigger pipeline runs' - why='Operational decoupling of user-facing requests from "
    "heavier pipeline execution to maintain responsiveness under load.' WRONG: no fact establishes any "
    "performance measurement or load test.\n"
    "Example (BAD - invented mechanism dressed as intent, same facts): why='API design intended to support "
    "triggering longer-running work without blocking the interface, with an eye toward future scale.' STILL "
    "WRONG even though it avoids claiming an achieved outcome: 'without blocking the interface' asserts a "
    "specific execution model (non-blocking/async) the facts never state - triggering a run could just as "
    "easily be a normal synchronous call.\n"
    "Example (GOOD - same facts, stays within what's actually stated): why='Providing an API-driven interface "
    "for initiating pipeline work programmatically.' result=(none). This restates the capability's own stated "
    "purpose (an API that can trigger runs) without inventing how the execution is implemented.\n"
    "Example (GOOD - inherent purpose of a stated capability, not an invented mechanism): fact states 'the "
    "webapp can manage the skills cache' - why='Reducing redundant repeated work by centralizing shared state "
    "in a managed cache.' This is fine: avoiding redundant work is what a cache inherently exists to do, not "
    "a separate invented technical detail.\n"
    "Example (GOOD - a result that IS directly evidenced): fact states 'Fixed staged pipeline achieved 94.53% "
    "F1 over five trials.' why='Evaluating alternative designs empirically before committing to production.' "
    "result='Achieved 94.53% F1 over five trials.' - directly grounded in a fact that literally states it.\n\n"
    "- A `result` is a concrete, ALREADY-ACHIEVED outcome, and it must be DIRECTLY evidenced: before including "
    "one, you must be able to point to a SPECIFIC given fact that literally states this outcome happened or "
    "was measured (a number, an explicit before/after comparison, an explicit test/benchmark result). If no "
    "such fact exists, leave `result` empty. It is normal and expected for a good `why` to stand alone with no "
    "separate result; do not force one.\n\n"
    "- A `why`'s scope is about ABSTRACTION, not about how many facts or technologies support it. A `why` can "
    "legitimately be supported by many different facts spanning multiple technologies, as long as they all "
    "genuinely exemplify the SAME single abstract principle. Do not narrow down to fewer facts or fewer "
    "technologies for its own sake - only avoid grouping facts that don't actually share the same underlying "
    "theme.\n"
    "- The 3 candidates must be MUTUALLY DISTINCT: no two candidates may be built primarily from the same "
    "fact(s), and no two may represent the same underlying theme just phrased differently. If two ideas would "
    "essentially restate the same accomplishment or lean on the same evidence, merge them into ONE candidate "
    "instead of listing both separately - a fact may still be cited by more than one candidate only when each "
    "candidate's own PRIMARY basis is genuinely different.\n"
    "- A posting sentence's literal keywords are a weak signal for what is actually resume-strong for that "
    "theme. Infer the broader underlying intent behind the posting as a whole.\n\n"
    "IMPORTANT: if the given candidate facts do NOT genuinely, honestly support any coherent theme for this "
    "posting, set `possible` to false and return an EMPTY `candidate_bullets` list. Do not force a fabricated "
    "or strained connection just to produce an answer.\n\n"
    "Propose exactly 3 candidate whys whenever the facts genuinely support 3 mutually distinct themes (see "
    "above). If the facts only genuinely support fewer than 3 truly distinct themes, propose fewer - even zero "
    "- rather than inventing a forced or redundant one just to reach 3. For each, explain in "
    "`strength_rationale` why this theme would be compelling to a hiring manager reading this posting.\n\n"
    "First, in `posting_interpretation`, state your own read on what this posting, taken as a whole, is really "
    "probing for, beyond its literal keyword list."
)


def generate_posting_nucleus_claims(
    project_id: str,
    requirements: JobRequirements,
    candidate_facts: Sequence[FactAtom],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = NUCLEUS_REASONING_EFFORT,
) -> Tuple[List[PostingNucleusClaim], str, bool]:
    """Propose 0-3 MUTUALLY DISTINCT why/result nuclei for the ENTIRE
    posting, scoped to `candidate_facts` (the caller's own whole-posting
    retrieval result - this function does no retrieval itself).

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
) -> Tuple[List[PostingNucleusClaim], List[AnnotatedProposal], List[ProjectFactMatch]]:
    """Full posting -> nucleus -> synthesis chain for one project against
    one posting. ONE retrieval call (the posting's whole flattened
    target-skill list, reusing `retrieve_project_fact_pool` unchanged) and
    ONE nucleus-generation call - not one per sentence. No ranking/
    selection - every synthesized proposal is returned; verification
    (unchanged, `tailoring.verification.verify_proposal`) and Phase 6
    competition are the caller's responsibility.
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
        synthesize_proposal(claim, fact_atoms_by_id, synthesis_llm_provider, reasoning_effort=synthesis_reasoning_effort)
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
