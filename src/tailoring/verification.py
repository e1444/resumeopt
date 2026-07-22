"""Phase 5: proposal synthesis, verification, and typed repair.

Neither `CoreClaimMolecule` (Phase 3) nor `ExpandedClaimMolecule` (Phase 4)
carries actual bullet text - Phase 4 deliberately deferred text-authoring
(see its module docstring). This phase is where that happens:

1. `synthesize_proposal` - ONE bounded LLM call turns a core claim plus its
   expansion decision into a fluent `AnnotatedProposal.proposal_text`,
   using only the cited facts. This is generation, but grounded and
   immediately checked, not free-form (per AGENTS.md: "Use LLMs for
   extraction and judgment, not uncontrolled generation").
2. `verify_proposal` - a DETERMINISTIC protected-fact-reuse check first
   (cheap, short-circuits before any LLM call if it fires - this proposal
   should never have reached verification if Phase 2 excluded protected
   facts correctly, but this is a defense-in-depth check, not reliance on
   upstream correctness alone), then up to 4 narrow, single-purpose LLM
   classifiers in a fixed order that doubles as failure-type priority:
   fact_support (-> `hallucination`), same_claim_integrity (-> `bad_flow`),
   semantic_duplication then project_relevance (both -> `bad_wording`).
   Each classifier's own verdict is `yes`/`no`/`idk` (not merely boolean) -
   `yes` always means "this classifier's own problem is present" (a hard
   failure), `no` means "no problem", regardless of what the specific
   English question each system prompt asks reads like on its surface.
   verification's own status is genuinely 3-way (`pass`/`idk`/`fail`) per
   the dev plan, and `idk` must stay visible, never coerced into
   acceptance or rejection. Processing stops at the first `yes` found (a
   hard failure short-circuits remaining checks); an `idk` does NOT
   short-circuit, since a later classifier's `yes` should still win.
3. `repair_proposal` - one bounded repair attempt per DISTINCT failure
   type actually encountered (never retrying the same type twice),
   reverifying after each attempt via `verify_proposal` itself, so the
   natural classifier order above already enforces the dev plan's fixed
   repair sequence (`hallucination` -> `bad_flow` -> `bad_wording`).
   Discards (stays `fail`) on a repair that doesn't resolve its own
   target failure, or immediately on `unresolvable` (protected-fact reuse
   can never be repaired - repair may not retrieve facts or change
   project context, so there is nothing a rewording could fix).

   Phase 5.1: BEFORE any rewrite is attempted for a given failure, an
   explicit 2-stage resolvability gate decides HOW to attempt the fix -
   `resolvable_by_editing_alone` (keep every currently-cited fact, just
   reword), then, only if that is not `yes`, `resolvable_by_removing_facts`
   (drop specific currently-cited fact(s), naming which, then reword using
   only the rest). If neither is viable, the failure becomes `unresolvable`
   immediately - no rewrite prompt is ever called, and no further repair
   attempts of any kind are made. This makes fact-dropping an explicit,
   auditable decision (`RepairStep.resolution`/`removed_fact_ids`) instead
   of an implicit rewrite side-effect, and lets a `remove_facts` repair's
   `AnnotatedProposal.supporting_fact_ids` be pruned deterministically
   rather than left stale - a gap Phase 5's own live benchmark documented
   and left open. An `idk` from either gate classifier is treated as `no`
   for dispatch (never assume repairability from uncertainty).

Per the Phase 3.7 hygiene rule, every anchor example below is a fully
invented scenario, not copied or paraphrased from this module's own
fixtures or the real project's data.

Reasoning-tier classifiers default to `reasoning_effort="low"` from the
start (not the project-wide `"minimal"` default) - Phase 3.6/3.7 found
`"minimal"` can intermittently fail to credit information already present
in its own input, which is exactly the kind of reliability gap a
verification gate cannot afford.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from llm import LLMProvider

from tailoring.models import (
    AnnotatedProposal,
    BaselineBullet,
    CoreClaimMolecule,
    ExpandedClaimMolecule,
    FactAtom,
    RepairResolution,
    RepairStep,
    RepairType,
    VerificationResult,
    VerificationStatus,
)

VERIFICATION_REASONING_EFFORT = "low"

# Fixed repair sequence per the dev plan. "unresolvable" is deliberately
# excluded - it is never attempted, only ever discarded immediately.
_REPAIRABLE_TYPES: Tuple[RepairType, ...] = ("hallucination", "bad_flow", "bad_wording")

_VERDICT_JSON_SCHEMA = {
    "name": "verification_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no", "idk"]},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
}

_REPAIR_JSON_SCHEMA = {
    "name": "repair_output",
    "schema": {
        "type": "object",
        "properties": {
            "repaired_text": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["repaired_text", "reasoning"],
        "additionalProperties": False,
    },
}

_RESOLVABILITY_VERDICT_JSON_SCHEMA = {
    "name": "resolvability_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no", "idk"]},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
}

_RESOLVABILITY_WITH_REMOVALS_JSON_SCHEMA = {
    "name": "resolvability_with_removals_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no", "idk"]},
            "fact_ids_to_remove": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "fact_ids_to_remove", "reasoning"],
        "additionalProperties": False,
    },
}

_SYNTHESIS_JSON_SCHEMA = {
    "name": "proposal_synthesis",
    "schema": {
        "type": "object",
        "properties": {
            "proposal_text": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["proposal_text", "reasoning"],
        "additionalProperties": False,
    },
}

_SYNTHESIS_SYSTEM_PROMPT = (
    "You write ONE fluent, natural-reading resume bullet point centered on a claim's own NUCLEUS: `why` (the "
    "underlying motivation/theme) and, when present, a separate `result` (a concrete payoff distinct from the "
    "why). Use the cited facts as the nucleus's SUPPORTING EVIDENCE, not as a checklist to flatly enumerate - "
    "the bullet must incorporate every cited fact's own content, but should read as one unified statement with "
    "a clear center of gravity, not a list of everything the facts state. The 'grouping rationale' text given "
    "alongside the nucleus is background context only, for coherence-checking - never quote it verbatim or "
    "treat it as the sentence to rewrite. Stay strictly within what the facts state - do not add any number, "
    "tool, outcome, or scope not already present in them, and do not introduce a second, different "
    "accomplishment.\n\n"
    "When `result` is present and genuinely distinct from `why`, foreground that concrete payoff (the bullet's "
    "center of gravity is IMPACT). When `result` is absent or collapses into `why`, foreground the capability/"
    "principle itself (the bullet's center of gravity is CAPABILITY/ROBUSTNESS) rather than inventing a result "
    "that was never stated.\n\n"
    "Example (why + separate result): why=\"validating changes automatically before they reach users\", "
    "result=\"cut post-release defects by half\", cited facts \"Added an automated regression-test suite that "
    "runs on every pull request.\" and \"Post-release defect reports dropped by roughly 50% after the suite was "
    "introduced.\" -> \"Added an automated regression-test suite that runs on every pull request, cutting "
    "post-release defects by roughly 50%.\"\n"
    "Example (why alone / no separate result): why=\"letting users tailor the product to their own workflow\", "
    "result=(none), cited facts \"Built a settings panel for per-user notification preferences.\" and \"Added "
    "light/dark theme toggling to the same panel.\" -> \"Built a configurable settings panel letting users "
    "tailor notification preferences and visual theme to their own workflow.\"\n\n"
    "Return the bullet as `proposal_text`."
)

_FACT_SUPPORT_SYSTEM_PROMPT = (
    "You check exactly one thing: does this proposal state any specific detail - a number, a named tool, a "
    "measured outcome, or an unstated claim about WHO did something, WHY it happened, or WHEN it happened - that "
    "is not directly supported by its cited facts?\n\n"
    "Example (yes = unsupported detail present): cited fact \"Built a document-indexing service.\" proposal "
    "\"Single-handedly built a document-indexing service that became the company's most-used internal tool.\" -> "
    "yes, both \"single-handedly\" (an unstated ownership claim) and \"most-used internal tool\" (an unstated "
    "outcome) are not in the cited fact.\n"
    "Example (no = fully supported): cited facts \"Built a document-indexing service.\" and \"Reduced average "
    "query latency from 300ms to 90ms.\" proposal \"Built a document-indexing service, reducing average query "
    "latency from 300ms to 90ms.\" -> no, both details are restatements of the cited facts.\n\n"
    "Answer `no` if fully supported, `yes` if it states something unsupported, or `idk` only if you genuinely "
    "cannot tell whether a specific detail is supported or not."
)

_SAME_CLAIM_INTEGRITY_SYSTEM_PROMPT = (
    "You check exactly one thing: does this proposal describe exactly ONE coherent accomplishment, rather than "
    "blending two or more genuinely different, unrelated accomplishments together?\n\n"
    "Example (no = one accomplishment): \"Built a document-indexing service, reducing average query latency "
    "from 300ms to 90ms.\" -> no, both details describe the same service.\n"
    "Example (yes = blended accomplishments): \"Built a document-indexing service and redesigned the company's "
    "employee onboarding checklist.\" -> yes, two unrelated deliverables are stapled into one claim.\n\n"
    "Answer `no` if it is one coherent accomplishment, `yes` if it blends distinct ones, or `idk` only if you "
    "genuinely cannot tell."
)

_SEMANTIC_DUPLICATION_SYSTEM_PROMPT = (
    "You check exactly one thing: does this proposal substantially restate the SAME real accomplishment as one "
    "of the given PROTECTED prior bullets - not merely a similar topic or shared skill, but the same underlying "
    "achievement - even if worded differently?\n\n"
    "Example (yes = same accomplishment restated): protected bullet \"Migrated the billing service's database "
    "to a managed cloud provider.\" proposal \"Moved the billing service's database onto a managed cloud "
    "platform.\" -> yes, this is the same migration restated in different words.\n"
    "Example (no = a genuinely different accomplishment, even if related): protected bullet \"Migrated the "
    "billing service's database to a managed cloud provider.\" proposal \"Reduced the billing service's monthly "
    "query costs by 30% after the migration.\" -> no, a distinct, separately-measurable result, not a restatement "
    "of the migration itself.\n\n"
    "Answer `no` if it is not a restatement of any protected bullet, `yes` if it substantially restates one, or "
    "`idk` only if you genuinely cannot tell."
)

_PROJECT_RELEVANCE_SYSTEM_PROMPT = (
    "You check exactly one thing: is this proposal NOT plausibly relevant to any of the listed target skills - "
    "i.e. would it fail to help demonstrate any of them to a hiring reader?\n\n"
    "Example (no = clearly relevant, so this is a NO to \"not relevant\"): target skills include \"backend "
    "services\"; proposal \"Built a document-indexing service handling 2 million requests per day.\" -> no, "
    "clearly relevant.\n"
    "Example (idk = genuinely unclear): target skills list is empty or the proposal's connection to any listed "
    "skill is only loosely plausible (for example internal process documentation with no listed skill it "
    "directly demonstrates) -> idk, not confidently relevant or irrelevant.\n\n"
    "Answer `no` if it is plausibly relevant (i.e. NOT a relevance problem), `yes` if it is clearly NOT relevant "
    "to any listed skill, or `idk` if you genuinely cannot tell either way."
)

_HALLUCINATION_REPAIR_SYSTEM_PROMPT = (
    "Rewrite this proposal to remove or correct the specific detail(s) not supported by its cited facts, "
    "changing as little else as possible. Do not add any new fact, number, tool, or outcome not already in the "
    "cited facts, and do not change what accomplishment is being described."
)

_BAD_FLOW_REPAIR_SYSTEM_PROMPT = (
    "This proposal currently blends two different accomplishments into one claim. If possible, rewrite it to "
    "focus on only ONE of the accomplishments, using only the facts that support that one, and drop content "
    "belonging to the other. Do not fabricate a connection between them and do not add any fact not already "
    "cited."
)

_BAD_WORDING_REPAIR_SYSTEM_PROMPT = (
    "This proposal currently reads as substantially restating already-established prior work. Rewrite it to "
    "foreground whatever is genuinely NEW in its own cited facts. Entirely REMOVE the portion that restates the "
    "prior work - do not merely reorder, shorten, or de-emphasize it while keeping its same wording or verb "
    "phrase. The rewritten proposal must read as being ABOUT the new content, only incidentally mentioning the "
    "prior work's subject if a connecting word is unavoidable, and must not add any fact not already cited.\n\n"
    "Example: proposal \"Built a document-indexing service, reducing average query latency from 300ms to 90ms.\" "
    "where \"Built a document-indexing service\" restates prior work -> \"Reduced the document-indexing "
    "service's average query latency from 300ms to 90ms.\" (the prior work's own phrasing is gone entirely, not "
    "just reordered)."
)

_RESOLVABLE_BY_EDITING_SYSTEM_PROMPT = (
    "You check exactly one thing: given a proposal that failed verification for a stated reason, can the "
    "failure be fixed by ONLY rewording the proposal - keeping every one of its currently cited facts, without "
    "dropping any of them?\n\n"
    "The rewritten proposal must still pass the SAME check that originally failed. For a `bad_flow` failure "
    "(blends two different accomplishments), this specifically means the rewritten text must read as exactly "
    "ONE coherent accomplishment - splitting the two facts into separate sentences, clauses, or adding "
    "'also'/'additionally' does NOT fix this, since the result still describes two accomplishments, just "
    "written adjacently instead of blended into one sentence. For a `bad_wording` failure (substantially "
    "restates an existing protected bullet's real accomplishment), this is NOT a grammar, causation-implication, "
    "or phrasing-style problem - fixing implied causation between two true facts does not resolve it. It is "
    "fixed only if the rewritten text no longer restates the protected accomplishment at all while still citing "
    "every currently-cited fact.\n\n"
    "Example (yes): failure reason \"states a detail not supported by its cited facts\". Proposal \"Built a "
    "document-indexing service that became the company's most-used internal tool.\" Cited fact \"Built a "
    "document-indexing service.\" -> yes, removing the unsupported \"most-used internal tool\" claim and "
    "keeping only what the fact states fixes it without dropping the fact itself.\n"
    "Example (no): failure reason \"blends two different accomplishments\". Cited facts \"Built a "
    "document-indexing service.\" and \"Redesigned the billing service's monthly invoice email template.\" -> "
    "no, these are two unrelated deliverables; no rewording - including splitting them into two adjacent "
    "sentences - can honestly present both as ONE accomplishment while still citing both facts.\n"
    "Example (no): failure reason \"substantially restates a protected prior bullet's accomplishment\". "
    "Protected bullet \"Migrated the billing service's database to a managed cloud provider.\" Cited facts "
    "\"Migrated the billing service's database to a managed cloud provider.\" and \"Reduced the billing "
    "service's monthly query costs by 30% after the migration.\" -> no, the first cited fact IS the restated "
    "accomplishment itself; no rewording can discuss that fact's own content without restating it, so editing "
    "alone (keeping both facts) cannot fix it.\n\n"
    "Answer `yes` if editing alone (keeping every cited fact) can fix it, `no` if it cannot, or `idk` only if "
    "you genuinely cannot tell."
)

_RESOLVABLE_BY_REMOVING_FACTS_SYSTEM_PROMPT = (
    "You check exactly one thing: given a proposal that failed verification for a stated reason, where editing "
    "alone (keeping every currently cited fact) is NOT sufficient, can dropping ONE OR MORE of the currently "
    "cited facts - then rewording using only the rest - resolve the failure? At least one fact must remain "
    "after dropping.\n\n"
    "For a `bad_flow` failure (blends two different accomplishments), dropping the fact(s) belonging to the "
    "accomplishment NOT being kept, then rewording around only the remaining fact(s), is the standard fix. For "
    "a `bad_wording` failure (substantially restates an existing protected bullet's real accomplishment), "
    "dropping the specific fact whose content IS the restated accomplishment, then rewording around only the "
    "remaining genuinely-new fact(s), is the standard fix.\n\n"
    "Example (yes): failure reason \"blends two different accomplishments\". Cited facts (with IDs) "
    "\"fact_a: Built a document-indexing service.\" and \"fact_b: Redesigned the billing service's monthly "
    "invoice email template.\" -> yes, dropping fact_b and rewording around fact_a alone resolves it into one "
    "coherent accomplishment; fact_ids_to_remove: [\"fact_b\"].\n"
    "Example (no): failure reason \"fully restates already-established prior work\", and the ONLY cited fact is "
    "the restated content itself -> no, dropping the sole fact would leave zero supporting facts, and there is "
    "no other fact to fall back on.\n\n"
    "If yes, name exactly which currently-cited fact ID(s) must be dropped in `fact_ids_to_remove`. Answer `no` "
    "if no combination of dropped facts (leaving at least one) can fix it, or `idk` only if you genuinely cannot "
    "tell."
)

_FAILURE_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "hallucination": "the proposal states a specific detail (a number, tool, or outcome) not supported by any "
    "of its cited facts",
    "bad_flow": "the proposal blends two or more genuinely different, unrelated accomplishments into one claim "
    "- to pass, it must describe exactly ONE coherent accomplishment, not the same facts split into separate "
    "sentences or clauses",
    "bad_wording": "the proposal substantially restates the SAME real accomplishment as an existing protected "
    "prior bullet, even though it may cite a different fact id - this is a duplication problem, NOT a grammar, "
    "causation-implication, or phrasing-style problem",
}


def _format_fact_list(fact_texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in fact_texts) or "(none)"



def synthesize_proposal(
    core_claim: CoreClaimMolecule,
    expansion: Optional[ExpandedClaimMolecule],
    fact_atoms_by_id: Dict[str, FactAtom],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = VERIFICATION_REASONING_EFFORT,
) -> AnnotatedProposal:
    """Turn a core claim plus its (optional) expansion decision into ONE
    fluent `AnnotatedProposal`, via a single bounded LLM call.

    Phase 3.8: the bullet is built around `core_claim`'s why/result
    NUCLEUS (facts become supporting evidence for it, not a checklist).
    `claim_text` is passed only as background grouping rationale, never
    as the literal sentence to rewrite - `claim_text` is a Phase 3
    grouping artifact, not itself the bullet's source text.
    """

    added_ids = expansion.added_support_fact_ids if expansion is not None else ()
    supporting_fact_ids = tuple(dict.fromkeys((*core_claim.supporting_fact_ids, *added_ids)))
    fact_texts = [fact_atoms_by_id[fact_id].fact for fact_id in supporting_fact_ids if fact_id in fact_atoms_by_id]

    result_line = f'Nucleus - result: "{core_claim.result}"' if core_claim.result else (
        "Nucleus - result: (none - why and result collapse into the same idea; do not invent a separate result)"
    )
    prompt = (
        f'Nucleus - why: "{core_claim.why}"\n'
        f"{result_line}\n"
        f'(Grouping rationale, background context only, not to be quoted verbatim: "{core_claim.claim_text}")\n\n'
        f"Cited facts (supporting evidence for the nucleus above):\n{_format_fact_list(fact_texts)}\n\n"
        "Write one fluent resume bullet point centered on the nucleus above, incorporating all cited facts "
        "as its supporting evidence."
    )
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
        json_schema=_SYNTHESIS_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )

    return AnnotatedProposal(
        id=f"{core_claim.id}_proposal",
        project_id=core_claim.project_id,
        core_claim_id=core_claim.id,
        proposal_text=response.get("proposal_text", core_claim.claim_text),
        supporting_fact_ids=supporting_fact_ids,
        target_skills=core_claim.target_skills,
    )


def _classify(reasoning_llm: LLMProvider, system_prompt: str, prompt: str, reasoning_effort: Optional[str]) -> Dict[str, Any]:
    response = reasoning_llm.call_json(
        prompt=prompt,
        system_prompt=system_prompt,
        json_schema=_VERDICT_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    verdict = response.get("verdict")
    if verdict not in ("yes", "no", "idk"):
        verdict = "idk"
    return {"verdict": verdict, "reasoning": response.get("reasoning", "")}


def _classify_resolvable_by_editing(
    proposal_text: str,
    failure_type: RepairType,
    cited_fact_texts: Sequence[str],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str],
    protected_baseline_bullet_texts: Sequence[str] = (),
) -> Dict[str, Any]:
    prompt = (
        f'Proposal: "{proposal_text}"\n\n'
        f"Verification failure type: {failure_type} - "
        f"{_FAILURE_TYPE_DESCRIPTIONS.get(failure_type, '(no description available)')}\n\n"
        f"Currently cited facts:\n{_format_fact_list(cited_fact_texts)}\n\n"
    )
    if protected_baseline_bullet_texts:
        prompt += (
            f"Existing protected prior bullets (for `bad_wording`, these are what the proposal may be "
            f"restating):\n{_format_fact_list(protected_baseline_bullet_texts)}\n\n"
        )
    prompt += "Can this failure be fixed by editing alone, keeping every one of these cited facts?"
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_RESOLVABLE_BY_EDITING_SYSTEM_PROMPT,
        json_schema=_RESOLVABILITY_VERDICT_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    verdict = response.get("verdict")
    if verdict not in ("yes", "no", "idk"):
        verdict = "idk"
    return {"verdict": verdict, "reasoning": response.get("reasoning", "")}


def _classify_resolvable_by_removing_facts(
    proposal_text: str,
    failure_type: RepairType,
    cited_facts: Sequence[Tuple[str, str]],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str],
    protected_baseline_bullet_texts: Sequence[str] = (),
) -> Dict[str, Any]:
    facts_listing = "\n".join(f"- {fact_id}: {text}" for fact_id, text in cited_facts) or "(none)"
    prompt = (
        f'Proposal: "{proposal_text}"\n\n'
        f"Verification failure type: {failure_type} - "
        f"{_FAILURE_TYPE_DESCRIPTIONS.get(failure_type, '(no description available)')}\n\n"
        f"Currently cited facts (with IDs):\n{facts_listing}\n\n"
    )
    if protected_baseline_bullet_texts:
        prompt += (
            f"Existing protected prior bullets (for `bad_wording`, identify which cited fact's content IS one "
            f"of these, and drop that one):\n{_format_fact_list(protected_baseline_bullet_texts)}\n\n"
        )
    prompt += (
        "Editing alone (keeping every cited fact) is not sufficient. Can dropping one or more of these facts - "
        "then rewording using only the rest - resolve the failure? At least one fact must remain."
    )
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=_RESOLVABLE_BY_REMOVING_FACTS_SYSTEM_PROMPT,
        json_schema=_RESOLVABILITY_WITH_REMOVALS_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    verdict = response.get("verdict")
    if verdict not in ("yes", "no", "idk"):
        verdict = "idk"
    fact_ids_to_remove = response.get("fact_ids_to_remove")
    if not isinstance(fact_ids_to_remove, list):
        fact_ids_to_remove = []
    return {
        "verdict": verdict,
        "reasoning": response.get("reasoning", ""),
        "fact_ids_to_remove": [str(fact_id) for fact_id in fact_ids_to_remove],
    }


def verify_proposal(
    proposal: AnnotatedProposal,
    fact_atoms_by_id: Dict[str, FactAtom],
    protected_fact_ids: Set[str],
    protected_baseline_bullets: Sequence[BaselineBullet],
    target_skills: Sequence[str],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = VERIFICATION_REASONING_EFFORT,
) -> VerificationResult:
    """Verify one `AnnotatedProposal`. Never infers a target replacement
    slot - this only judges the proposal's own grounding/coherence/
    relevance, independent of any particular baseline bullet it might
    eventually compete with.
    """

    reused_protected_ids = sorted(set(proposal.supporting_fact_ids) & protected_fact_ids)
    if reused_protected_ids:
        return VerificationResult(
            proposal_id=proposal.id,
            project_id=proposal.project_id,
            status="fail",
            failure_type="unresolvable",
        )

    cited_fact_texts = [
        fact_atoms_by_id[fact_id].fact for fact_id in proposal.supporting_fact_ids if fact_id in fact_atoms_by_id
    ]
    protected_bullet_texts = [bullet.text for bullet in protected_baseline_bullets]
    skills_text = ", ".join(target_skills) or "(none listed)"

    fact_support = _classify(
        llm_provider,
        _FACT_SUPPORT_SYSTEM_PROMPT,
        f'Proposal: "{proposal.proposal_text}"\n\nCited facts:\n{_format_fact_list(cited_fact_texts)}\n\n'
        "Does this proposal state anything not supported by its cited facts?",
        reasoning_effort,
    )
    if fact_support["verdict"] == "yes":
        return VerificationResult(
            proposal_id=proposal.id, project_id=proposal.project_id, status="fail", failure_type="hallucination"
        )

    integrity = _classify(
        llm_provider,
        _SAME_CLAIM_INTEGRITY_SYSTEM_PROMPT,
        f'Proposal: "{proposal.proposal_text}"\n\nDoes this describe exactly one coherent accomplishment?',
        reasoning_effort,
    )
    if integrity["verdict"] == "yes":
        return VerificationResult(
            proposal_id=proposal.id, project_id=proposal.project_id, status="fail", failure_type="bad_flow"
        )

    duplication = _classify(
        llm_provider,
        _SEMANTIC_DUPLICATION_SYSTEM_PROMPT,
        f'Proposal: "{proposal.proposal_text}"\n\nProtected prior bullets:\n'
        f'{_format_fact_list(protected_bullet_texts)}\n\n'
        "Does this proposal substantially restate any protected prior bullet's accomplishment?",
        reasoning_effort,
    )
    if duplication["verdict"] == "yes":
        return VerificationResult(
            proposal_id=proposal.id, project_id=proposal.project_id, status="fail", failure_type="bad_wording"
        )

    relevance = _classify(
        llm_provider,
        _PROJECT_RELEVANCE_SYSTEM_PROMPT,
        f'Proposal: "{proposal.proposal_text}"\n\nTarget skills: {skills_text}\n\n'
        "Is this proposal NOT plausibly relevant to any of the listed target skills?",
        reasoning_effort,
    )
    if relevance["verdict"] == "yes":
        return VerificationResult(
            proposal_id=proposal.id, project_id=proposal.project_id, status="fail", failure_type="bad_wording"
        )

    if "idk" in (fact_support["verdict"], integrity["verdict"], duplication["verdict"], relevance["verdict"]):
        return VerificationResult(proposal_id=proposal.id, project_id=proposal.project_id, status="idk")

    return VerificationResult(
        proposal_id=proposal.id, project_id=proposal.project_id, status="pass", final_text=proposal.proposal_text
    )


def _repair_text(
    proposal: AnnotatedProposal,
    failure_type: RepairType,
    resolution: RepairResolution,
    fact_atoms_by_id: Dict[str, FactAtom],
    protected_baseline_bullets: Sequence[BaselineBullet],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str],
) -> str:
    system_prompt = {
        "hallucination": _HALLUCINATION_REPAIR_SYSTEM_PROMPT,
        "bad_flow": _BAD_FLOW_REPAIR_SYSTEM_PROMPT,
        "bad_wording": _BAD_WORDING_REPAIR_SYSTEM_PROMPT,
    }[failure_type]
    cited_fact_texts = [
        fact_atoms_by_id[fact_id].fact for fact_id in proposal.supporting_fact_ids if fact_id in fact_atoms_by_id
    ]
    prompt = (
        f'Current proposal: "{proposal.proposal_text}"\n\n'
        f"Its cited facts (rewrite to use ONLY these):\n{_format_fact_list(cited_fact_texts)}\n\n"
    )
    if resolution == "remove_facts":
        prompt += (
            "One or more facts previously cited by this proposal have been removed from the list above. The "
            "rewritten proposal must not reference any content tied only to a removed fact - remove that "
            "content entirely, do not just reword around it.\n\n"
        )
    if failure_type == "bad_wording":
        protected_bullet_texts = [bullet.text for bullet in protected_baseline_bullets]
        prompt += (
            f"Prior work it currently restates (remove this content entirely, do not just reorder it):\n"
            f"{_format_fact_list(protected_bullet_texts)}\n\n"
        )
    prompt += "Rewrite the proposal per the instructions above."
    response = llm_provider.call_json(
        prompt=prompt,
        system_prompt=system_prompt,
        json_schema=_REPAIR_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
    )
    return response.get("repaired_text", proposal.proposal_text)


def repair_proposal(
    proposal: AnnotatedProposal,
    verification: VerificationResult,
    fact_atoms_by_id: Dict[str, FactAtom],
    protected_fact_ids: Set[str],
    protected_baseline_bullets: Sequence[BaselineBullet],
    target_skills: Sequence[str],
    llm_provider: LLMProvider,
    reasoning_effort: Optional[str] = VERIFICATION_REASONING_EFFORT,
) -> Tuple[AnnotatedProposal, VerificationResult]:
    """Attempt bounded, typed repairs following the fixed sequence
    `hallucination` -> `bad_flow` -> `bad_wording`, one attempt per
    distinct failure type ever encountered, reverifying after each.
    Discards (returns the final `fail` result unchanged) on a repair that
    doesn't resolve its own target failure, or immediately on
    `unresolvable`/`idk`/`pass` (nothing to repair). Repair never
    retrieves facts, changes project context, or replaces the core
    molecule - each repair call only rewords the EXISTING proposal text
    using only its OWN already-cited facts.

    Phase 5.1: before any rewrite, a 2-stage resolvability gate decides
    HOW to attempt the fix - `edit_only` (keep every currently-cited
    fact) or, only if that is not viable, `remove_facts` (drop specific
    currently-cited fact(s), naming which, then reword using only the
    rest). If neither is viable, the failure becomes `unresolvable`
    immediately: no rewrite prompt is called, and the whole repair loop
    stops (a fundamentally unresolvable failure isn't worth continuing
    past, regardless of what other failure types might also apply). A
    `remove_facts` resolution deterministically prunes the repaired
    proposal's own `supporting_fact_ids`, so lineage never goes stale.
    """

    current_proposal = proposal
    current_verification = verification
    repair_steps: List[RepairStep] = []
    attempted_types: Set[str] = set()

    while (
        current_verification.status == "fail"
        and current_verification.failure_type in _REPAIRABLE_TYPES
        and current_verification.failure_type not in attempted_types
    ):
        failure_type = current_verification.failure_type
        attempted_types.add(failure_type)

        before_text = current_proposal.proposal_text
        cited_fact_texts = [
            fact_atoms_by_id[fact_id].fact
            for fact_id in current_proposal.supporting_fact_ids
            if fact_id in fact_atoms_by_id
        ]

        protected_bullet_texts = [bullet.text for bullet in protected_baseline_bullets]

        editing_gate = _classify_resolvable_by_editing(
            before_text,
            failure_type,
            cited_fact_texts,
            llm_provider,
            reasoning_effort,
            protected_baseline_bullet_texts=protected_bullet_texts,
        )

        resolution: Optional[RepairResolution] = None
        removed_fact_ids: Tuple[str, ...] = ()
        proposal_for_rewrite = current_proposal

        if editing_gate["verdict"] == "yes":
            resolution = "edit_only"
        else:
            cited_facts_with_ids = [
                (fact_id, fact_atoms_by_id[fact_id].fact)
                for fact_id in current_proposal.supporting_fact_ids
                if fact_id in fact_atoms_by_id
            ]
            removing_gate = _classify_resolvable_by_removing_facts(
                before_text,
                failure_type,
                cited_facts_with_ids,
                llm_provider,
                reasoning_effort,
                protected_baseline_bullet_texts=protected_bullet_texts,
            )
            if removing_gate["verdict"] == "yes":
                candidate_removals = tuple(
                    fact_id
                    for fact_id in removing_gate["fact_ids_to_remove"]
                    if fact_id in current_proposal.supporting_fact_ids
                )
                remaining_fact_ids = tuple(
                    fact_id for fact_id in current_proposal.supporting_fact_ids if fact_id not in candidate_removals
                )
                if candidate_removals and remaining_fact_ids:
                    resolution = "remove_facts"
                    removed_fact_ids = candidate_removals
                    proposal_for_rewrite = replace(current_proposal, supporting_fact_ids=remaining_fact_ids)

        if resolution is None:
            # Neither editing alone nor removing facts is viable - genuinely
            # unresolvable. No rewrite is ever attempted, and no further
            # repair attempts of any kind are made for this proposal.
            repair_steps.append(
                RepairStep(
                    repair_type=failure_type,
                    before_text=before_text,
                    after_text=None,
                    reverified_status=None,
                    resolution=None,
                    removed_fact_ids=(),
                )
            )
            current_verification = replace(current_verification, failure_type="unresolvable")
            break

        after_text = _repair_text(
            proposal_for_rewrite,
            failure_type,
            resolution,
            fact_atoms_by_id,
            protected_baseline_bullets,
            llm_provider,
            reasoning_effort,
        )
        repaired_proposal = replace(proposal_for_rewrite, proposal_text=after_text)
        new_verification = verify_proposal(
            repaired_proposal,
            fact_atoms_by_id,
            protected_fact_ids,
            protected_baseline_bullets,
            target_skills,
            llm_provider,
            reasoning_effort,
        )
        repair_steps.append(
            RepairStep(
                repair_type=failure_type,
                before_text=before_text,
                after_text=after_text,
                reverified_status=new_verification.status,
                resolution=resolution,
                removed_fact_ids=removed_fact_ids,
            )
        )
        current_proposal = repaired_proposal
        current_verification = new_verification

    final_verification = replace(
        current_verification,
        repair_steps=tuple(repair_steps),
        final_text=current_proposal.proposal_text if current_verification.status == "pass" else None,
    )
    return current_proposal, final_verification


def annotated_proposals_to_dicts(proposals: Sequence[AnnotatedProposal]) -> List[dict]:
    return [
        {
            "id": proposal.id,
            "project_id": proposal.project_id,
            "core_claim_id": proposal.core_claim_id,
            "proposal_text": proposal.proposal_text,
            "supporting_fact_ids": list(proposal.supporting_fact_ids),
            "target_skills": list(proposal.target_skills),
        }
        for proposal in proposals
    ]


def write_annotated_proposal_set_json(proposals: Sequence[AnnotatedProposal], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(annotated_proposals_to_dicts(proposals), handle, indent=2)


def verification_results_to_dicts(results: Sequence[VerificationResult]) -> List[dict]:
    return [
        {
            "proposal_id": result.proposal_id,
            "project_id": result.project_id,
            "status": result.status,
            "failure_type": result.failure_type,
            "final_text": result.final_text,
            "repair_steps": [
                {
                    "repair_type": step.repair_type,
                    "before_text": step.before_text,
                    "after_text": step.after_text,
                    "reverified_status": step.reverified_status,
                    "resolution": step.resolution,
                    "removed_fact_ids": list(step.removed_fact_ids),
                }
                for step in result.repair_steps
            ],
        }
        for result in results
    ]


def write_verification_report_json(results: Sequence[VerificationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(verification_results_to_dicts(results), handle, indent=2)
