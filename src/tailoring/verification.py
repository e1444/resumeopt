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
   verification's own status is genuinely 3-way (`pass`/`idk`/`fail`) per
   the dev plan, and `idk` must stay visible, never coerced into
   acceptance or rejection. Processing stops at the first `no` found (a
   hard failure short-circuits remaining checks); an `idk` does NOT
   short-circuit, since a later classifier's `no` should still win.
3. `repair_proposal` - one bounded repair attempt per DISTINCT failure
   type actually encountered (never retrying the same type twice),
   reverifying after each attempt via `verify_proposal` itself, so the
   natural classifier order above already enforces the dev plan's fixed
   repair sequence (`hallucination` -> `bad_flow` -> `bad_wording`).
   Discards (stays `fail`) on a repair that doesn't resolve its own
   target failure, or immediately on `unresolvable` (protected-fact reuse
   can never be repaired - repair may not retrieve facts or change
   project context, so there is nothing a rewording could fix).

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
    "You rewrite a resume claim as ONE fluent, natural-reading bullet point that incorporates ALL of the given "
    "cited facts. Stay strictly within what the facts state - do not add any number, tool, outcome, or scope not "
    "already present in them, and do not introduce a second, different accomplishment. Prefer weaving the "
    "supporting details in naturally (for example as a trailing clause or combined phrase) rather than simply "
    "appending them as an afterthought.\n\n"
    "Example: cited facts \"Built a document-indexing service.\" and \"Reduced average query latency from 300ms "
    "to 90ms.\" -> \"Built a document-indexing service, reducing average query latency from 300ms to 90ms.\"\n\n"
    "Return the rewritten bullet as `proposal_text`."
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
    "You check exactly one thing: is this proposal plausibly relevant to at least one of the listed target "
    "skills - would it help demonstrate at least one of them to a hiring reader?\n\n"
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
    "foreground whatever is genuinely NEW in its own cited facts and remove or de-emphasize the part that "
    "duplicates the prior work, without adding any fact not already cited."
)


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
    fluent `AnnotatedProposal`, via a single bounded LLM call."""

    added_ids = expansion.added_support_fact_ids if expansion is not None else ()
    supporting_fact_ids = tuple(dict.fromkeys((*core_claim.supporting_fact_ids, *added_ids)))
    fact_texts = [fact_atoms_by_id[fact_id].fact for fact_id in supporting_fact_ids if fact_id in fact_atoms_by_id]

    prompt = (
        f'Existing claim: "{core_claim.claim_text}"\n\n'
        f"Cited facts to incorporate:\n{_format_fact_list(fact_texts)}\n\n"
        "Rewrite this as one fluent bullet incorporating all of the cited facts."
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
        "Is this proposal plausibly relevant to at least one target skill?",
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
    fact_atoms_by_id: Dict[str, FactAtom],
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
        f"Its cited facts:\n{_format_fact_list(cited_fact_texts)}\n\n"
        "Rewrite the proposal per the instructions above."
    )
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
        after_text = _repair_text(current_proposal, failure_type, fact_atoms_by_id, llm_provider, reasoning_effort)
        repaired_proposal = replace(current_proposal, proposal_text=after_text)
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
