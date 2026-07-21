"""Phase 4: bounded support expansion and verbosity prefilter.

Strengthens ONE selected Phase 3 core claim with a handful of additional,
currently-unused LOCAL facts from the same project, without turning it into
a second accomplishment - never a per-slot or per-job-posting judgment.
`ExpandedClaimMolecule` (defined in `tailoring.models` since Phase 0) is a
DECISION/lineage record only: which facts were added, which were excluded
and why, and why expansion stopped. It deliberately has no text field -
this phase does not author or rewrite the final expanded bullet's wording;
that synthesis is deferred to a later phase (see `VerificationResult.final_text`
in `tailoring.models`).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm import DEFAULT_REASONING_EFFORT, LLMProvider
from matcher import EmbeddingCache, SemanticMatcher, SkillRecord

from tailoring.models import CoreClaimMolecule, ExpandedClaimMolecule, FactAtom
from tailoring.retrieval import DEFAULT_EMBEDDING_CACHE_PATH

MAX_SUPPORT_POOL_SIZE = 4
MAX_SUPPORT_ADDITIONS = 3

# This ranking step deliberately applies NO similarity threshold/gate - per
# AGENTS.md's "LLM Scoring Rubric Design" (a threshold calibrated for one
# task must be re-validated before reuse on a different one), and there is
# no calibrated threshold for THIS claim-to-fact relevance task. Setting it
# to -1.0 (below any real cosine similarity) keeps every candidate ranked,
# never gated - `expand_claim_molecule`'s per-fact marginal-value classifier
# is what actually decides inclusion.
_NO_GATING_SIMILARITY_THRESHOLD = -1.0

# Derived from data/template.tex's real geometry: \documentclass[10pt]{article}
# with \geometry[...]{left=.5in,right=.5in} on a US Letter (8.5in wide) page
# gives a 7.5in text width; the itemize list's own left indent
# (leftmargin=*) takes roughly another ~0.3in, leaving ~7.2in of usable
# bullet width. A commonly used typography estimate for 10pt mixed-case
# English prose is ~12 characters per horizontal inch. 7.2 * 12 ~= 86,
# rounded DOWN to bias this prefilter toward flagging borderline cases
# rather than missing them (it is explicitly a rough, NON-AUTHORITATIVE
# estimate per the dev plan - real page-fit validation against the
# compiled PDF is a separate, deferred phase; this prefilter exists only to
# avoid spending verification calls on obviously overlong wording).
DEFAULT_CHARS_PER_LINE = 85
DEFAULT_MAX_BULLET_LINES = 2

# Rough, conservative per-added-fact character allowance. Phase 4 does not
# author the final expanded bullet text (see module docstring), so there is
# no real added text to measure yet - this estimates that naturally folding
# one more supporting detail into a bullet (e.g. ", achieving X" or "using
# Y") typically costs about this many extra characters.
ESTIMATED_CHARS_PER_ADDED_FACT = 45

_MARGINAL_VALUE_JSON_SCHEMA = {
    "name": "support_expansion_decision",
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["add_support", "keep_out", "stop"]},
            "reasoning": {"type": "string"},
        },
        "required": ["decision", "reasoning"],
        "additionalProperties": False,
    },
}

_MARGINAL_VALUE_SYSTEM_PROMPT = (
    "You decide whether ONE candidate fact should be added as extra supporting evidence for an existing resume "
    "claim, without turning the claim into a second, different accomplishment. Candidate facts are given to you "
    "in descending order of estimated relevance to the claim, so once a candidate is clearly weak the remaining "
    "candidates (all found LESS relevant than this one) are unlikely to be worth considering either.\n\n"
    "Decide exactly one of:\n"
    "- add_support: this fact strengthens the SAME accomplishment already described by the claim with a "
    "genuinely relevant, verifiable detail (a supporting metric, the tool/mechanism that enabled the result, or "
    "a scale/scope detail) - it should be added.\n"
    "- keep_out: this fact does not clearly strengthen the SAME accomplishment (it is unrelated, it describes a "
    "different accomplishment, or it is only superficially related) - skip it, but other, more-relevant "
    "remaining candidates may still be worth considering.\n"
    "- stop: this fact - and, since it is already less relevant than every fact considered so far, every "
    "remaining candidate too - would not strengthen this claim. Halt expansion entirely rather than continuing "
    "to check weaker candidates.\n\n"
    "Never choose add_support merely because a fact is generally impressive or from the same project - it must "
    "support this SPECIFIC accomplishment, not describe a separate one. A fact that only MENTIONS a related "
    "team/technology in passing (for example, a backend deliverable that is merely consumed by a frontend team) "
    "is judged by what it itself accomplished, not by who benefits from it.\n\n"
    "Example (add_support): claim \"Built a REST API for order processing using FastAPI.\" candidate fact "
    "\"Added pagination to the order-processing API to handle large result sets.\" -> add_support, a concrete "
    "detail strengthening the same API deliverable.\n"
    "Example (keep_out): the same claim; candidate fact \"Built a React dashboard for viewing orders.\" -> "
    "keep_out, this describes a separate frontend accomplishment, not the backend API itself."
)


def _format_fact_list(fact_texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in fact_texts) or "(none)"


def _build_marginal_value_prompt(
    claim: CoreClaimMolecule,
    core_fact_texts: Sequence[str],
    added_so_far_texts: Sequence[str],
    candidate_fact_text: str,
) -> str:
    return (
        f'Existing claim: "{claim.claim_text}"\n\n'
        f"Facts already cited by this claim:\n{_format_fact_list(core_fact_texts)}\n\n"
        f"Facts already added as extra support this round:\n{_format_fact_list(added_so_far_texts)}\n\n"
        f'Candidate fact to evaluate: "{candidate_fact_text}"\n\n'
        "Should this candidate fact be added as extra support for the SAME accomplishment?"
    )


def build_support_pool(
    claim: CoreClaimMolecule,
    fact_atoms: Sequence[FactAtom],
    llm_provider: Optional[LLMProvider] = None,
    embedding_cache_path: Optional[Path] = DEFAULT_EMBEDDING_CACHE_PATH,
    max_pool_size: int = MAX_SUPPORT_POOL_SIZE,
) -> List[FactAtom]:
    """Rank this project's UNUSED facts (not already cited by `claim`) by
    semantic similarity to the claim's own narrative (`claim.claim_text`),
    then cap at `max_pool_size`.

    This step only RANKS candidates - see `_NO_GATING_SIMILARITY_THRESHOLD`
    above for why no threshold is applied here. Reuses `SemanticMatcher`
    (each unused fact atom becomes a one-off `SkillRecord` keyed by its own
    fact text, not its skill_tags - the point is narrative-to-narrative
    relevance to THIS claim, not target-skill overlap, which Phase 2's
    retrieval already covers) rather than duplicating cosine-similarity
    logic. Falls back to input order (first `max_pool_size` unused facts)
    when no `llm_provider` is given or it doesn't support embeddings.
    """

    unused = [atom for atom in fact_atoms if atom.id not in claim.supporting_fact_ids]
    if not unused or llm_provider is None:
        return unused[:max_pool_size]

    records = [SkillRecord(name=atom.id, aliases=(atom.fact,)) for atom in unused]
    embedding_cache = EmbeddingCache(embedding_cache_path) if embedding_cache_path is not None else None
    try:
        matcher = SemanticMatcher(
            records,
            llm_provider,
            similarity_threshold=_NO_GATING_SIMILARITY_THRESHOLD,
            embedding_cache=embedding_cache,
        )
        matches = matcher.match(claim.claim_text)
    except NotImplementedError:
        # Provider doesn't support embeddings (e.g. Anthropic, Ollama today).
        return unused[:max_pool_size]

    atoms_by_id = {atom.id: atom for atom in unused}
    ranked = [atoms_by_id[match.canonical_name] for match in matches if match.canonical_name in atoms_by_id]
    return ranked[:max_pool_size]


def expand_claim_molecule(
    claim: CoreClaimMolecule,
    support_pool: Sequence[FactAtom],
    fact_atoms_by_id: Dict[str, FactAtom],
    llm_provider: LLMProvider,
    max_additions: int = MAX_SUPPORT_ADDITIONS,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
) -> ExpandedClaimMolecule:
    """Decide, one narrow LLM call per candidate fact (in `support_pool`'s
    given, already-ranked order), whether each should be added as extra
    support for `claim`. Every decision and its reasoning is recorded -
    added facts in `added_support_fact_ids`, everything else in
    `excluded_fact_ids`/`exclusion_reasons` (parallel, positional), per
    this project's "never silently drop" convention.
    """

    if not support_pool:
        return ExpandedClaimMolecule(
            core_claim_id=claim.id,
            project_id=claim.project_id,
            stop_reason="empty_support_pool",
        )

    core_fact_texts = [
        fact_atoms_by_id[fact_id].fact for fact_id in claim.supporting_fact_ids if fact_id in fact_atoms_by_id
    ]

    added_ids: List[str] = []
    excluded_ids: List[str] = []
    exclusion_reasons: List[str] = []
    stop_reason = ""

    for atom in support_pool:
        if len(added_ids) >= max_additions:
            stop_reason = "max_additions_reached"
            break

        added_so_far_texts = [fact_atoms_by_id[fact_id].fact for fact_id in added_ids if fact_id in fact_atoms_by_id]
        prompt = _build_marginal_value_prompt(claim, core_fact_texts, added_so_far_texts, atom.fact)
        response = llm_provider.call_json(
            prompt=prompt,
            system_prompt=_MARGINAL_VALUE_SYSTEM_PROMPT,
            json_schema=_MARGINAL_VALUE_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
        )
        decision = response.get("decision")
        reasoning = response.get("reasoning", "")

        if decision == "add_support":
            added_ids.append(atom.id)
        elif decision == "stop":
            stop_reason = "model_stop"
            break
        elif decision == "keep_out":
            excluded_ids.append(atom.id)
            exclusion_reasons.append(reasoning or "keep_out")
        else:
            # Never silently drop an unrecognized decision - treat it as
            # excluded but flag it distinctly from a normal keep_out.
            excluded_ids.append(atom.id)
            exclusion_reasons.append(f"unrecognized_decision:{decision!r}")

    if not stop_reason:
        stop_reason = "pool_exhausted"

    return ExpandedClaimMolecule(
        core_claim_id=claim.id,
        project_id=claim.project_id,
        added_support_fact_ids=tuple(added_ids),
        excluded_fact_ids=tuple(excluded_ids),
        exclusion_reasons=tuple(exclusion_reasons),
        stop_reason=stop_reason,
    )


def estimate_expanded_line_count(
    claim: CoreClaimMolecule,
    added_fact_count: int,
    chars_per_line: int = DEFAULT_CHARS_PER_LINE,
) -> int:
    """Rough, deterministic, NON-AUTHORITATIVE estimate of how many
    rendered lines the expanded bullet would occupy - see
    `DEFAULT_CHARS_PER_LINE`'s derivation above. Exists only to avoid
    spending verification calls on obviously overlong wording; it is not a
    substitute for validating against the real compiled PDF (deferred).
    """

    estimated_chars = len(claim.claim_text) + added_fact_count * ESTIMATED_CHARS_PER_ADDED_FACT
    return max(1, math.ceil(estimated_chars / chars_per_line))


def apply_verbosity_prefilter(
    claim: CoreClaimMolecule,
    expansion: ExpandedClaimMolecule,
    max_lines: int = DEFAULT_MAX_BULLET_LINES,
    chars_per_line: int = DEFAULT_CHARS_PER_LINE,
) -> ExpandedClaimMolecule:
    """Conservative, deterministic, template-specific line-estimate
    prefilter (dev plan Phase 4, task 6).

    If the estimated rendered line count exceeds `max_lines`, removes the
    LOWEST-VALUE added support fact first - the LAST element of
    `added_support_fact_ids`, since `build_support_pool` ranks candidates
    by relevance descending and `expand_claim_molecule` accepts them in
    that order, so additions are already in descending-relevance order -
    and re-estimates, repeating until within budget or no additions
    remain. Never silently replaces the core claim: if the estimate is
    STILL over budget with zero additions, this is recorded via
    `stop_reason` (advisory), not by discarding or rewriting
    `claim.claim_text` - rewriting the core claim's own wording is out of
    this phase's bounded scope.
    """

    added_ids = list(expansion.added_support_fact_ids)
    excluded_ids = list(expansion.excluded_fact_ids)
    exclusion_reasons = list(expansion.exclusion_reasons)
    stop_reason = expansion.stop_reason

    while added_ids and estimate_expanded_line_count(claim, len(added_ids), chars_per_line) > max_lines:
        removed_id = added_ids.pop()
        excluded_ids.append(removed_id)
        exclusion_reasons.append("verbosity_prefilter_removed_lowest_value")

    if estimate_expanded_line_count(claim, len(added_ids), chars_per_line) > max_lines:
        stop_reason = (
            f"{stop_reason};core_claim_exceeds_line_budget" if stop_reason else "core_claim_exceeds_line_budget"
        )

    return replace(
        expansion,
        added_support_fact_ids=tuple(added_ids),
        excluded_fact_ids=tuple(excluded_ids),
        exclusion_reasons=tuple(exclusion_reasons),
        stop_reason=stop_reason,
    )


def expanded_claim_molecules_to_dicts(expansions: Sequence[ExpandedClaimMolecule]) -> List[dict]:
    return [
        {
            "core_claim_id": expansion.core_claim_id,
            "project_id": expansion.project_id,
            "added_support_fact_ids": list(expansion.added_support_fact_ids),
            "excluded_fact_ids": list(expansion.excluded_fact_ids),
            "exclusion_reasons": list(expansion.exclusion_reasons),
            "stop_reason": expansion.stop_reason,
        }
        for expansion in expansions
    ]


def write_expanded_claim_molecules_json(expansions: Sequence[ExpandedClaimMolecule], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(expanded_claim_molecules_to_dicts(expansions), handle, indent=2)
