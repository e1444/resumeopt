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

Phase 3.6: `expand_claim_molecule`'s per-candidate decision is 2 narrow,
single-purpose classifiers plus a deterministic judge rule (add only if
BOTH pass), called in short-circuit sequence - NOT one monolithic prompt
bundling both judgments into a single 3-way decision. This replaced an
earlier single-prompt design after live data showed the bundled prompt's
own decision was unstable (2/3 vs 1/3 across identical repeated runs on
the same input) while a dedicated single-purpose classifier asking only
one of the two questions was perfectly consistent (12/12) on the same
underlying judgment - see the dev plan's Phase 3.5/3.6 for the full
finding.

Phase 3.7: the second classifier is framed around MERGEABILITY, not
literal-wording preservation - a claim describing one system/method may
legitimately BROADEN to cover an additional measured dimension of that
SAME underlying deliverable (it does not have to stay the exact original
claim to remain ONE claim), while a fact about a genuinely different
deliverable, or one that would only staple together an incoherent
narrative, is still rejected. Per this subphase's hygiene rule, NEITHER
classifier's anchor examples may be copied or closely paraphrased from
this repository's fixtures or benchmark scripts - both were rewritten
away from an earlier version that violated this.
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

_VERDICT_JSON_SCHEMA = {
    "name": "single_purpose_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
}

_SAME_UNDERLYING_DELIVERABLE_SYSTEM_PROMPT = (
    "You check exactly one thing: is a candidate fact a result or capability OF THE SAME underlying system, "
    "method, or deliverable that an existing claim is about - not a different deliverable entirely, and not "
    "merely a tool or process used to build or manage it (which is not itself a result of it)?\n\n"
    "Rules:\n"
    "- A fact reporting a DIFFERENT measured dimension or capability of the exact same system still counts as "
    "the SAME underlying deliverable - one system can have more than one measured strength.\n"
    "- A fact describing a genuinely different deliverable (a different feature, component, or product) is NOT "
    "the same underlying deliverable, even if it came from the same broader project.\n"
    "- A fact describing a tool, library, or process used to build or manage the system - rather than a "
    "capability or outcome of the system itself - does not count as a result of that deliverable.\n\n"
    "Example (true - different dimension, same deliverable): claim about a document-search service's query "
    "latency; candidate fact reports the SAME service's search-relevance accuracy on a benchmark -> true, both "
    "are measured capabilities of the same service.\n"
    "Example (false - different deliverable): claim about a document-search service; candidate fact describes a "
    "separate internal admin dashboard built for the support team -> false, a different deliverable.\n"
    "Example (false - tool/process, not a result): claim about a document-search service's performance; "
    "candidate fact names the continuous-deployment tool used to ship it -> false, a deployment tool is not a "
    "capability of the service itself.\n\n"
    "Answer with a single boolean verdict (true = a result of the same underlying deliverable, "
    "false = a different deliverable or not a result at all)."
)

_MERGEABLE_INTO_ONE_CLAIM_SYSTEM_PROMPT = (
    "You check exactly one thing: if an existing claim were BROADENED - not merely appended to, but rewritten "
    "as one restated claim - to naturally incorporate ONE additional candidate fact, would the result still "
    "read as ONE coherent, sensible accomplishment about the same underlying system or method?\n\n"
    "A claim's scope CAN legitimately widen to cover an additional measured dimension of the SAME system if the "
    "combined statement still reads naturally as one deliverable's combined strengths (for example, a system "
    "praised for being both fast and accurate can be described in a single coherent claim covering both). It "
    "should NOT be judged mergeable if doing so would read as awkwardly stapling together unrelated details, or "
    "as narrating a sequence of separate, disconnected events rather than one accomplishment.\n\n"
    "Example (true): claim about a document-search service's low query latency; candidate fact reports the SAME "
    "service's high search-relevance accuracy -> true, \"Built a document-search service achieving both low "
    "query latency and high search-relevance accuracy\" reads as one coherent, combined accomplishment.\n"
    "Example (false): claim about building a data-processing pipeline; candidate fact describes later "
    "decommissioning that SAME pipeline after a vendor migration -> false, combining \"built X\" with \"later "
    "decommissioned X\" reads as a confusing, even contradictory sequence of events, not one coherent "
    "accomplishment.\n\n"
    "Answer with a single boolean verdict (true = merges into one coherent claim, false = does not)."
)


def _format_fact_list(fact_texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in fact_texts) or "(none)"


def _build_deliverable_prompt(
    claim: CoreClaimMolecule,
    core_fact_texts: Sequence[str],
    candidate_fact_text: str,
) -> str:
    return (
        f'Existing claim: "{claim.claim_text}"\n\n'
        f"Facts already cited by this claim:\n{_format_fact_list(core_fact_texts)}\n\n"
        f'Candidate fact to evaluate: "{candidate_fact_text}"\n\n'
        "Is this candidate fact a result of the same underlying system/method/deliverable as this claim?"
    )


def _build_mergeability_prompt(
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
        "If this claim were broadened to naturally incorporate this candidate fact, would the result still read "
        "as one coherent accomplishment?"
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
    """Decide, for each candidate fact in `support_pool`'s given, already-
    ranked order, whether it should be added as extra support for `claim`.

    Per Phase 3.6/3.7, this is a CLASSIFIER + JUDGE design, not one
    monolithic prompt: two narrow, single-purpose LLM calls per candidate -
    `same_underlying_deliverable` then (only if that passes)
    `mergeable_into_one_claim` - called in short-circuit sequence, and a
    deterministic judge rule (`add_support` only if BOTH pass) combines
    them. Per Phase 3.7, the second classifier judges MERGEABILITY (would
    a broadened restatement of the claim incorporating this fact still
    read as one coherent accomplishment about the same deliverable) rather
    than whether the fact matches the claim's CURRENT literal wording - a
    claim may legitimately broaden to cover an additional measured
    dimension of the same underlying system.

    There is no model-decided early "stop": `support_pool` is already
    ranked and capped (`build_support_pool`), so every candidate is
    evaluated up to `max_additions`, rather than relying on the model's own
    judgment of when remaining candidates aren't worth checking.

    Every decision and its deciding classifier's own reasoning is recorded
    - added facts in `added_support_fact_ids`, everything else in
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

        deliverable_response = llm_provider.call_json(
            prompt=_build_deliverable_prompt(claim, core_fact_texts, atom.fact),
            system_prompt=_SAME_UNDERLYING_DELIVERABLE_SYSTEM_PROMPT,
            json_schema=_VERDICT_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
        )
        if not bool(deliverable_response.get("verdict")):
            excluded_ids.append(atom.id)
            exclusion_reasons.append(f"different_deliverable_or_tooling:{deliverable_response.get('reasoning', '')}")
            continue

        added_so_far_texts = [fact_atoms_by_id[fact_id].fact for fact_id in added_ids if fact_id in fact_atoms_by_id]
        mergeability_response = llm_provider.call_json(
            prompt=_build_mergeability_prompt(claim, core_fact_texts, added_so_far_texts, atom.fact),
            system_prompt=_MERGEABLE_INTO_ONE_CLAIM_SYSTEM_PROMPT,
            json_schema=_VERDICT_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
        )
        if not bool(mergeability_response.get("verdict")):
            excluded_ids.append(atom.id)
            exclusion_reasons.append(f"not_mergeable_into_one_claim:{mergeability_response.get('reasoning', '')}")
            continue

        added_ids.append(atom.id)

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
