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

_EVIDENCES_SPECIFIC_CLAIM_SYSTEM_PROMPT = (
    "You check exactly one thing: does ONE candidate fact add genuine EVIDENCE for the SPECIFIC thing an "
    "existing resume claim asserts (its method's design, novelty, or soundness; its measured outcome; or its "
    "scale/scope) - not merely relate to the same broader project or work phase?\n\n"
    "Answer true only if the candidate fact would make a reader more convinced of THIS SPECIFIC assertion. Be "
    "especially strict when the claim is about a method's design/novelty/theoretical soundness: a fact about "
    "GENERAL-PURPOSE tooling or workflow infrastructure (an experiment tracker, a config-management tool, a CI "
    "pipeline) does not evidence that kind of claim just because it was used somewhere in the same project - "
    "such a fact would belong to a separate claim ABOUT the infrastructure/workflow itself. A fact that only "
    "MENTIONS a related team/technology in passing (for example, a backend deliverable that is merely consumed "
    "by a frontend team) is judged by what it itself accomplished, not by who benefits from it.\n\n"
    "Example (true): claim \"Built a REST API for order processing using FastAPI.\" candidate fact \"Added "
    "pagination to the order-processing API to handle large result sets.\" -> true, a concrete detail "
    "strengthening the same API deliverable.\n"
    "Example (false - different accomplishment): the same claim; candidate fact \"Built a React dashboard for "
    "viewing orders.\" -> false, this describes a separate frontend accomplishment.\n"
    "Example (false - generic tooling doesn't evidence a design/novelty claim): claim \"Applied constrained "
    "optimization to jointly optimize predictive accuracy and probabilistic calibration.\" candidate fact "
    "\"Used Weights & Biases (W&B) to manage hyperparameter sweeps and experiment tracking.\" -> false - W&B is "
    "a general-purpose experiment-tracking tool; it provides no evidence about the optimization method's "
    "design, novelty, or soundness.\n\n"
    "Answer with a single boolean verdict (true = adds genuine evidence for this specific assertion, "
    "false = does not)."
)

_PRESERVES_SAME_ACCOMPLISHMENT_SYSTEM_PROMPT = (
    "You check exactly one thing: if ONE candidate fact were added as extra supporting detail to an existing "
    "resume claim, would the expanded claim still describe exactly ONE single accomplishment - not introduce a "
    "second, different one?\n\n"
    "A fact that measures a genuinely DIFFERENT axis or capability of the same system (for example, a "
    "discriminative/classification result added to a claim about generative-model quality) can read as a "
    "separate accomplishment even when it comes from the same underlying system, if the claim's own wording "
    "does not already establish that broader scope.\n\n"
    "Example (true): claim \"Built a REST API for order processing using FastAPI.\" candidate fact \"Added "
    "pagination to the order-processing API to handle large result sets.\" -> true, still the same API "
    "deliverable.\n"
    "Example (false): claim \"Developed a flow-based generative classifier that achieved state-of-the-art "
    "generative quality (0.88 bits/dim).\" candidate fact \"Maintained 98.5% classification accuracy on "
    "MNIST.\" -> false, classification accuracy is a different measured capability than generative quality, "
    "and the claim's own wording only asserts the generative-quality result.\n\n"
    "Answer with a single boolean verdict (true = still exactly one accomplishment, false = a second "
    "accomplishment would be introduced)."
)


def _format_fact_list(fact_texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in fact_texts) or "(none)"


def _build_evidence_prompt(
    claim: CoreClaimMolecule,
    core_fact_texts: Sequence[str],
    candidate_fact_text: str,
) -> str:
    return (
        f'Existing claim: "{claim.claim_text}"\n\n'
        f"Facts already cited by this claim:\n{_format_fact_list(core_fact_texts)}\n\n"
        f'Candidate fact to evaluate: "{candidate_fact_text}"\n\n'
        "Does this candidate fact add genuine evidence for the specific thing this claim asserts?"
    )


def _build_integrity_prompt(
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
        "If this candidate fact were added, would the claim still describe exactly one accomplishment?"
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

    Per Phase 3.6, this is a CLASSIFIER + JUDGE design, not one monolithic
    prompt: two narrow, single-purpose LLM calls per candidate -
    `evidences_specific_claim` then (only if that passes)
    `preserves_same_accomplishment` - called in short-circuit sequence, and
    a deterministic judge rule (`add_support` only if BOTH pass) combines
    them. This replaced an earlier single-prompt 3-way decision after live
    data showed the bundled prompt's own decision was measurably less
    consistent than a dedicated single-purpose classifier asking only one
    of the two questions (see the dev plan's Phase 3.5/3.6).

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

        evidence_response = llm_provider.call_json(
            prompt=_build_evidence_prompt(claim, core_fact_texts, atom.fact),
            system_prompt=_EVIDENCES_SPECIFIC_CLAIM_SYSTEM_PROMPT,
            json_schema=_VERDICT_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
        )
        if not bool(evidence_response.get("verdict")):
            excluded_ids.append(atom.id)
            exclusion_reasons.append(f"no_specific_evidence:{evidence_response.get('reasoning', '')}")
            continue

        added_so_far_texts = [fact_atoms_by_id[fact_id].fact for fact_id in added_ids if fact_id in fact_atoms_by_id]
        integrity_response = llm_provider.call_json(
            prompt=_build_integrity_prompt(claim, core_fact_texts, added_so_far_texts, atom.fact),
            system_prompt=_PRESERVES_SAME_ACCOMPLISHMENT_SYSTEM_PROMPT,
            json_schema=_VERDICT_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
        )
        if not bool(integrity_response.get("verdict")):
            excluded_ids.append(atom.id)
            exclusion_reasons.append(f"introduces_second_accomplishment:{integrity_response.get('reasoning', '')}")
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
