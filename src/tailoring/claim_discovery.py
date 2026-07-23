"""Phase 3.9: posting-sentence-seeded claim discovery, bridging Phase 2
(fact retrieval) and Phase 3 (claim generation/ranking) into one per-
sentence-scoped flow, with a residual whole-pool pass for coverage.

This module exists because the corrected Phase 3.9 scope is NOT just
`tailoring.claims.generate_core_claim_molecules`'s optional
`requirement_sentence` parameter in isolation - a real caller must also
decide WHICH facts belong to each sentence's own candidate pool (Phase 2's
job), and what to do with facts that don't cleanly belong to any one
sentence. Neither `tailoring.requirements` nor `tailoring.retrieval` nor
`tailoring.claims` owns that cross-phase decision on its own, so it lives
here rather than being duplicated ad hoc by every caller (e.g.
`tests/tailoring/end_to_end_benchmark.py`).

Design (human-approved, 2026-07-22):
- For each `JobRequirements.requirement_sentences` entry, retrieve a
  candidate pool scoped to JUST that sentence's own skill terms (reusing
  `tailoring.retrieval.retrieve_project_fact_pool` unchanged - it already
  accepts an arbitrary `target_skills` list per call, no Phase 2 code
  changes needed), then run Phase 3's existing, already-atomicity-
  preserving `generate_core_claim_molecules` scoped to that pool, with the
  sentence's own text passed as grounding context.
- A fact that matched INTO at least one sentence's own retrieval is
  considered "captured" by the sentence-seeded pass, whether or not the
  generation call actually cited it in a resulting claim - declining to
  use a matched fact is a legitimate generation-time judgment, not a
  coverage gap to re-surface.
- A residual, POSTING-AGNOSTIC pass (the original Phase 3 behavior, no
  `requirement_sentence`) still runs afterward over any fact that matched
  the posting's whole flattened target-skill list but was never captured
  by any individual sentence's own retrieval - this preserves today's
  coverage guarantee and is trivial to disable later (skip the residual
  pass entirely) if it turns out not to be useful.
- The two kinds of claims are always distinguishable in the artifact:
  `CoreClaimMolecule.source_requirement_sentence` is the seeding sentence
  text for a sentence-seeded claim, `None` for a residual/whole-pool claim.
- If `requirements.requirement_sentences` is empty (e.g. a posting that
  predates this field, or one loaded from an older persisted
  `requirements.json`), this degrades to exactly today's single whole-pool
  behavior - not a design gap, a deliberate, tested fallback for genuinely
  absent per-sentence data.

NOT decided/implemented yet (deferred pending investigation, per explicit
instruction not to guess at a mechanism): cross-sentence deduplication for
the case where two different requirement sentences' own generation calls
independently produce similar/overlapping claims. No merge step exists
here - callers see every claim produced by every pass, as-is.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from llm import DEFAULT_REASONING_EFFORT, LLMProvider

from tailoring.claims import generate_core_claim_molecules
from tailoring.models import CoreClaimMolecule, FactAtom, JobRequirements, ProjectFactMatch
from tailoring.retrieval import retrieve_project_fact_pool, target_skills_from_requirements


def discover_core_claims_for_posting(
    project_id: str,
    fact_atoms: Sequence[FactAtom],
    fact_atoms_by_project: Dict[str, Sequence[FactAtom]],
    protected_fact_ids: Set[str],
    requirements: JobRequirements,
    reasoning_llm_provider: LLMProvider,
    embedding_llm_provider: Optional[LLMProvider] = None,
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
) -> Tuple[List[CoreClaimMolecule], List[ProjectFactMatch]]:
    """Discover core claim molecules for one project against one posting.

    Returns `(claims, fact_matches)`. `claims` combines every sentence-
    seeded claim (in `requirement_sentences` order) followed by any
    residual whole-pool claims. `fact_matches` is every `ProjectFactMatch`
    produced by every retrieval call made along the way (one per sentence,
    plus the residual whole-pool retrieval) - kept as separate per-call
    records rather than deduplicated/merged, since a fact can legitimately
    be a candidate under more than one sentence and that is itself useful
    audit information, not noise.
    """

    fact_atoms_by_id = {atom.id: atom for atom in fact_atoms}
    all_fact_matches: List[ProjectFactMatch] = []
    claims: List[CoreClaimMolecule] = []
    captured_fact_ids: Set[str] = set()

    for sentence_match in requirements.requirement_sentences:
        sentence_matches = retrieve_project_fact_pool(
            project_id,
            fact_atoms_by_project,
            protected_fact_ids,
            sentence_match.skill_terms,
            llm_provider=embedding_llm_provider,
        )
        all_fact_matches.extend(sentence_matches)

        sentence_fact_ids = {match.fact_id for match in sentence_matches}
        captured_fact_ids.update(sentence_fact_ids)

        candidate_atoms = [
            fact_atoms_by_id[match.fact_id]
            for match in sentence_matches
            if match.included and match.fact_id in fact_atoms_by_id
        ]
        if not candidate_atoms:
            continue

        claims.extend(
            generate_core_claim_molecules(
                project_id,
                candidate_atoms,
                reasoning_llm_provider,
                reasoning_effort=reasoning_effort,
                requirement_sentence=sentence_match.sentence,
            )
        )

    # Residual whole-pool pass: facts matched against the posting's whole
    # flattened target-skill list but never captured by any one sentence's
    # own retrieval above (or covering the entire pool, when the posting
    # has no requirement_sentences at all - the backward-compatible
    # fallback).
    whole_pool_target_skills = target_skills_from_requirements(requirements)
    whole_pool_matches = retrieve_project_fact_pool(
        project_id,
        fact_atoms_by_project,
        protected_fact_ids,
        whole_pool_target_skills,
        llm_provider=embedding_llm_provider,
    )
    all_fact_matches.extend(whole_pool_matches)

    leftover_atoms = [
        fact_atoms_by_id[match.fact_id]
        for match in whole_pool_matches
        if match.included and match.fact_id not in captured_fact_ids and match.fact_id in fact_atoms_by_id
    ]
    if leftover_atoms:
        claims.extend(
            generate_core_claim_molecules(
                project_id,
                leftover_atoms,
                reasoning_llm_provider,
                reasoning_effort=reasoning_effort,
            )
        )

    # Each `generate_core_claim_molecules` call numbers its OWN claims
    # starting from 1 (`{project_id}_claim_01`, ...) - correct for a
    # single whole-pool call, but this function makes one call PER
    # sentence plus a residual call, so those per-call ids collide across
    # passes (e.g. two different sentences would each produce their own
    # "..._claim_01"). Renumber globally, once, after every pass has run,
    # so every claim this function returns has a genuinely unique id -
    # every downstream phase (expansion, synthesis, ranking, competition)
    # keys dicts by claim id and would otherwise silently overwrite
    # distinct claims that happened to share a per-call-local id.
    claims = [replace(claim, id=f"{project_id}_claim_{index + 1:02d}") for index, claim in enumerate(claims)]

    return claims, all_fact_matches
