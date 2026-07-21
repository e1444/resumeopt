"""Phase 2: project-level fact retrieval.

For one project's fact pool, matches job-derived target skills against each
fact's own `skill_tags` using the existing tiered matcher package (reused,
not duplicated, per AGENTS.md) - `ExactAliasMatcher` first (free,
deterministic), escalating to `SemanticMatcher` only for a target skill the
exact tier found nothing for (same escalation convention as
`parser.factory.parse_posting`'s own cache-matching loop). Grounded (LLM)
matching is deferred - not yet exercised by this phase's validation; add it
only if the exact+semantic comparison below shows a real recall gap it would
close.

Cross-project exclusion is enforced STRUCTURALLY, not by a per-fact
ownership check: callers pass a `fact_atoms_by_project` mapping keyed by
project id, and this function only ever reads the entry for
`target_project_id` - another project's fact atoms are never even visited,
let alone returned as candidates.

Protected-fact exclusion (facts reserved by non-triaged `keep`/`idk`
baseline bullets, per `tailoring.validation.derive_protection_states`) is
applied via the caller-supplied `protected_fact_ids` set: every candidate
fact still gets matched (for auditability - the pool records how a fact
WOULD have scored even if ultimately excluded), but a protected fact's
`included` is always forced to `False`.

A fact with literally no matching target skill is not a "candidate" at all
(no evidence it relates to this job) and is simply omitted from the
returned pool, rather than recorded with a placeholder match tier.

NOT YET IMPLEMENTED: the dev plan's "broad context" carve-out (a fact
otherwise excluded for project/protection reasons could still be included
if explicitly classified as broad context) - no such classification exists
in the current `FactAtom` schema, so this is deferred rather than guessed
at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from llm import LLMProvider
from matcher import EmbeddingCache, ExactAliasMatcher, SemanticMatcher, SkillRecord

from tailoring.models import FactAtom, JobRequirements, ProjectFactMatch

DEFAULT_EMBEDDING_CACHE_PATH: Optional[Path] = Path("build/cache/skill_embeddings_cache.json")
DEFAULT_MAX_POOL_SIZE = 20

_TIER_RANK: Dict[str, int] = {"exact": 3, "alias": 2, "semantic": 1}


def target_skills_from_requirements(requirements: JobRequirements) -> List[str]:
    """Job-derived skill terms to match against fact `skill_tags`.

    Deliberately combines `matched_skills` (already cache-matched canonical
    names) AND `missing_skills` (raw extracted terms not yet in
    `data/skills.yaml`) - both are genuinely grounded, job-derived terms;
    restricting to only cache-matched skills would badly under-represent
    the posting for a small/generic skills cache (observed live: a real ML
    research posting matched only 3 cache skills, one of them a poor
    semantic match, while `missing_skills` held ~30 genuinely relevant
    terms like "normalizing flows", "Hydra-style configs", "ECE").
    """

    seen: Set[str] = set()
    ordered: List[str] = []
    for skill in (*[match["canonical_name"] for match in requirements.matched_skills], *requirements.missing_skills):
        key = skill.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(skill.strip())
    return ordered


def _fact_records(fact_atoms: Sequence[FactAtom]) -> List[SkillRecord]:
    # Each fact atom becomes one SkillRecord with its skill_tags as ALIASES
    # (not name - the fact id is not a meaningful skill term). A
    # consequence: a target-skill match against a fact's tags always
    # reports as alias tier from ExactAliasMatcher, never exact - that
    # tier is reachable only if a target skill literally equalled a fact's
    # id, which never happens. This is expected, not a bug.
    return [SkillRecord(name=atom.id, aliases=tuple(atom.skill_tags)) for atom in fact_atoms]


def _build_semantic_matcher(
    records: Sequence[SkillRecord],
    llm_provider: Optional[LLMProvider],
    embedding_cache_path: Optional[Path],
) -> Optional[SemanticMatcher]:
    if llm_provider is None:
        return None
    embedding_cache = EmbeddingCache(embedding_cache_path) if embedding_cache_path is not None else None
    try:
        return SemanticMatcher(records, llm_provider, embedding_cache=embedding_cache)
    except NotImplementedError:
        # Provider doesn't support embeddings (e.g. Anthropic, Ollama today).
        return None


def retrieve_project_fact_pool(
    target_project_id: str,
    fact_atoms_by_project: Dict[str, Sequence[FactAtom]],
    protected_fact_ids: Set[str],
    target_skills: Sequence[str],
    llm_provider: Optional[LLMProvider] = None,
    embedding_cache_path: Optional[Path] = DEFAULT_EMBEDDING_CACHE_PATH,
    max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
) -> List[ProjectFactMatch]:
    """Retrieve one auditable pool of job-relevant facts for one project.

    Returns a `ProjectFactMatch` for every fact that matched at least one
    target skill, each flagged `included` (True/False) with an
    `exclusion_reason` when False. Facts belonging to any OTHER project in
    `fact_atoms_by_project` are never read at all.
    """

    fact_atoms = fact_atoms_by_project.get(target_project_id, ())
    if not fact_atoms:
        return []

    atoms_by_id = {atom.id: atom for atom in fact_atoms}
    records = _fact_records(fact_atoms)
    exact_matcher = ExactAliasMatcher(records)
    semantic_matcher = _build_semantic_matcher(records, llm_provider, embedding_cache_path)

    # fact_id -> (match_tier, matched_target_skill, score)
    best_by_fact_id: Dict[str, Tuple[str, str, float]] = {}

    for target_skill in target_skills:
        candidates = exact_matcher.match(target_skill)
        if not candidates and semantic_matcher is not None:
            candidates = semantic_matcher.match(target_skill)

        for candidate in candidates:
            fact_id = candidate.canonical_name  # SkillRecord.name == fact.id
            score = candidate.similarity if candidate.similarity is not None else candidate.confidence
            existing = best_by_fact_id.get(fact_id)
            if existing is None or _TIER_RANK.get(candidate.match_type, 0) > _TIER_RANK.get(existing[0], 0) or (
                _TIER_RANK.get(candidate.match_type, 0) == _TIER_RANK.get(existing[0], 0) and score > existing[2]
            ):
                best_by_fact_id[fact_id] = (candidate.match_type, target_skill, score)

    protected_matches: List[Tuple[str, str, str, float]] = []
    eligible_matches: List[Tuple[str, str, str, float]] = []
    for fact_id, (tier, target_skill, score) in best_by_fact_id.items():
        if fact_id in protected_fact_ids:
            protected_matches.append((fact_id, tier, target_skill, score))
        else:
            eligible_matches.append((fact_id, tier, target_skill, score))

    eligible_matches.sort(key=lambda item: item[3], reverse=True)

    results: List[ProjectFactMatch] = []
    for fact_id, tier, target_skill, score in protected_matches:
        results.append(
            ProjectFactMatch(
                fact_id=fact_id,
                project_id=target_project_id,
                match_tier=tier,  # type: ignore[arg-type]
                matched_target_skill=target_skill,
                score=round(score, 4),
                included=False,
                exclusion_reason="protected_by_baseline_bullet",
            )
        )

    for index, (fact_id, tier, target_skill, score) in enumerate(eligible_matches):
        included = index < max_pool_size
        results.append(
            ProjectFactMatch(
                fact_id=fact_id,
                project_id=target_project_id,
                match_tier=tier,  # type: ignore[arg-type]
                matched_target_skill=target_skill,
                score=round(score, 4),
                included=included,
                exclusion_reason=None if included else "pool_capped",
            )
        )

    return results


def project_fact_matches_to_dicts(matches: Sequence[ProjectFactMatch]) -> List[dict]:
    return [
        {
            "fact_id": match.fact_id,
            "project_id": match.project_id,
            "match_tier": match.match_tier,
            "matched_target_skill": match.matched_target_skill,
            "score": match.score,
            "included": match.included,
            "exclusion_reason": match.exclusion_reason,
        }
        for match in matches
    ]


def write_project_fact_matches_json(matches: Sequence[ProjectFactMatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(project_fact_matches_to_dicts(matches), handle, indent=2)
