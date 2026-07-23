"""Deterministic tests for `tailoring.claim_discovery` (Phase 3.9 pipeline
integration): the per-sentence-scoped retrieval + generation bridge, plus
its residual whole-pool pass, no real API key needed.

Uses `llm_provider=None` for the embedding/retrieval call (exact-alias
matching only, no semantic escalation) so fixture fact `skill_tags` are
designed to match sentence `skill_terms` exactly - keeps this test focused
on the discovery/coverage logic, not matcher tiering (already covered by
`tests/tailoring/test_retrieval.py`).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from tailoring.claim_discovery import discover_core_claims_for_posting
from tailoring.models import FactAtom, JobRequirements, RequirementSentenceMatch

PROJECT_ID = "proj"

_FACT_ATOMS = (
    FactAtom(id="proj_fact_001", fact="Built a Python microservice.", skill_tags=("python",)),
    FactAtom(id="proj_fact_002", fact="Built a React dashboard.", skill_tags=("react",)),
    FactAtom(id="proj_fact_003", fact="Wrote Dockerfiles for deployment.", skill_tags=("docker",)),
)

_REQUIREMENTS = JobRequirements(
    role_title="Software Engineer",
    seniority="mid",
    industry_domain="tech",
    core_requirements=(),
    nice_to_have=(),
    summary_paragraph="",
    matched_skills=(),
    missing_skills=("python", "react", "docker"),
    requirement_sentences=(
        RequirementSentenceMatch(sentence="Backend Python experience.", skill_terms=("python",)),
        RequirementSentenceMatch(sentence="Frontend React experience.", skill_terms=("react",)),
    ),
)


def _claim(fact_id: str, why: str) -> Dict[str, Any]:
    return {
        "claim_text": f"Claim about {fact_id}.",
        "supporting_fact_ids": [fact_id],
        "target_skills": [],
        "primary_proof": fact_id,
        "rationale": "test",
        "why": why,
        "result": "",
    }


class FakeGenerationProvider(LLMProvider):
    """Returns a different canned claim depending on which sentence (or
    residual, no-sentence) prompt it was called with."""

    def __init__(self):
        super().__init__()
        self.prompts_seen: List[str] = []

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
        few_shot_messages: Optional[List[Dict[str, str]]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        self.prompts_seen.append(prompt)
        if "Backend Python experience." in prompt:
            claims = [_claim("proj_fact_001", "speeding up backend delivery")]
        elif "Frontend React experience." in prompt:
            claims = [_claim("proj_fact_002", "improving the dashboard UX")]
        else:
            # Residual whole-pool pass: no requirement_sentence in the prompt.
            claims = [_claim("proj_fact_003", "supporting reliable deployment")]
        return json.dumps({"claims": claims})


class DiscoverCoreClaimsForPostingTest(unittest.TestCase):
    def test_sentence_seeded_and_residual_claims_are_produced_and_tagged(self) -> None:
        provider = FakeGenerationProvider()

        claims, matches = discover_core_claims_for_posting(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=_REQUIREMENTS,
            reasoning_llm_provider=provider,
        )

        by_fact = {claim.supporting_fact_ids[0]: claim for claim in claims}
        self.assertEqual(set(by_fact.keys()), {"proj_fact_001", "proj_fact_002", "proj_fact_003"})

        # Sentence-seeded claims carry their seeding sentence text.
        self.assertEqual(by_fact["proj_fact_001"].source_requirement_sentence, "Backend Python experience.")
        self.assertEqual(by_fact["proj_fact_002"].source_requirement_sentence, "Frontend React experience.")

        # The residual claim (fact_003, matched no sentence's own skill
        # terms) has no source_requirement_sentence - distinguishable from
        # the sentence-seeded claims above.
        self.assertIsNone(by_fact["proj_fact_003"].source_requirement_sentence)

        # 3 generation calls total: one per sentence + one residual pass.
        self.assertEqual(len(provider.prompts_seen), 3)

        # Fact matches were recorded for every retrieval call made (2
        # sentence-scoped + 1 whole-pool residual).
        self.assertTrue(any(match.fact_id == "proj_fact_003" for match in matches))

    def test_claim_ids_are_unique_across_multiple_generation_calls(self) -> None:
        # Regression: `generate_core_claim_molecules` numbers each call's
        # own claims starting from 1 ("{project_id}_claim_01", ...) - since
        # this function makes one call per sentence plus a residual call,
        # ids would collide across passes (every single-claim call here
        # would otherwise produce the same "proj_claim_01") unless
        # discover_core_claims_for_posting renumbers globally afterward.
        # A collision silently corrupts every downstream dict keyed by
        # claim id (expansion, ranking, synthesis, competition).
        provider = FakeGenerationProvider()

        claims, _ = discover_core_claims_for_posting(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=_REQUIREMENTS,
            reasoning_llm_provider=provider,
        )

        ids = [claim.id for claim in claims]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate claim ids found: {ids}")

    def test_no_requirement_sentences_falls_back_to_single_whole_pool_pass(self) -> None:
        provider = FakeGenerationProvider()
        requirements_no_sentences = JobRequirements(
            role_title="Software Engineer",
            seniority="mid",
            industry_domain="tech",
            core_requirements=(),
            nice_to_have=(),
            summary_paragraph="",
            matched_skills=(),
            missing_skills=("python", "react", "docker"),
            requirement_sentences=(),
        )

        claims, _ = discover_core_claims_for_posting(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=requirements_no_sentences,
            reasoning_llm_provider=provider,
        )

        # Only the residual/whole-pool pass runs - exactly 1 generation call.
        self.assertEqual(len(provider.prompts_seen), 1)
        self.assertEqual(len(claims), 1)
        self.assertIsNone(claims[0].source_requirement_sentence)

    def test_protected_facts_are_excluded_from_every_pass(self) -> None:
        provider = FakeGenerationProvider()

        claims, _ = discover_core_claims_for_posting(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids={"proj_fact_001"},
            requirements=_REQUIREMENTS,
            reasoning_llm_provider=provider,
        )

        cited_fact_ids = {fid for claim in claims for fid in claim.supporting_fact_ids}
        self.assertNotIn("proj_fact_001", cited_fact_ids)


if __name__ == "__main__":
    unittest.main()
