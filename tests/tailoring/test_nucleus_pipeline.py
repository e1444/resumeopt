"""Deterministic tests for `tailoring.nucleus_pipeline` (Phase 3 replacement:
whole-posting-seeded nucleus generation + direct synthesis), no API key
needed.

Covers the module's own non-trivial deterministic behavior that only a
live benchmark script (`tests/tailoring/end_to_end_benchmark.py`) had
exercised until now: candidate short-circuiting (`possible=False`, empty
candidate pool), dropping candidates with missing/invalid
`supporting_fact_ids`, deterministic `target_skills` derivation (the
union of cited facts' own `skill_tags`), `MAX_NUCLEUS_CANDIDATES`
truncation, and the full `discover_and_synthesize_posting_nuclei`
orchestration (retrieval -> nucleus generation -> synthesis).

`FakeLLMProvider` returns each queued response in call order (same
convention as `tests/tailoring/test_verification.py`'s fake) and records
every prompt it was called with, so orchestration tests can assert on
call count and prompt content (e.g. `project_summary` threading) without
depending on real model output.

Uses `llm_provider=None` for the embedding/retrieval call (exact-alias
matching only, no semantic escalation) so fixture fact `skill_tags` are
designed to match target skills exactly - keeps this test focused on
`nucleus_pipeline`'s own logic, not matcher tiering (already covered by
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
from tailoring.models import FactAtom, JobRequirements
from tailoring.nucleus_pipeline import (
    MAX_NUCLEUS_CANDIDATES,
    discover_and_synthesize_posting_nuclei,
    generate_posting_nucleus_claims,
    posting_nucleus_claims_to_dicts,
)

PROJECT_ID = "proj"

_FACT_ATOMS = (
    FactAtom(id="proj_fact_001", fact="Built a Python microservice.", skill_tags=("python", "backend")),
    FactAtom(id="proj_fact_002", fact="Built a React dashboard.", skill_tags=("react", "frontend")),
    FactAtom(id="proj_fact_003", fact="Wrote Dockerfiles for deployment.", skill_tags=("docker",)),
)

_REQUIREMENTS = JobRequirements(
    role_title="Software Engineer",
    seniority="mid",
    industry_domain="tech",
    core_requirements=(),
    nice_to_have=(),
    summary_paragraph="Builds full-stack web applications.",
    matched_skills=(),
    missing_skills=("python", "react", "docker"),
    requirement_sentences=(),
)


class FakeLLMProvider(LLMProvider):
    """Returns each queued response in order, one per `call_json`. Records
    every prompt seen so orchestration tests can assert on call count and
    prompt content."""

    def __init__(self, responses: List[Dict[str, Any]]):
        super().__init__()
        self._responses = list(responses)
        self.call_count = 0
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
        response = self._responses[self.call_count]
        self.call_count += 1
        return json.dumps(response)


def _candidate(why: str, result: str, fact_ids: List[str], rationale: str = "test rationale") -> Dict[str, Any]:
    return {
        "why": why,
        "result": result,
        "supporting_fact_ids": fact_ids,
        "strength_rationale": rationale,
    }


class GeneratePostingNucleusClaimsTest(unittest.TestCase):
    def test_empty_candidate_pool_short_circuits_without_any_llm_call(self) -> None:
        provider = FakeLLMProvider([])

        claims, interpretation, possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, [], provider
        )

        self.assertEqual(claims, [])
        self.assertEqual(interpretation, "")
        self.assertFalse(possible)
        self.assertEqual(provider.call_count, 0)

    def test_possible_false_returns_no_claims_but_keeps_interpretation(self) -> None:
        provider = FakeLLMProvider(
            [{"posting_interpretation": "Not a genuine match.", "possible": False, "candidate_bullets": []}]
        )

        claims, interpretation, possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )

        self.assertEqual(claims, [])
        self.assertEqual(interpretation, "Not a genuine match.")
        self.assertFalse(possible)
        self.assertEqual(provider.call_count, 1)

    def test_candidate_with_empty_supporting_fact_ids_is_dropped(self) -> None:
        provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "x",
                    "possible": True,
                    "candidate_bullets": [_candidate("a why", "", [])],
                }
            ]
        )

        claims, _interpretation, possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )

        self.assertTrue(possible)
        self.assertEqual(claims, [])

    def test_candidate_citing_an_unknown_fact_id_is_dropped_entirely(self) -> None:
        # A candidate mixing one valid and one unknown fact_id is dropped
        # as a whole - never partially filtered down to just the valid id.
        provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "x",
                    "possible": True,
                    "candidate_bullets": [
                        _candidate("a why", "", ["proj_fact_001", "not_a_real_fact_id"]),
                        _candidate("a valid why", "", ["proj_fact_002"]),
                    ],
                }
            ]
        )

        claims, _interpretation, possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )

        self.assertTrue(possible)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].supporting_fact_ids, ("proj_fact_002",))

    def test_target_skills_are_deterministic_deduped_union_of_cited_facts_tags(self) -> None:
        provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "x",
                    "possible": True,
                    "candidate_bullets": [
                        _candidate(
                            "a why spanning both facts",
                            "",
                            ["proj_fact_001", "proj_fact_002", "proj_fact_001"],
                        )
                    ],
                }
            ]
        )

        claims, _interpretation, _possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )

        self.assertEqual(len(claims), 1)
        # Deduped supporting_fact_ids (repeated proj_fact_001 collapses to one).
        self.assertEqual(claims[0].supporting_fact_ids, ("proj_fact_001", "proj_fact_002"))
        # target_skills is the deterministic, order-preserving union of
        # both cited facts' own skill_tags - never LLM-generated.
        self.assertEqual(claims[0].target_skills, ("python", "backend", "react", "frontend"))

    def test_claim_ids_are_assigned_in_order_starting_from_01(self) -> None:
        provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "x",
                    "possible": True,
                    "candidate_bullets": [
                        _candidate("first why", "", ["proj_fact_001"]),
                        _candidate("second why", "", ["proj_fact_002"]),
                    ],
                }
            ]
        )

        claims, _interpretation, _possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )

        self.assertEqual([claim.id for claim in claims], ["proj_claim_01", "proj_claim_02"])

    def test_candidates_beyond_max_nucleus_candidates_are_truncated(self) -> None:
        many_facts = [
            FactAtom(id=f"proj_fact_many_{i:02d}", fact=f"Fact {i}.", skill_tags=(f"skill_{i}",))
            for i in range(MAX_NUCLEUS_CANDIDATES + 5)
        ]
        candidates = [
            _candidate(f"why {i}", "", [fact.id]) for i, fact in enumerate(many_facts)
        ]
        provider = FakeLLMProvider(
            [{"posting_interpretation": "x", "possible": True, "candidate_bullets": candidates}]
        )

        claims, _interpretation, _possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, many_facts, provider
        )

        self.assertEqual(len(claims), MAX_NUCLEUS_CANDIDATES)


class PostingNucleusClaimsToDictsTest(unittest.TestCase):
    def test_round_trips_every_field(self) -> None:
        provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "x",
                    "possible": True,
                    "candidate_bullets": [_candidate("a why", "a result", ["proj_fact_001"])],
                }
            ]
        )

        claims, _interpretation, _possible = generate_posting_nucleus_claims(
            PROJECT_ID, _REQUIREMENTS, list(_FACT_ATOMS), provider
        )
        dicts = posting_nucleus_claims_to_dicts(claims)

        self.assertEqual(
            dicts,
            [
                {
                    "id": "proj_claim_01",
                    "project_id": PROJECT_ID,
                    "supporting_fact_ids": ["proj_fact_001"],
                    "target_skills": ["python", "backend"],
                    "rationale": "test rationale",
                    "why": "a why",
                    "result": "a result",
                }
            ],
        )


class DiscoverAndSynthesizePostingNucleiTest(unittest.TestCase):
    def test_no_matching_facts_short_circuits_before_any_llm_call(self) -> None:
        provider = FakeLLMProvider([])
        requirements = JobRequirements(
            role_title="Software Engineer",
            seniority="mid",
            industry_domain="tech",
            core_requirements=(),
            nice_to_have=(),
            summary_paragraph="",
            matched_skills=(),
            missing_skills=("rust",),  # matches no fixture fact's skill_tags
            requirement_sentences=(),
        )

        claims, proposals, matches = discover_and_synthesize_posting_nuclei(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=requirements,
            nucleus_llm_provider=provider,
            synthesis_llm_provider=provider,
        )

        self.assertEqual(claims, [])
        self.assertEqual(proposals, [])
        self.assertEqual(provider.call_count, 0)
        self.assertTrue(all(not match.included for match in matches))

    def test_full_chain_produces_one_proposal_per_claim_via_synthesis(self) -> None:
        nucleus_provider = FakeLLMProvider(
            [
                {
                    "posting_interpretation": "A full-stack engineering posting.",
                    "possible": True,
                    "candidate_bullets": [
                        _candidate("building maintainable full-stack systems", "", ["proj_fact_001", "proj_fact_002"]),
                    ],
                }
            ]
        )
        synthesis_provider = FakeLLMProvider(
            [{"proposal_text": "Built a full-stack web application.", "reasoning": "x"}]
        )

        claims, proposals, matches = discover_and_synthesize_posting_nuclei(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=_REQUIREMENTS,
            nucleus_llm_provider=nucleus_provider,
            synthesis_llm_provider=synthesis_provider,
            project_summary="A demo project for testing.",
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].id, "proj_claim_01_proposal")
        self.assertEqual(proposals[0].core_claim_id, claims[0].id)
        self.assertEqual(proposals[0].proposal_text, "Built a full-stack web application.")
        self.assertEqual(nucleus_provider.call_count, 1)
        self.assertEqual(synthesis_provider.call_count, 1)
        # project_summary is threaded through to the synthesis prompt.
        self.assertIn("A demo project for testing.", synthesis_provider.prompts_seen[0])
        # The claim only cites proj_fact_001/002 - proj_fact_003 (docker)
        # matched the posting's own target skills too, but was never cited
        # by the (single, fake-provider-controlled) generated claim above.
        self.assertEqual(claims[0].supporting_fact_ids, ("proj_fact_001", "proj_fact_002"))

    def test_zero_claims_produces_zero_synthesis_calls(self) -> None:
        # possible=False from the nucleus call: no candidates to synthesize,
        # so synthesize_proposal must never be invoked.
        nucleus_provider = FakeLLMProvider(
            [{"posting_interpretation": "No coherent theme.", "possible": False, "candidate_bullets": []}]
        )
        synthesis_provider = FakeLLMProvider([])

        claims, proposals, _matches = discover_and_synthesize_posting_nuclei(
            PROJECT_ID,
            _FACT_ATOMS,
            {PROJECT_ID: _FACT_ATOMS},
            protected_fact_ids=set(),
            requirements=_REQUIREMENTS,
            nucleus_llm_provider=nucleus_provider,
            synthesis_llm_provider=synthesis_provider,
        )

        self.assertEqual(claims, [])
        self.assertEqual(proposals, [])
        self.assertEqual(synthesis_provider.call_count, 0)


if __name__ == "__main__":
    unittest.main()
