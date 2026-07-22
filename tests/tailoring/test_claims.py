"""Deterministic tests for `tailoring.claims` (Phase 3, no API key needed).

`FakeLLMProvider` mirrors the real `_CLAIM_GENERATION_JSON_SCHEMA` shape
(same convention as `tests/tailoring/test_triage.py`), keyed by project_id
so a test can select which canned response a given call should return.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from tailoring.claims import (
    MAX_SUPPORTING_FACTS,
    classify_claim_concreteness,
    generate_core_claim_molecules,
    is_generation_validation_failure,
    rank_core_claim_molecules,
    write_core_claim_molecules_json,
    write_unranked_core_claim_molecules_json,
)
from tailoring.models import CoreClaimMolecule, FactAtom


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: Dict[str, Any]):
        super().__init__()
        self._response = response
        self.last_prompt: Optional[str] = None

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
        assert json_schema is not None and json_schema["name"] == "core_claim_molecules"
        self.last_prompt = prompt
        return json.dumps(self._response)


_ATOMS = (
    FactAtom(id="p_fact_001", fact="Built a React UI.", skill_tags=("react", "frontend")),
    FactAtom(id="p_fact_002", fact="Added CSS styling.", skill_tags=("css", "frontend")),
)


class GenerateCoreClaimMoleculesTest(unittest.TestCase):
    def test_empty_pool_returns_no_claims_without_calling_llm(self) -> None:
        provider = FakeLLMProvider({"claims": []})
        molecules = generate_core_claim_molecules("p", (), provider)

        self.assertEqual(molecules, [])

    def test_empty_claims_response_is_a_valid_no_claim_outcome(self) -> None:
        provider = FakeLLMProvider({"claims": []})
        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(molecules, [])

    def test_valid_claim_is_parsed_with_no_non_advancement_reason(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built a React-based UI with custom CSS styling.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_002"],
                        "target_skills": ["react", "css"],
                        "primary_proof": "React UI + CSS styling",
                        "rationale": "Both facts describe the same frontend UI work.",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(len(molecules), 1)
        molecule = molecules[0]
        self.assertEqual(molecule.id, "p_claim_01")
        self.assertEqual(molecule.project_id, "p")
        self.assertEqual(molecule.supporting_fact_ids, ("p_fact_001", "p_fact_002"))
        self.assertIsNone(molecule.non_advancement_reason)

    def test_why_result_nucleus_is_parsed_from_response(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built a React-based UI with custom CSS styling.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_002"],
                        "target_skills": ["react", "css"],
                        "primary_proof": "React UI + CSS styling",
                        "rationale": "Both facts describe the same frontend UI work.",
                        "why": "improving the product's visual polish",
                        "result": "a fully styled, production-ready UI",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules[0].why, "improving the product's visual polish")
        self.assertEqual(molecules[0].result, "a fully styled, production-ready UI")

    def test_why_result_default_to_empty_string_when_absent_from_response(self) -> None:
        # A response that omits why/result entirely (e.g. an older/malformed
        # payload) must never crash generation - both fields default to "",
        # the same "no separable result/nucleus stated" sentinel used when
        # the model itself legitimately leaves result empty.
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built a React-based UI with custom CSS styling.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_002"],
                        "target_skills": ["react", "css"],
                        "primary_proof": "React UI + CSS styling",
                        "rationale": "Both facts describe the same frontend UI work.",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(molecules[0].why, "")
        self.assertEqual(molecules[0].result, "")

    def test_unsupported_fact_id_is_flagged_not_silently_trusted(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built something.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_999"],
                        "target_skills": ["react"],
                        "primary_proof": "x",
                        "rationale": "x",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(len(molecules), 1)
        self.assertIsNotNone(molecules[0].non_advancement_reason)
        self.assertIn("p_fact_999", molecules[0].non_advancement_reason)

    def test_duplicate_supporting_fact_ids_are_deduplicated(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built a React UI.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_001", "p_fact_002"],
                        "target_skills": ["react"],
                        "primary_proof": "x",
                        "rationale": "x",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules[0].supporting_fact_ids, ("p_fact_001", "p_fact_002"))
        self.assertIsNone(molecules[0].non_advancement_reason)

    def test_empty_supporting_fact_ids_flagged_with_invalid_count_reason(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built something.",
                        "supporting_fact_ids": [],
                        "target_skills": ["react"],
                        "primary_proof": "x",
                        "rationale": "x",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules[0].non_advancement_reason, "invalid_supporting_fact_count:0")
        self.assertTrue(is_generation_validation_failure(molecules[0]))

    def test_too_many_supporting_fact_ids_flagged_with_invalid_count_reason(self) -> None:
        big_pool = tuple(
            FactAtom(id=f"p_fact_{i:03d}", fact=f"Fact {i}.") for i in range(1, MAX_SUPPORTING_FACTS + 2)
        )
        too_many_ids = [atom.id for atom in big_pool]  # MAX_SUPPORTING_FACTS + 1 unique, known ids
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built something huge.",
                        "supporting_fact_ids": too_many_ids,
                        "target_skills": ["react"],
                        "primary_proof": "x",
                        "rationale": "x",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules("p", big_pool, provider)

        self.assertEqual(len(molecules), 1)
        self.assertEqual(
            molecules[0].non_advancement_reason, f"invalid_supporting_fact_count:{MAX_SUPPORTING_FACTS + 1}"
        )
        self.assertTrue(is_generation_validation_failure(molecules[0]))


class RankCoreClaimMoleculesTest(unittest.TestCase):
    def test_selects_top_claims_and_reserves_facts(self) -> None:
        claims = [
            CoreClaimMolecule(
                id="c1",
                project_id="p",
                claim_text="Claim 1",
                supporting_fact_ids=("f1", "f2"),
                target_skills=("python", "django"),
                primary_proof="x",
                rationale="x",
            ),
            CoreClaimMolecule(
                id="c2",
                project_id="p",
                claim_text="Claim 2 (overlaps c1 entirely)",
                supporting_fact_ids=("f1", "f2"),
                target_skills=("python",),
                primary_proof="x",
                rationale="x",
            ),
            CoreClaimMolecule(
                id="c3",
                project_id="p",
                claim_text="Claim 3 (independent facts)",
                supporting_fact_ids=("f3", "f4"),
                target_skills=("hydra",),
                primary_proof="x",
                rationale="x",
            ),
        ]

        ranked = rank_core_claim_molecules(claims, max_selected=2)
        ranked_by_id = {claim.id: claim for claim in ranked}

        # c1 has broader skill coverage than c2 (same facts) so it should
        # win the first slot; c2 then has zero non-reserved facts left and
        # must not be selected; c3 has fully independent facts and should
        # fill the second slot.
        self.assertEqual(ranked_by_id["c1"].rank, 1)
        self.assertEqual(ranked_by_id["c3"].rank, 2)
        self.assertIsNone(ranked_by_id["c2"].rank)
        self.assertEqual(ranked_by_id["c2"].non_advancement_reason, "not_selected_this_round")

    def test_invalid_claims_are_left_untouched(self) -> None:
        invalid_claim = CoreClaimMolecule(
            id="c1",
            project_id="p",
            claim_text="Claim citing an unknown fact",
            supporting_fact_ids=("f1", "f999"),
            target_skills=("python",),
            primary_proof="x",
            rationale="x",
            non_advancement_reason="unsupported_fact_ids:['f999']",
        )

        ranked = rank_core_claim_molecules([invalid_claim], max_selected=2)

        self.assertEqual(ranked, [invalid_claim])

    def test_respects_max_selected(self) -> None:
        claims = [
            CoreClaimMolecule(
                id=f"c{i}",
                project_id="p",
                claim_text=f"Claim {i}",
                supporting_fact_ids=(f"f{i}",),
                target_skills=("python",),
                primary_proof="x",
                rationale="x",
            )
            for i in range(4)
        ]

        ranked = rank_core_claim_molecules(claims, max_selected=2)
        selected = [claim for claim in ranked if claim.rank is not None]

        self.assertEqual(len(selected), 2)

    def test_not_selected_this_round_claims_remain_eligible_on_rerank(self) -> None:
        # Independent (non-overlapping) fact sets so all 3 CAN be selected
        # once max_selected allows it; c1 scores highest (more skills).
        claims = [
            CoreClaimMolecule(
                id="c1",
                project_id="p",
                claim_text="Claim 1",
                supporting_fact_ids=("f1", "f2"),
                target_skills=("python", "django"),
                primary_proof="x",
                rationale="x",
            ),
            CoreClaimMolecule(
                id="c2",
                project_id="p",
                claim_text="Claim 2",
                supporting_fact_ids=("f3",),
                target_skills=("skillb",),
                primary_proof="x",
                rationale="x",
            ),
            CoreClaimMolecule(
                id="c3",
                project_id="p",
                claim_text="Claim 3",
                supporting_fact_ids=("f4",),
                target_skills=("skillc",),
                primary_proof="x",
                rationale="x",
            ),
        ]

        first_round = rank_core_claim_molecules(claims, max_selected=1)
        first_by_id = {claim.id: claim for claim in first_round}
        self.assertEqual(first_by_id["c1"].rank, 1)
        self.assertIsNone(first_by_id["c2"].rank)
        self.assertEqual(first_by_id["c2"].non_advancement_reason, "not_selected_this_round")
        self.assertIsNone(first_by_id["c3"].rank)
        self.assertEqual(first_by_id["c3"].non_advancement_reason, "not_selected_this_round")

        # Rerank the FIRST ROUND'S OWN OUTPUT (not the original claims) with
        # a larger max_selected - c2/c3 must still be selectable even though
        # they currently carry "not_selected_this_round", and their stale
        # reason must be cleared once selected.
        second_round = rank_core_claim_molecules(first_round, max_selected=3)
        second_by_id = {claim.id: claim for claim in second_round}
        self.assertEqual(second_by_id["c1"].rank, 1)
        self.assertIsNotNone(second_by_id["c2"].rank)
        self.assertIsNone(second_by_id["c2"].non_advancement_reason)
        self.assertIsNotNone(second_by_id["c3"].rank)
        self.assertIsNone(second_by_id["c3"].non_advancement_reason)

        # Rerank AGAIN with max_selected back down to 1 - c2/c3 must lose
        # their now-stale rank (not keep it) alongside the fresh reason.
        third_round = rank_core_claim_molecules(second_round, max_selected=1)
        third_by_id = {claim.id: claim for claim in third_round}
        self.assertEqual(third_by_id["c1"].rank, 1)
        self.assertIsNone(third_by_id["c2"].rank)
        self.assertEqual(third_by_id["c2"].non_advancement_reason, "not_selected_this_round")
        self.assertIsNone(third_by_id["c3"].rank)
        self.assertEqual(third_by_id["c3"].non_advancement_reason, "not_selected_this_round")

    def test_generation_validation_failure_never_becomes_eligible_on_rerank(self) -> None:
        invalid_claim = CoreClaimMolecule(
            id="c1",
            project_id="p",
            claim_text="Claim citing an unknown fact",
            supporting_fact_ids=("f1", "f999"),
            target_skills=("python",),
            primary_proof="x",
            rationale="x",
            non_advancement_reason="unsupported_fact_ids:['f999']",
        )

        # Even with room to select it and no competing candidates, a
        # genuine generation-validation failure must never be selected.
        ranked = rank_core_claim_molecules([invalid_claim], max_selected=5)

        self.assertEqual(ranked, [invalid_claim])


class RequirementSentenceContextTest(unittest.TestCase):
    """Phase 3.9: `requirement_sentence` is grounding context passed into
    the SAME generation call, not a new schema."""

    def test_requirement_sentence_is_included_in_the_prompt(self) -> None:
        provider = FakeLLMProvider({"claims": []})

        generate_core_claim_molecules(
            "p", _ATOMS, provider, requirement_sentence="Experience with Python microservices."
        )

        self.assertIn("Experience with Python microservices.", provider.last_prompt)

    def test_no_requirement_sentence_reproduces_prior_prompt_shape(self) -> None:
        provider = FakeLLMProvider({"claims": []})

        generate_core_claim_molecules("p", _ATOMS, provider)

        self.assertNotIn("job-posting requirement", provider.last_prompt)

    def test_claims_still_parsed_normally_with_requirement_sentence(self) -> None:
        provider = FakeLLMProvider(
            {
                "claims": [
                    {
                        "claim_text": "Built a React-based UI with custom CSS styling.",
                        "supporting_fact_ids": ["p_fact_001", "p_fact_002"],
                        "target_skills": ["react", "css"],
                        "primary_proof": "React UI + CSS styling",
                        "rationale": "Both facts describe the same frontend UI work.",
                        "why": "improving the product's visual polish",
                        "result": "",
                    }
                ]
            }
        )

        molecules = generate_core_claim_molecules(
            "p", _ATOMS, provider, requirement_sentence="Experience with frontend UI development."
        )

        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules[0].why, "improving the product's visual polish")


class FakeConcretenessProvider(LLMProvider):
    def __init__(self, response: Dict[str, Any]):
        super().__init__()
        self._response = response

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
        assert json_schema is not None and json_schema["name"] == "claim_concreteness"
        return json.dumps(self._response)


class ClassifyClaimConcretenessTest(unittest.TestCase):
    def test_concrete_metric_claim_returns_true(self) -> None:
        claim = CoreClaimMolecule(
            id="c1",
            project_id="p",
            claim_text="x",
            supporting_fact_ids=("p_fact_001",),
            target_skills=(),
            primary_proof="x",
            rationale="x",
            why="reducing product-search latency",
            result="cut average product-search latency from 250ms to 40ms",
        )
        fact_atoms_by_id = {"p_fact_001": FactAtom(id="p_fact_001", fact="Cut search latency from 250ms to 40ms.")}
        provider = FakeConcretenessProvider({"concrete": True, "reasoning": "hard metric"})

        result = classify_claim_concreteness(claim, fact_atoms_by_id, provider)

        self.assertTrue(result)

    def test_generic_claim_returns_false(self) -> None:
        claim = CoreClaimMolecule(
            id="c2",
            project_id="p",
            claim_text="x",
            supporting_fact_ids=("p_fact_002",),
            target_skills=(),
            primary_proof="x",
            rationale="x",
            why="giving customers visibility into their order activity",
            result="",
        )
        fact_atoms_by_id = {"p_fact_002": FactAtom(id="p_fact_002", fact="Built a customer dashboard.")}
        provider = FakeConcretenessProvider({"concrete": False, "reasoning": "routine, no metric"})

        result = classify_claim_concreteness(claim, fact_atoms_by_id, provider)

        self.assertFalse(result)


class WriteClaimJsonTest(unittest.TestCase):
    def test_write_unranked_and_ranked_json(self) -> None:
        claim = CoreClaimMolecule(
            id="c1",
            project_id="p",
            claim_text="Claim 1",
            supporting_fact_ids=("f1",),
            target_skills=("python",),
            primary_proof="x",
            rationale="x",
            rank=1,
            why="proving reliability under load",
            result="",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            unranked_path = Path(tmp_dir) / "unranked_core_claim_molecules.json"
            ranked_path = Path(tmp_dir) / "core_claim_molecules.json"
            write_unranked_core_claim_molecules_json([claim], unranked_path)
            write_core_claim_molecules_json([claim], ranked_path)

            with unranked_path.open() as handle:
                unranked_data = json.load(handle)
            with ranked_path.open() as handle:
                ranked_data = json.load(handle)

            self.assertEqual(unranked_data[0]["id"], "c1")
            self.assertEqual(ranked_data[0]["rank"], 1)
            self.assertEqual(ranked_data[0]["why"], "proving reliability under load")
            self.assertEqual(ranked_data[0]["result"], "")


if __name__ == "__main__":
    unittest.main()
