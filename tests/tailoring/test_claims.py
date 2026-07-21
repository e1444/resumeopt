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
    generate_core_claim_molecules,
    rank_core_claim_molecules,
    write_core_claim_molecules_json,
    write_unranked_core_claim_molecules_json,
)
from tailoring.models import CoreClaimMolecule, FactAtom


class FakeLLMProvider(LLMProvider):
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
        assert json_schema is not None and json_schema["name"] == "core_claim_molecules"
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


if __name__ == "__main__":
    unittest.main()
