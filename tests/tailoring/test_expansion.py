"""Deterministic tests for `tailoring.expansion` (Phase 4/3.6/3.7, no API key needed).

`FakeLLMProvider` mirrors the real `_VERDICT_JSON_SCHEMA` shape and lets a
test queue up canned per-call verdicts, in call order (same convention as
`tests/tailoring/test_claims.py`'s `FakeLLMProvider`). Per Phase 3.6/3.7,
`expand_claim_molecule` makes UP TO 2 calls per candidate fact - a
`same_underlying_deliverable` classifier, then (only if that passes) a
`mergeable_into_one_claim` classifier - so tests must queue responses
in that exact per-candidate order.
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
from tailoring.expansion import (
    MAX_SUPPORT_ADDITIONS,
    apply_verbosity_prefilter,
    build_support_pool,
    estimate_expanded_line_count,
    expand_claim_molecule,
    write_expanded_claim_molecules_json,
)
from tailoring.models import CoreClaimMolecule, ExpandedClaimMolecule, FactAtom


class FakeLLMProvider(LLMProvider):
    """Returns each queued response in order, one per `call_json`."""

    def __init__(self, responses: List[Dict[str, Any]]):
        super().__init__()
        self._responses = list(responses)
        self.call_count = 0

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
        assert json_schema is not None and json_schema["name"] == "single_purpose_verdict"
        response = self._responses[self.call_count]
        self.call_count += 1
        return json.dumps(response)


_CORE_CLAIM = CoreClaimMolecule(
    id="p_claim_01",
    project_id="p",
    claim_text="Built a REST API for order processing using FastAPI.",
    supporting_fact_ids=("p_fact_001",),
    target_skills=("fastapi", "backend"),
    primary_proof="FastAPI REST API",
    rationale="One backend deliverable.",
)

_FACT_ATOMS_BY_ID = {
    "p_fact_001": FactAtom(id="p_fact_001", fact="Built a REST API for order processing using FastAPI."),
    "p_fact_002": FactAtom(id="p_fact_002", fact="Added pagination to the order-processing API."),
    "p_fact_003": FactAtom(id="p_fact_003", fact="Built a React dashboard for viewing orders."),
    "p_fact_004": FactAtom(id="p_fact_004", fact="Added request-rate limiting to the order-processing API."),
    "p_fact_005": FactAtom(id="p_fact_005", fact="Redesigned the marketing landing page."),
}


class BuildSupportPoolTest(unittest.TestCase):
    def test_excludes_already_used_facts(self) -> None:
        fact_atoms = list(_FACT_ATOMS_BY_ID.values())

        pool = build_support_pool(_CORE_CLAIM, fact_atoms, llm_provider=None)

        pool_ids = {atom.id for atom in pool}
        self.assertNotIn("p_fact_001", pool_ids)

    def test_falls_back_to_input_order_without_llm_provider(self) -> None:
        fact_atoms = list(_FACT_ATOMS_BY_ID.values())

        pool = build_support_pool(_CORE_CLAIM, fact_atoms, llm_provider=None, max_pool_size=2)

        self.assertEqual([atom.id for atom in pool], ["p_fact_002", "p_fact_003"])

    def test_respects_max_pool_size(self) -> None:
        fact_atoms = list(_FACT_ATOMS_BY_ID.values())

        pool = build_support_pool(_CORE_CLAIM, fact_atoms, llm_provider=None, max_pool_size=1)

        self.assertEqual(len(pool), 1)

    def test_empty_when_all_facts_already_used(self) -> None:
        claim = CoreClaimMolecule(
            id="c",
            project_id="p",
            claim_text="x",
            supporting_fact_ids=("p_fact_001", "p_fact_002", "p_fact_003", "p_fact_004", "p_fact_005"),
            target_skills=(),
            primary_proof="x",
            rationale="x",
        )

        pool = build_support_pool(claim, list(_FACT_ATOMS_BY_ID.values()), llm_provider=None)

        self.assertEqual(pool, [])


class ExpandClaimMoleculeTest(unittest.TestCase):
    def test_empty_support_pool_short_circuits_without_calling_llm(self) -> None:
        provider = FakeLLMProvider([])

        expansion = expand_claim_molecule(_CORE_CLAIM, [], _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.stop_reason, "empty_support_pool")
        self.assertEqual(provider.call_count, 0)

    def test_both_verdicts_true_adds_support(self) -> None:
        pool = [_FACT_ATOMS_BY_ID["p_fact_002"]]
        provider = FakeLLMProvider(
            [
                {"verdict": True, "reasoning": "strengthens the same API deliverable"},
                {"verdict": True, "reasoning": "still one accomplishment"},
            ]
        )

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.added_support_fact_ids, ("p_fact_002",))
        self.assertEqual(expansion.excluded_fact_ids, ())
        self.assertEqual(expansion.stop_reason, "pool_exhausted")
        self.assertEqual(provider.call_count, 2)

    def test_evidence_failure_short_circuits_without_calling_integrity_classifier(self) -> None:
        pool = [_FACT_ATOMS_BY_ID["p_fact_003"]]
        provider = FakeLLMProvider([{"verdict": False, "reasoning": "describes a separate frontend accomplishment"}])

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.added_support_fact_ids, ())
        self.assertEqual(expansion.excluded_fact_ids, ("p_fact_003",))
        self.assertTrue(expansion.exclusion_reasons[0].startswith("different_deliverable_or_tooling:"))
        # Only 1 call made - the integrity classifier is never reached once
        # the evidence classifier already rejects the candidate.
        self.assertEqual(provider.call_count, 1)

    def test_evidence_passes_but_integrity_fails_still_excludes(self) -> None:
        pool = [_FACT_ATOMS_BY_ID["p_fact_002"]]
        provider = FakeLLMProvider(
            [
                {"verdict": True, "reasoning": "looks relevant"},
                {"verdict": False, "reasoning": "introduces a second accomplishment"},
            ]
        )

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.added_support_fact_ids, ())
        self.assertEqual(expansion.excluded_fact_ids, ("p_fact_002",))
        self.assertTrue(expansion.exclusion_reasons[0].startswith("not_mergeable_into_one_claim:"))
        self.assertEqual(provider.call_count, 2)

    def test_mixed_candidates_are_recorded_correctly(self) -> None:
        pool = [_FACT_ATOMS_BY_ID["p_fact_002"], _FACT_ATOMS_BY_ID["p_fact_003"]]
        provider = FakeLLMProvider(
            [
                {"verdict": True, "reasoning": "strengthens the API deliverable"},  # fact_002 evidence
                {"verdict": True, "reasoning": "still one accomplishment"},  # fact_002 integrity
                {"verdict": False, "reasoning": "describes a separate frontend accomplishment"},  # fact_003 evidence
            ]
        )

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.added_support_fact_ids, ("p_fact_002",))
        self.assertEqual(expansion.excluded_fact_ids, ("p_fact_003",))
        self.assertEqual(len(expansion.excluded_fact_ids), len(expansion.exclusion_reasons))
        self.assertEqual(expansion.stop_reason, "pool_exhausted")
        self.assertEqual(provider.call_count, 3)

    def test_max_additions_cap_is_enforced(self) -> None:
        pool = [
            _FACT_ATOMS_BY_ID["p_fact_002"],
            _FACT_ATOMS_BY_ID["p_fact_003"],
            _FACT_ATOMS_BY_ID["p_fact_004"],
            _FACT_ATOMS_BY_ID["p_fact_005"],
        ]
        # 3 additions x 2 calls each (evidence=True, integrity=True) = 6 calls;
        # the 4th candidate is never reached once max_additions is hit.
        provider = FakeLLMProvider([{"verdict": True, "reasoning": "ok"} for _ in range(6)])

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider, max_additions=3)

        self.assertEqual(len(expansion.added_support_fact_ids), 3)
        self.assertEqual(expansion.stop_reason, "max_additions_reached")
        self.assertEqual(provider.call_count, 6)
        self.assertLessEqual(len(expansion.added_support_fact_ids), MAX_SUPPORT_ADDITIONS)

    def test_missing_verdict_key_defaults_to_excluded_not_silently_dropped(self) -> None:
        pool = [_FACT_ATOMS_BY_ID["p_fact_002"]]
        provider = FakeLLMProvider([{"reasoning": "unsure"}])  # no "verdict" key at all

        expansion = expand_claim_molecule(_CORE_CLAIM, pool, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(expansion.added_support_fact_ids, ())
        self.assertEqual(expansion.excluded_fact_ids, ("p_fact_002",))
        self.assertIn("unsure", expansion.exclusion_reasons[0])


class VerbosityPrefilterTest(unittest.TestCase):
    def test_within_budget_is_unchanged(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002",),
            stop_reason="pool_exhausted",
        )

        result = apply_verbosity_prefilter(_CORE_CLAIM, expansion, max_lines=5)

        self.assertEqual(result, expansion)

    def test_over_budget_removes_lowest_value_added_fact_first(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002", "p_fact_004", "p_fact_005"),
            stop_reason="max_additions_reached",
        )

        # max_lines=1 with a chars_per_line small enough that 3 additions
        # overflow but progressively removing the lowest-value ones fits.
        result = apply_verbosity_prefilter(_CORE_CLAIM, expansion, max_lines=1, chars_per_line=100)

        self.assertLess(len(result.added_support_fact_ids), len(expansion.added_support_fact_ids))
        # The LAST-added (lowest-ranked) fact is removed first.
        self.assertNotIn("p_fact_005", result.added_support_fact_ids)
        self.assertIn("p_fact_005", result.excluded_fact_ids)
        self.assertIn("verbosity_prefilter_removed_lowest_value", result.exclusion_reasons)

    def test_never_mutates_core_claim(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002", "p_fact_004", "p_fact_005"),
        )
        original_claim_text = _CORE_CLAIM.claim_text

        apply_verbosity_prefilter(_CORE_CLAIM, expansion, max_lines=1, chars_per_line=1)

        self.assertEqual(_CORE_CLAIM.claim_text, original_claim_text)

    def test_still_over_budget_at_zero_additions_flags_core_claim(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002",),
            stop_reason="pool_exhausted",
        )

        # chars_per_line=1 guarantees even the bare core claim text overflows.
        result = apply_verbosity_prefilter(_CORE_CLAIM, expansion, max_lines=1, chars_per_line=1)

        self.assertEqual(result.added_support_fact_ids, ())
        self.assertIn("core_claim_exceeds_line_budget", result.stop_reason)
        self.assertIn("pool_exhausted", result.stop_reason)

    def test_estimate_grows_with_added_fact_count(self) -> None:
        base = estimate_expanded_line_count(_CORE_CLAIM, 0)
        expanded = estimate_expanded_line_count(_CORE_CLAIM, 3)

        self.assertGreaterEqual(expanded, base)


class WriteExpandedClaimJsonTest(unittest.TestCase):
    def test_write_json(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002",),
            excluded_fact_ids=("p_fact_003",),
            exclusion_reasons=("keep_out",),
            stop_reason="pool_exhausted",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "expanded_claim_molecules.json"
            write_expanded_claim_molecules_json([expansion], path)

            with path.open() as handle:
                data = json.load(handle)

            self.assertEqual(data[0]["core_claim_id"], "p_claim_01")
            self.assertEqual(data[0]["added_support_fact_ids"], ["p_fact_002"])


if __name__ == "__main__":
    unittest.main()
