"""Deterministic tests for `tailoring.verification` (Phase 5, no API key needed).

`FakeLLMProvider` returns each queued response in call order (same
convention as `tests/tailoring/test_expansion.py`'s fake), across all three
call kinds this module makes (`synthesize_proposal`'s single synthesis
call, `verify_proposal`'s up-to-4 classifier calls, `repair_proposal`'s
repair-text calls) - so a test queues responses in the exact order the
production code will request them.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from tailoring.models import (
    AnnotatedProposal,
    BaselineBullet,
    CoreClaimMolecule,
    ExpandedClaimMolecule,
    FactAtom,
)
from tailoring.verification import (
    repair_proposal,
    synthesize_proposal,
    verify_proposal,
)


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
        response = self._responses[self.call_count]
        self.call_count += 1
        return json.dumps(response)


def _verdict(verdict: str) -> Dict[str, Any]:
    return {"verdict": verdict, "reasoning": "test reasoning"}


_CORE_CLAIM = CoreClaimMolecule(
    id="p_claim_01",
    project_id="p",
    claim_text="Built a document-indexing service.",
    supporting_fact_ids=("p_fact_001",),
    target_skills=("backend", "search"),
    primary_proof="document-indexing service",
    rationale="One backend deliverable.",
)

_FACT_ATOMS_BY_ID = {
    "p_fact_001": FactAtom(id="p_fact_001", fact="Built a document-indexing service."),
    "p_fact_002": FactAtom(
        id="p_fact_002", fact="Reduced average query latency from 300ms to 90ms."
    ),
}

_PROPOSAL = AnnotatedProposal(
    id="p_claim_01_proposal",
    project_id="p",
    core_claim_id="p_claim_01",
    proposal_text="Built a document-indexing service, reducing average query latency from 300ms to 90ms.",
    supporting_fact_ids=("p_fact_001", "p_fact_002"),
    target_skills=("backend", "search"),
)

_PROTECTED_BASELINE_BULLETS = [
    BaselineBullet(
        id="p_bullet_kept_01",
        project_id="p",
        order=0,
        text="Migrated the billing service's database to a managed cloud provider.",
        position="start",
        fact_ids=("p_fact_protected_01",),
    )
]


class SynthesizeProposalTest(unittest.TestCase):
    def test_builds_proposal_with_union_fact_ids(self) -> None:
        expansion = ExpandedClaimMolecule(
            core_claim_id="p_claim_01",
            project_id="p",
            added_support_fact_ids=("p_fact_002",),
        )
        provider = FakeLLMProvider(
            [{"proposal_text": "Built a document-indexing service, reducing latency.", "reasoning": "x"}]
        )

        proposal = synthesize_proposal(_CORE_CLAIM, expansion, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(proposal.supporting_fact_ids, ("p_fact_001", "p_fact_002"))
        self.assertEqual(provider.call_count, 1)

    def test_no_expansion_keeps_core_claim_facts_only(self) -> None:
        provider = FakeLLMProvider([{"proposal_text": "Built a document-indexing service.", "reasoning": "x"}])

        proposal = synthesize_proposal(_CORE_CLAIM, None, _FACT_ATOMS_BY_ID, provider)

        self.assertEqual(proposal.supporting_fact_ids, ("p_fact_001",))


class VerifyProposalTest(unittest.TestCase):
    def test_protected_fact_reuse_is_unresolvable_with_zero_llm_calls(self) -> None:
        proposal = AnnotatedProposal(
            id="x_proposal",
            project_id="p",
            core_claim_id="x",
            proposal_text="Reused a protected fact directly.",
            supporting_fact_ids=("p_fact_protected_01",),
        )
        provider = FakeLLMProvider([])

        result = verify_proposal(
            proposal,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids={"p_fact_protected_01"},
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_type, "unresolvable")
        self.assertEqual(provider.call_count, 0)

    def test_all_classifiers_pass_yields_pass_with_final_text(self) -> None:
        provider = FakeLLMProvider([_verdict("no"), _verdict("no"), _verdict("no"), _verdict("no")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.final_text, _PROPOSAL.proposal_text)
        self.assertEqual(provider.call_count, 4)

    def test_fact_support_failure_is_hallucination_and_short_circuits(self) -> None:
        provider = FakeLLMProvider([_verdict("yes")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_type, "hallucination")
        self.assertEqual(provider.call_count, 1)

    def test_same_claim_integrity_failure_is_bad_flow(self) -> None:
        provider = FakeLLMProvider([_verdict("no"), _verdict("yes")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_type, "bad_flow")
        self.assertEqual(provider.call_count, 2)

    def test_semantic_duplication_failure_is_bad_wording(self) -> None:
        provider = FakeLLMProvider([_verdict("no"), _verdict("no"), _verdict("yes")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_type, "bad_wording")
        self.assertEqual(provider.call_count, 3)

    def test_project_relevance_failure_is_bad_wording(self) -> None:
        provider = FakeLLMProvider([_verdict("no"), _verdict("no"), _verdict("no"), _verdict("yes")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.failure_type, "bad_wording")
        self.assertEqual(provider.call_count, 4)

    def test_idk_verdict_with_no_hard_failure_yields_idk(self) -> None:
        provider = FakeLLMProvider([_verdict("no"), _verdict("no"), _verdict("no"), _verdict("idk")])

        result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=(),
            llm_provider=provider,
        )

        self.assertEqual(result.status, "idk")
        self.assertIsNone(result.failure_type)
        self.assertEqual(provider.call_count, 4)


class RepairProposalTest(unittest.TestCase):
    def test_successful_hallucination_repair_reaches_pass(self) -> None:
        hallucination_result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=FakeLLMProvider([_verdict("yes")]),
        )
        self.assertEqual(hallucination_result.failure_type, "hallucination")

        provider = FakeLLMProvider(
            [
                {"repaired_text": "Built a document-indexing service.", "reasoning": "x"},
                _verdict("no"),
                _verdict("no"),
                _verdict("no"),
                _verdict("no"),
            ]
        )

        new_proposal, final_result = repair_proposal(
            _PROPOSAL,
            hallucination_result,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(final_result.status, "pass")
        self.assertEqual(len(final_result.repair_steps), 1)
        self.assertEqual(final_result.repair_steps[0].repair_type, "hallucination")
        self.assertEqual(final_result.final_text, new_proposal.proposal_text)

    def test_failed_repair_discards_after_one_attempt_per_type(self) -> None:
        initial_result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=FakeLLMProvider([_verdict("yes")]),
        )
        self.assertEqual(initial_result.failure_type, "hallucination")

        # Repair attempt 1 (hallucination) "fixes" the text but reverify still
        # finds bad_flow. Repair attempt 2 (bad_flow) still finds bad_flow
        # again -> bad_flow already attempted -> loop stops, discarded.
        provider = FakeLLMProvider(
            [
                {"repaired_text": "attempt one", "reasoning": "x"},
                _verdict("no"),
                _verdict("yes"),
                {"repaired_text": "attempt two", "reasoning": "x"},
                _verdict("no"),
                _verdict("yes"),
            ]
        )

        _, final_result = repair_proposal(
            _PROPOSAL,
            initial_result,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(final_result.status, "fail")
        self.assertEqual(final_result.failure_type, "bad_flow")
        self.assertEqual(len(final_result.repair_steps), 2)
        self.assertEqual(provider.call_count, 6)

    def test_unresolvable_is_never_attempted(self) -> None:
        unresolvable_result = verify_proposal(
            AnnotatedProposal(
                id="x_proposal",
                project_id="p",
                core_claim_id="x",
                proposal_text="Reused a protected fact directly.",
                supporting_fact_ids=("p_fact_protected_01",),
            ),
            _FACT_ATOMS_BY_ID,
            protected_fact_ids={"p_fact_protected_01"},
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=FakeLLMProvider([]),
        )
        provider = FakeLLMProvider([])

        _, final_result = repair_proposal(
            _PROPOSAL,
            unresolvable_result,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids={"p_fact_protected_01"},
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=("backend",),
            llm_provider=provider,
        )

        self.assertEqual(final_result.status, "fail")
        self.assertEqual(final_result.failure_type, "unresolvable")
        self.assertEqual(len(final_result.repair_steps), 0)
        self.assertEqual(provider.call_count, 0)

    def test_idk_is_never_attempted(self) -> None:
        idk_result = verify_proposal(
            _PROPOSAL,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=(),
            llm_provider=FakeLLMProvider([_verdict("no"), _verdict("no"), _verdict("no"), _verdict("idk")]),
        )
        self.assertEqual(idk_result.status, "idk")
        provider = FakeLLMProvider([])

        _, final_result = repair_proposal(
            _PROPOSAL,
            idk_result,
            _FACT_ATOMS_BY_ID,
            protected_fact_ids=set(),
            protected_baseline_bullets=_PROTECTED_BASELINE_BULLETS,
            target_skills=(),
            llm_provider=provider,
        )

        self.assertEqual(final_result.status, "idk")
        self.assertEqual(len(final_result.repair_steps), 0)
        self.assertEqual(provider.call_count, 0)


if __name__ == "__main__":
    unittest.main()
