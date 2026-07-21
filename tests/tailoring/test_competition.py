"""Deterministic tests for `tailoring.competition` (Phase 6, no API key needed).

`FakeLLMProvider` returns each queued response in call order (same
convention as `tests/tailoring/test_verification.py`'s fake) - a test
queues `_classify_primary_proof_overlap` responses in the exact order
`build_global_recommendation`'s greedy walk will request them.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from tailoring.competition import (
    build_global_recommendation,
    overlap_decisions_to_dicts,
    rank_local_candidates,
    slot_candidate_sets_to_dicts,
)
from tailoring.models import AnnotatedProposal, SlotCandidateSet, SlotTriageResult


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


def _overlap_verdict(verdict: str, dimension: str = "responsibility") -> Dict[str, Any]:
    # `dimension` is required (not Optional) because `_OVERLAP_JSON_SCHEMA`
    # always requires `primary_dimension` to be one of the 5 real
    # dimension strings, even for an `idk` verdict - keeping this fake
    # aligned with that contract makes it more likely to catch real
    # prompt/schema drift.
    return {"verdict": verdict, "primary_dimension": dimension, "reasoning": "test reasoning"}


def _proposal(
    proposal_id: str,
    project_id: str,
    core_claim_id: str,
    target_skills: tuple = (),
    supporting_fact_ids: tuple = (),
) -> AnnotatedProposal:
    return AnnotatedProposal(
        id=proposal_id,
        project_id=project_id,
        core_claim_id=core_claim_id,
        proposal_text=f"{proposal_id} text",
        supporting_fact_ids=supporting_fact_ids,
        target_skills=target_skills,
    )


class RankLocalCandidatesTest(unittest.TestCase):
    def test_eligible_bullets_excludes_protected_keep_labels(self):
        triage = [
            SlotTriageResult(bullet_id="b1", project_id="p", label="keep", reason=""),
            SlotTriageResult(bullet_id="b2", project_id="p", label="candidate_for_replacement", reason=""),
            SlotTriageResult(bullet_id="b3", project_id="p", label="deprioritize", reason=""),
            SlotTriageResult(bullet_id="b4", project_id="p", label="idk", reason=""),
        ]
        result = rank_local_candidates("p", triage, [], {}, [])
        self.assertEqual(result.eligible_original_bullet_ids, ("b2", "b3"))

    def test_ranks_all_proposals_without_pruning(self):
        proposals = [
            _proposal("prop_low", "p", "claim_low", target_skills=("x",), supporting_fact_ids=("f1",)),
            _proposal("prop_high", "p", "claim_high", target_skills=("x", "y", "z"), supporting_fact_ids=("f1", "f2", "f3")),
        ]
        proof_by_claim = {"claim_low": "Did a small thing.", "claim_high": "Did a much bigger, more detailed thing with real impact."}
        result = rank_local_candidates("p", [], proposals, proof_by_claim, ["x", "y", "z"])
        self.assertEqual(set(result.verified_proposal_ids), {"prop_low", "prop_high"})
        self.assertEqual(result.verified_proposal_ids[0], "prop_high")

    def test_local_duplicate_primary_proof_scores_lower(self):
        proposals = [
            _proposal("prop_a", "p", "claim_a", target_skills=("x",), supporting_fact_ids=("f1",)),
            _proposal("prop_b", "p", "claim_b", target_skills=("x",), supporting_fact_ids=("f1",)),
            _proposal("prop_c", "p", "claim_c", target_skills=("x",), supporting_fact_ids=("f1",)),
        ]
        # prop_a and prop_b share the exact same primary_proof (a local
        # duplicate); prop_c is distinct and otherwise identically scored.
        proof_by_claim = {
            "claim_a": "Reduced latency by 50%.",
            "claim_b": "Reduced latency by 50%.",
            "claim_c": "Improved cache hit rate by 30%.",
        }
        result = rank_local_candidates("p", [], proposals, proof_by_claim, ["x"])
        self.assertEqual(result.verified_proposal_ids[0], "prop_c")

    def test_empty_proposals_yields_explanatory_rationale_and_no_ids(self):
        result = rank_local_candidates("p", [], [], {}, [])
        self.assertEqual(result.verified_proposal_ids, ())
        self.assertIn("No verified proposals", result.ranking_rationale)


class BuildGlobalRecommendationTest(unittest.TestCase):
    def test_no_conflict_recommends_top_candidate_per_project(self):
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
        }
        proof_by_claim = {"claim_p1_a": "Did thing A.", "claim_p2_a": "Did thing B."}
        provider = FakeLLMProvider([_overlap_verdict("no")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        self.assertEqual({cs.project_id: cs.recommended_proposal_id for cs in updated}, {"p1": "p1_a", "p2": "p2_a"})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(warnings, [])

    def test_conflicting_top_candidates_only_one_recommended(self):
        # Both projects' TOP local candidate overlaps with the other -
        # round-robin priority means p1's candidate is accepted first, and
        # p2's conflicting candidate must be excluded from the
        # recommendation (but never removed from verified_proposal_ids).
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
        }
        proof_by_claim = {
            "claim_p1_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p2_a": "Reduced onboarding time via self-serve tooling.",
        }
        provider = FakeLLMProvider([_overlap_verdict("yes", "outcome")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated}
        self.assertEqual(recommended["p1"], "p1_a")
        self.assertIsNone(recommended["p2"])
        p2_set = next(cs for cs in updated if cs.project_id == "p2")
        self.assertEqual(p2_set.verified_proposal_ids, ("p2_a",))  # never pruned
        self.assertIn("p1_a", p2_set.recommendation_reason)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].verdict, "yes")
        self.assertEqual(decisions[0].primary_dimension, "outcome")
        # Only p1's proposal was ultimately recommended (p2's conflicting
        # candidate never got in), so there is nothing left to warn about.
        self.assertEqual(warnings, [])

    def test_proposal_id_missing_from_proposals_by_id_is_skipped_not_recommended(self):
        # A candidate_set can in principle reference a proposal_id that
        # proposals_by_id doesn't have an entry for (a data-integrity gap
        # upstream). This must never be silently treated as an
        # automatically-accepted, conflict-free recommendation - it
        # should be skipped, with zero overlap-classifier calls made for
        # it, and the reason should explain why nothing was recommended.
        candidate_sets = [SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_missing",))]
        provider = FakeLLMProvider([])  # no overlap call should ever be made

        updated, decisions, warnings = build_global_recommendation(candidate_sets, {}, {}, provider)

        self.assertIsNone(updated[0].recommended_proposal_id)
        self.assertIn("p1_missing", updated[0].recommendation_reason)
        self.assertEqual(decisions, [])
        self.assertEqual(provider.call_count, 0)
        self.assertEqual(warnings, [])

    def test_missing_proposal_falls_through_to_next_real_candidate(self):
        # Same data-integrity gap as above, but this project's SECOND
        # candidate is a real entry - it should still be recommended, and
        # since nothing was accepted yet when it's considered, no overlap
        # call is needed to accept it.
        candidate_sets = [SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_missing", "p1_real"))]
        proposals_by_id = {"p1_real": _proposal("p1_real", "p1", "claim_p1_real")}
        proof_by_claim = {"claim_p1_real": "Did a real, verifiable thing."}
        provider = FakeLLMProvider([])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        self.assertEqual(updated[0].recommended_proposal_id, "p1_real")
        self.assertEqual(decisions, [])
        self.assertEqual(provider.call_count, 0)
        self.assertEqual(warnings, [])

    def test_conflict_falls_through_to_project_second_candidate(self):
        # p2's FIRST candidate conflicts with p1's accepted pick, but its
        # SECOND candidate does not - p2 should still get a recommendation.
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a", "p2_b")),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
            "p2_b": _proposal("p2_b", "p2", "claim_p2_b"),
        }
        proof_by_claim = {
            "claim_p1_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p2_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p2_b": "Built a completely unrelated internal dashboard.",
        }
        provider = FakeLLMProvider([_overlap_verdict("yes"), _overlap_verdict("no")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated}
        self.assertEqual(recommended["p1"], "p1_a")
        self.assertEqual(recommended["p2"], "p2_b")
        self.assertEqual(len(decisions), 2)
        self.assertEqual(warnings, [])

    def test_idk_verdict_is_treated_as_non_overlapping(self):
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
        }
        proof_by_claim = {"claim_p1_a": "Ambiguous proof A.", "claim_p2_a": "Ambiguous proof B."}
        provider = FakeLLMProvider([_overlap_verdict("idk", "evidence_type")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated}
        self.assertEqual(recommended, {"p1": "p1_a", "p2": "p2_a"})
        self.assertEqual(decisions[0].verdict, "idk")
        # "Ambiguous proof A." and "Ambiguous proof B." are not identical
        # text, so the deterministic sanity net has nothing to flag even
        # though the LLM itself could not decide.
        self.assertEqual(warnings, [])

    def test_project_with_no_verified_proposals_gets_explanatory_reason(self):
        candidate_sets = [SlotCandidateSet(project_id="p1", verified_proposal_ids=())]
        updated, decisions, warnings = build_global_recommendation(candidate_sets, {}, {}, FakeLLMProvider([]))

        self.assertIsNone(updated[0].recommended_proposal_id)
        self.assertIn("No verified proposals", updated[0].recommendation_reason)
        self.assertEqual(decisions, [])
        self.assertEqual(warnings, [])
        self.assertEqual(FakeLLMProvider([]).call_count, 0)

    def test_never_prunes_eligible_original_bullet_ids(self):
        candidate_sets = [
            SlotCandidateSet(
                project_id="p1",
                eligible_original_bullet_ids=("b1", "b2"),
                verified_proposal_ids=("p1_a",),
            ),
        ]
        proposals_by_id = {"p1_a": _proposal("p1_a", "p1", "claim_p1_a")}
        proof_by_claim = {"claim_p1_a": "Did a thing."}
        updated, _, _ = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, FakeLLMProvider([_overlap_verdict("no")]))

        self.assertEqual(updated[0].eligible_original_bullet_ids, ("b1", "b2"))

    def test_three_projects_two_unique_points_third_gets_no_recommendation(self):
        # n=3 projects, but only m=2 genuinely distinct underlying
        # accomplishments: p1 and p2's only candidates are the SAME
        # accomplishment, p3's is genuinely different. No special-case
        # handling should be needed - p3 should still be recommended (it
        # never conflicts with anything), while whichever of p1/p2 loses
        # the round-robin priority tie should end up with NO
        # recommendation (never a crash, never a forced/incorrect pick).
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
            SlotCandidateSet(project_id="p3", verified_proposal_ids=("p3_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
            "p3_a": _proposal("p3_a", "p3", "claim_p3_a"),
        }
        proof_by_claim = {
            "claim_p1_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p2_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p3_a": "Migrated the billing service to a new provider.",
        }
        # Priority order is p1_a, p2_a, p3_a (round-robin position 0).
        # p1_a is accepted with no prior candidates to conflict with.
        # p2_a is checked against p1_a -> conflict (yes).
        # p3_a is checked against p1_a -> no conflict.
        provider = FakeLLMProvider([_overlap_verdict("yes", "outcome"), _overlap_verdict("no")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated}
        self.assertEqual(recommended["p1"], "p1_a")
        self.assertIsNone(recommended["p2"])
        self.assertEqual(recommended["p3"], "p3_a")
        # p2's own verified_proposal_ids must still be intact (advisory
        # only - it remains a selectable option even with no recommendation).
        p2_set = next(cs for cs in updated if cs.project_id == "p2")
        self.assertEqual(p2_set.verified_proposal_ids, ("p2_a",))
        # No exact-text duplicates among what actually got RECOMMENDED
        # (p1_a and p3_a), so the sanity net has nothing to add here -
        # the classifier already correctly excluded p2_a.
        self.assertEqual(warnings, [])


class DuplicateRecommendationWarningTest(unittest.TestCase):
    """Tests for the deterministic exact-text sanity net that runs after
    the LLM-judged greedy filter, covering the case where an n-projects-
    vs-m<n-unique-points scenario slips past `_classify_primary_proof_
    overlap` (e.g. an unexpected `no`/`idk` verdict for identical text).
    """

    def test_identical_recommended_proof_text_produces_a_warning(self):
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
        }
        # Same primary_proof text for both, but the fake classifier
        # (standing in for a real misclassification) says "no" - the
        # deterministic sanity net must still catch this.
        proof_by_claim = {
            "claim_p1_a": "Reduced onboarding time via self-serve tooling.",
            "claim_p2_a": "Reduced onboarding time via self-serve tooling.",
        }
        provider = FakeLLMProvider([_overlap_verdict("no")])

        updated, decisions, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        recommended = {cs.project_id: cs.recommended_proposal_id for cs in updated}
        self.assertEqual(recommended, {"p1": "p1_a", "p2": "p2_a"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("p1_a", warnings[0])
        self.assertIn("p2_a", warnings[0])

    def test_no_warning_when_recommended_proofs_are_distinct(self):
        candidate_sets = [
            SlotCandidateSet(project_id="p1", verified_proposal_ids=("p1_a",)),
            SlotCandidateSet(project_id="p2", verified_proposal_ids=("p2_a",)),
        ]
        proposals_by_id = {
            "p1_a": _proposal("p1_a", "p1", "claim_p1_a"),
            "p2_a": _proposal("p2_a", "p2", "claim_p2_a"),
        }
        proof_by_claim = {"claim_p1_a": "Did thing A.", "claim_p2_a": "Did thing B."}
        provider = FakeLLMProvider([_overlap_verdict("no")])

        _, _, warnings = build_global_recommendation(candidate_sets, proposals_by_id, proof_by_claim, provider)

        self.assertEqual(warnings, [])


class DictConversionTest(unittest.TestCase):
    def test_slot_candidate_sets_to_dicts_round_trips_fields(self):
        candidate_set = SlotCandidateSet(
            project_id="p1",
            eligible_original_bullet_ids=("b1",),
            verified_proposal_ids=("p1_a",),
            ranking_rationale="rationale text",
            recommended_proposal_id="p1_a",
            recommendation_reason="reason text",
        )
        [as_dict] = slot_candidate_sets_to_dicts([candidate_set])
        self.assertEqual(as_dict["project_id"], "p1")
        self.assertEqual(as_dict["eligible_original_bullet_ids"], ["b1"])
        self.assertEqual(as_dict["recommended_proposal_id"], "p1_a")

    def test_overlap_decisions_to_dicts_round_trips_fields(self):
        from tailoring.models import ProofOverlapDecision

        decision = ProofOverlapDecision(
            proposal_id_a="a", proposal_id_b="b", verdict="yes", primary_dimension="outcome", reasoning="why"
        )
        [as_dict] = overlap_decisions_to_dicts([decision])
        self.assertEqual(as_dict, {"proposal_id_a": "a", "proposal_id_b": "b", "verdict": "yes", "primary_dimension": "outcome", "reasoning": "why"})


if __name__ == "__main__":
    unittest.main()
