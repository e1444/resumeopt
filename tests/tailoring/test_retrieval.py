"""Deterministic tests for `tailoring.retrieval` (Phase 2, no API key needed).

Uses a fake deterministic embedding provider (same convention as
`tests/matcher/test_matching.py`'s `FakeEmbeddingProvider`) for the
semantic-tier test case - no real embeddings API call.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from tailoring.loaders import load_fact_atoms
from tailoring.models import JobRequirements
from tailoring.retrieval import (
    retrieve_project_fact_pool,
    target_skills_from_requirements,
    write_project_fact_matches_json,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "evals" / "tailoring" / "fact_retrieval"


class FakeEmbeddingProvider(LLMProvider):
    """Deterministic fake embeddings - "machine learning" and "ml ranking" are
    near-identical vectors (a genuine semantic near-match); everything else
    is orthogonal/unrelated.
    """

    _KNOWN_VECTORS: Dict[str, List[float]] = {
        "machine learning": [1.0, 0.0, 0.0],
        "ml ranking": [0.95, 0.05, 0.0],
        "python": [0.0, 1.0, 0.0],
        "django": [0.0, 0.95, 0.0],
        "project management": [0.0, 0.0, 1.0],
        "baking": [0.0, 0.0, 0.0],
    }
    _UNKNOWN_VECTOR = [0.0, 0.0, 0.0]

    def call(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def call_json(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._KNOWN_VECTORS.get(text.strip().lower(), list(self._UNKNOWN_VECTOR)) for text in texts]


class RetrievalFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = yaml.safe_load((FIXTURES_DIR / "expected_outcome.yaml").read_text(encoding="utf-8"))
        alpha_atoms = load_fact_atoms(FIXTURES_DIR / "project_alpha_fact_atoms.yaml")
        beta_atoms = load_fact_atoms(FIXTURES_DIR / "project_beta_fact_atoms.yaml")
        self.fact_atoms_by_project = {"project_alpha": alpha_atoms, "project_beta": beta_atoms}
        self.protected_fact_ids = set(self.expected["protected_fact_ids"])
        self.target_skills = self.expected["target_skills"]

    def test_exact_tier_match_is_included(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=None,
        )
        matches_by_id = {match.fact_id: match for match in matches}

        alpha_1 = matches_by_id["alpha_fact_001"]
        self.assertTrue(alpha_1.included)
        self.assertEqual(alpha_1.match_tier, "alias")
        self.assertEqual(alpha_1.project_id, "project_alpha")

    def test_protected_fact_is_recorded_but_excluded(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=None,
        )
        matches_by_id = {match.fact_id: match for match in matches}

        alpha_2 = matches_by_id["alpha_fact_002"]
        self.assertFalse(alpha_2.included)
        self.assertEqual(alpha_2.exclusion_reason, "protected_by_baseline_bullet")
        # Still recorded WITH its real match tier/score, for auditability.
        self.assertEqual(alpha_2.match_tier, "alias")

    def test_unrelated_fact_is_absent_not_recorded(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=None,
        )
        fact_ids = {match.fact_id for match in matches}

        self.assertNotIn("alpha_fact_003", fact_ids)

    def test_cross_project_fact_never_returned(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=None,
        )
        fact_ids = {match.fact_id for match in matches}

        self.assertNotIn("beta_fact_001", fact_ids)
        self.assertTrue(all(match.project_id == "project_alpha" for match in matches))

    def test_semantic_tier_catches_exact_tier_miss(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=FakeEmbeddingProvider(),
        )
        matches_by_id = {match.fact_id: match for match in matches}

        alpha_4 = matches_by_id["alpha_fact_004"]
        self.assertTrue(alpha_4.included)
        self.assertEqual(alpha_4.match_tier, "semantic")
        self.assertEqual(alpha_4.matched_target_skill, "machine learning")

    def test_no_semantic_provider_means_no_semantic_matches(self) -> None:
        matches = retrieve_project_fact_pool(
            "project_alpha",
            self.fact_atoms_by_project,
            self.protected_fact_ids,
            self.target_skills,
            llm_provider=None,
        )
        fact_ids = {match.fact_id for match in matches}

        self.assertNotIn("alpha_fact_004", fact_ids)


class MaxPoolSizeTest(unittest.TestCase):
    def test_facts_beyond_the_cap_are_recorded_but_excluded(self) -> None:
        from tailoring.models import FactAtom

        atoms = [
            FactAtom(id=f"p_fact_{i:03d}", fact=f"Did thing {i}.", skill_tags=("python",)) for i in range(5)
        ]
        matches = retrieve_project_fact_pool(
            "p",
            {"p": atoms},
            protected_fact_ids=set(),
            target_skills=["python"],
            llm_provider=None,
            max_pool_size=3,
        )

        included = [match for match in matches if match.included]
        excluded = [match for match in matches if not match.included]
        self.assertEqual(len(included), 3)
        self.assertEqual(len(excluded), 2)
        for match in excluded:
            self.assertEqual(match.exclusion_reason, "pool_capped")


class TargetSkillsFromRequirementsTest(unittest.TestCase):
    def test_combines_matched_and_missing_skills_deduped(self) -> None:
        requirements = JobRequirements(
            role_title="Role",
            seniority="mid",
            industry_domain="tech",
            core_requirements=(),
            nice_to_have=(),
            summary_paragraph="",
            matched_skills=({"canonical_name": "Python", "raw_term": "python"},),
            missing_skills=("Hydra", "python"),
        )

        skills = target_skills_from_requirements(requirements)

        self.assertEqual(skills, ["Python", "Hydra"])


class WriteProjectFactMatchesJsonTest(unittest.TestCase):
    def test_writes_every_match_as_a_dict(self) -> None:
        from tailoring.models import FactAtom

        atoms = [FactAtom(id="p_fact_001", fact="Did a thing.", skill_tags=("python",))]
        matches = retrieve_project_fact_pool(
            "p", {"p": atoms}, protected_fact_ids=set(), target_skills=["python"], llm_provider=None
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "project_fact_matches.json"
            write_project_fact_matches_json(matches, path)

            with path.open() as handle:
                data = json.load(handle)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["fact_id"], "p_fact_001")
            self.assertTrue(data[0]["included"])


if __name__ == "__main__":
    unittest.main()
