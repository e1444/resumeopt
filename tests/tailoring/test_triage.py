"""Deterministic tests for `tailoring.triage` (Phase 1, no API key needed).

`FakeLLMProvider` mirrors the real `_TRIAGE_JSON_SCHEMA`/prompt shape (per
this project's established convention - see e.g. `tests/parser/
test_pipeline.py`'s own `FakeLLMProvider`) rather than stubbing out
`triage_bullet` itself, so a schema/prompt drift would actually be caught by
these tests.
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
from tailoring.models import BaselineBullet, JobRequirements, ProjectBaseline
from tailoring.triage import triage_bullet, triage_project_bullets, write_slot_triage_json


class FakeLLMProvider(LLMProvider):
    """`verdict_by_bullet_text: {bullet_text: verdict_dict}` - canned
    `_TRIAGE_JSON_SCHEMA`-shaped response keyed by the bullet's own text
    (each call's prompt embeds the exact bullet text, so this is enough to
    distinguish calls without parsing the whole prompt). A bullet text NOT
    present raises, so a test omitting a fixture entry fails loudly instead
    of silently returning some default verdict.
    """

    def __init__(self, verdict_by_bullet_text: Dict[str, Dict[str, Any]]):
        super().__init__()
        self._verdict_by_bullet_text = verdict_by_bullet_text

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
        assert json_schema is not None and json_schema["name"] == "slot_triage_verdict"
        for bullet_text, verdict in self._verdict_by_bullet_text.items():
            if bullet_text in prompt:
                return json.dumps(verdict)
        raise AssertionError(f"no canned verdict for prompt: {prompt!r}")


_PROJECT = ProjectBaseline(
    project_id="demo_project",
    project_title="Demo Project",
    role_context="Fixture Project",
    dates="2020",
    resume_section="PROJECTS",
    bullets=(),
)

_REQUIREMENTS = JobRequirements(
    role_title="ML Research Engineer",
    seniority="mid",
    industry_domain="machine learning research",
    core_requirements=("generative modeling", "constrained optimization"),
    nice_to_have=("Hydra", "Weights & Biases"),
    summary_paragraph="Builds generative models for structured prediction.",
)


def _bullet(text: str, bullet_id: str = "demo_project_b1", position: str = "start") -> BaselineBullet:
    return BaselineBullet(
        id=bullet_id, project_id="demo_project", order=0, text=text, position=position
    )


class TriageBulletLlmBackedTest(unittest.TestCase):
    def test_keep_label_round_trips_all_fields(self) -> None:
        bullet = _bullet("Developed a flow-based generative classifier.")
        provider = FakeLLMProvider(
            {
                bullet.text: {
                    "label": "keep",
                    "job_relevance": "Directly matches generative modeling requirement.",
                    "narrative_value": "Strong headline accomplishment.",
                    "replacement_opportunity": "None needed.",
                    "reason": "Core requirement match.",
                }
            }
        )

        result = triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=provider)

        self.assertEqual(result.bullet_id, bullet.id)
        self.assertEqual(result.project_id, "demo_project")
        self.assertEqual(result.label, "keep")
        self.assertEqual(result.job_relevance, "Directly matches generative modeling requirement.")
        self.assertEqual(result.reason, "Core requirement match.")

    def test_invalid_label_falls_back_to_idk(self) -> None:
        bullet = _bullet("Some unrelated bullet.")
        provider = FakeLLMProvider(
            {bullet.text: {"label": "not_a_real_label", "job_relevance": "", "narrative_value": "", "replacement_opportunity": "", "reason": ""}}
        )

        result = triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=provider)

        self.assertEqual(result.label, "idk")

    def test_schema_name_is_slot_triage_verdict(self) -> None:
        # Enforced by FakeLLMProvider's assert - this test just confirms the
        # call succeeds at all with the real production schema in place.
        bullet = _bullet("Achieved SOTA generative quality.")
        provider = FakeLLMProvider(
            {bullet.text: {"label": "keep", "job_relevance": "x", "narrative_value": "x", "replacement_opportunity": "x", "reason": "x"}}
        )

        triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=provider)


class DeterministicFallbackTest(unittest.TestCase):
    def test_no_provider_uses_deterministic_heuristic(self) -> None:
        bullet = _bullet("Developed a flow-based generative classifier using constrained optimization.")

        result = triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=None)

        self.assertIn(result.label, ("keep", "candidate_for_replacement", "idk"))
        self.assertIn("deterministic fallback", result.reason)

    def test_zero_overlap_bullet_is_candidate_for_replacement(self) -> None:
        bullet = _bullet("Baked bread for a community fundraiser.")

        result = triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=None)

        self.assertEqual(result.label, "candidate_for_replacement")


class TriageProjectBulletsTest(unittest.TestCase):
    def test_returns_one_result_per_bullet_in_order(self) -> None:
        bullets = (
            _bullet("First bullet.", bullet_id="demo_project_b1", position="start"),
            _bullet("Second bullet.", bullet_id="demo_project_b2", position="end"),
        )
        project = ProjectBaseline(
            project_id="demo_project",
            project_title="Demo Project",
            role_context="Fixture Project",
            dates="2020",
            resume_section="PROJECTS",
            bullets=bullets,
        )
        provider = FakeLLMProvider(
            {
                "First bullet.": {
                    "label": "keep",
                    "job_relevance": "x",
                    "narrative_value": "x",
                    "replacement_opportunity": "x",
                    "reason": "x",
                },
                "Second bullet.": {
                    "label": "candidate_for_replacement",
                    "job_relevance": "x",
                    "narrative_value": "x",
                    "replacement_opportunity": "x",
                    "reason": "x",
                },
            }
        )

        results = triage_project_bullets(project, _REQUIREMENTS, llm_provider=provider)

        self.assertEqual([r.bullet_id for r in results], ["demo_project_b1", "demo_project_b2"])
        self.assertEqual([r.label for r in results], ["keep", "candidate_for_replacement"])


class WriteSlotTriageJsonTest(unittest.TestCase):
    def test_writes_every_result_as_a_dict(self) -> None:
        bullet = _bullet("Developed a flow-based generative classifier.")
        provider = FakeLLMProvider(
            {bullet.text: {"label": "keep", "job_relevance": "x", "narrative_value": "x", "replacement_opportunity": "x", "reason": "x"}}
        )
        result = triage_bullet(bullet, _PROJECT, _REQUIREMENTS, llm_provider=provider)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slot_triage.json"
            write_slot_triage_json([result], path)

            with path.open() as handle:
                data = json.load(handle)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["bullet_id"], bullet.id)
            self.assertEqual(data[0]["label"], "keep")


if __name__ == "__main__":
    unittest.main()
