"""LLM-based grounding confirmation for borderline skill matches.

Not a `Matcher` in the match-producing sense (base.py): this validates
whether an already-selected canonical match is actually supported by the
posting text, for edge cases deterministic substring checks can't resolve
(e.g. ipynb implying Jupyter usage).
"""

from __future__ import annotations

from typing import Sequence

from llm import LLMProvider
from llm.schemas import GROUNDING_JSON_SCHEMA


class LLMGroundingMatcher:
    """LLM-based grounding confirmation for borderline skill matches."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def confirm_grounding(
        self,
        posting_text: str,
        canonical_name: str,
        aliases: Sequence[str],
        raw_term: str,
        evidence: str,
    ) -> bool:
        prompt = (
            "Determine whether the skill is actually supported by the posting text. "
            f"Posting Text:\n{posting_text}\n\n"
            f"Skill Canonical Name: {canonical_name}\n"
            f"Skill Aliases: {list(aliases)}\n"
            f"Parser Raw Term: {raw_term}\n"
            f"Parser Evidence: {evidence}\n"
            "Consider alias edge cases such as ipynb indicating Jupyter."
        )

        try:
            payload = self.llm.call_json(
                prompt=prompt,
                system_prompt=(
                    "You validate grounding for resume skills. "
                    "Return valid JSON only and do not infer unsupported skills."
                ),
                temperature=0.1,
                max_tokens=300,
                json_schema=GROUNDING_JSON_SCHEMA,
            )
        except Exception:
            return False

        return bool(payload.get("is_grounded", False))
