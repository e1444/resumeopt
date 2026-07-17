"""
Abstract base class for LLM providers.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self) -> None:
        # Real, provider-reported token usage, accumulated across every call
        # made through this instance. Populated only by providers that expose
        # authoritative usage data (currently OpenAI); left at zero otherwise,
        # with `usage_available` indicating whether the numbers are real.
        self.usage_available = False
        self.usage_totals: Dict[str, int] = {
            "call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        }

    def _record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_prompt_tokens: int = 0,
    ) -> None:
        """Accumulate real, provider-reported token usage for one call."""

        self.usage_available = True
        self.usage_totals["call_count"] += 1
        self.usage_totals["prompt_tokens"] += prompt_tokens
        self.usage_totals["completion_tokens"] += completion_tokens
        self.usage_totals["total_tokens"] += total_tokens
        self.usage_totals["cached_prompt_tokens"] += cached_prompt_tokens

    @abstractmethod
    def call(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
        few_shot_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            json_mode: If True, request JSON output
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            json_schema: Optional {"name": ..., "schema": {...}} contract for
                providers that support strict structured outputs (currently
                OpenAI). Providers that don't support it should ignore it and
                fall back to json_mode.
            few_shot_messages: Optional list of {"role": "user"/"assistant",
                "content": ...} message dicts inserted between the system
                message and the final user `prompt` - real conversation-turn
                few-shot examples (an example input + the exact expected
                output), rather than examples merely described in prose within
                `prompt` itself. Added 2026-07-16 specifically to demonstrate
                correct binding between a classifier's free-text `reason` and
                its boolean `excluded` value (the self-contradiction failure
                mode found in src/parser/parallel_extraction.py) - showing the
                model one real worked example response is a stronger signal
                than describing the same example in words. Providers that
                don't support multi-turn history should ignore this and fall
                back to a single combined prompt.
            
        Returns:
            Response text from the LLM
        """
        pass
    
    def call_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
        few_shot_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Call the LLM and parse JSON response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            json_schema: Optional strict structured-output contract; see `call`.
            few_shot_messages: Optional conversation-turn few-shot examples; see `call`.
            
        Returns:
            Parsed JSON as dictionary
        """
        response = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            few_shot_messages=few_shot_messages,
        )
        return json.loads(response)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors for semantic similarity matching.

        Providers that don't support embeddings should leave this as-is; it
        raises NotImplementedError so callers (e.g. SemanticMatcher) can
        detect unsupported providers and fall back gracefully.
        """

        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

