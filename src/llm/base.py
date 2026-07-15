"""
Abstract base class for LLM providers.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


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
    ) -> Dict[str, Any]:
        """
        Call the LLM and parse JSON response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            json_schema: Optional strict structured-output contract; see `call`.
            
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
        )
        return json.loads(response)

