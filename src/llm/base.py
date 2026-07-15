"""
Abstract base class for LLM providers.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def call(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            json_mode: If True, request JSON output
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
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
    ) -> Dict[str, Any]:
        """
        Call the LLM and parse JSON response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON as dictionary
        """
        response = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return json.loads(response)
