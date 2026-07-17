"""
Anthropic Claude API provider.
"""

import os
from typing import Any, Dict, List, Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (default: reads from ANTHROPIC_API_KEY env var)
            model: Model name (default: claude-3-sonnet)
        """
        super().__init__()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY env var not set")
        self.model = model
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
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
        # Claude doesn't have native strict structured outputs or JSON mode
        # here, so json_schema is accepted for interface compatibility but
        # ignored; JSON is requested via prompt instruction instead.
        user_prompt = prompt
        if json_mode:
            user_prompt = f"{prompt}\n\nRespond with valid JSON only."
        
        messages = list(few_shot_messages) if few_shot_messages else []
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        )
        return response.content[0].text
