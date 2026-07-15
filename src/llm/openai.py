"""
OpenAI API provider.
"""

import os
from typing import Any, Dict, List, Optional

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (default: reads from OPENAI_API_KEY env var)
            model: Model name (default: gpt-4o)
            embedding_model: Model used by embed() for semantic-similarity matching
        """
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        self.model = model
        self.embedding_model = embedding_model
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_schema is not None:
            # Structured outputs: the API enforces the schema server-side, so
            # responses can't come back with a malformed/unexpected shape the
            # way prompt-instructed JSON sometimes did.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "schema": json_schema.get("schema", json_schema),
                    "strict": True,
                },
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            cached_prompt_tokens = 0
            prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_tokens_details is not None:
                cached_prompt_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
            self._record_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cached_prompt_tokens=cached_prompt_tokens,
            )

        return response.choices[0].message.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the OpenAI embeddings API."""

        if not texts:
            return []

        response = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

