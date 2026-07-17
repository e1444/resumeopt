"""
OpenAI API provider.
"""

import os
from typing import Any, Dict, List, Optional

from .base import LLMProvider

# Reasoning-tier model families (gpt-5.x, o1/o3/o4) use a different Chat
# Completions contract than gpt-4.x: `max_completion_tokens` instead of
# `max_tokens`, and only the default temperature (1) is accepted - a custom
# `temperature` value is rejected outright with a 400 error. They also spend
# a substantial, variable amount of that token budget on hidden internal
# reasoning tokens before producing any visible output (observed: a trivial
# `{"ok": true}` JSON response consumed 256 reasoning tokens against a 273
# total), so a `max_tokens` value tuned for gpt-4.x can silently truncate the
# visible response to nothing. Detected 2026-07-16 while evaluating
# `gpt-5-nano` as a candidate classifier model (see docs/agent/DEV_PLAN.md).
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_MODEL_PREFIXES)


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
        few_shot_messages: Optional[List[Dict[str, str]]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if few_shot_messages:
            messages.extend(few_shot_messages)
        
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if _is_reasoning_model(self.model):
            # See `_REASONING_MODEL_PREFIXES`'s module-level docstring: these
            # models reject a custom `temperature` and use a different token-
            # budget parameter, one that must also absorb hidden reasoning
            # tokens on top of the visible response - pad generously so a
            # `max_tokens` value tuned for gpt-4.x doesn't truncate the
            # visible output to empty.
            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        
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
            reasoning_tokens = 0
            completion_tokens_details = getattr(usage, "completion_tokens_details", None)
            if completion_tokens_details is not None:
                reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", 0) or 0
            self._record_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cached_prompt_tokens=cached_prompt_tokens,
                reasoning_tokens=reasoning_tokens,
                call_label=(json_schema or {}).get("name", ""),
            )

        return response.choices[0].message.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the OpenAI embeddings API."""

        if not texts:
            return []

        response = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

