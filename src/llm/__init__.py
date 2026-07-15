"""
LLM module with providers for different APIs and local models.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables as-is

from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "get_llm_provider",
]


def get_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider.
    
    Args:
        provider_type: "openai", "anthropic", or "ollama"
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Choose from {list(providers.keys())}")
    
    return providers[provider_type](**kwargs)
