"""
Local Ollama provider.
"""

from typing import Any, Dict, Optional

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (default: mistral)
            base_url: Ollama server URL
        """
        super().__init__()
        self.model = model
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package not installed. Install with: pip install requests")
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Ollama models generally don't support strict structured outputs the
        # way OpenAI does, so json_schema is accepted for interface
        # compatibility but ignored; JSON is requested via prompt instruction.
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = prompt
        if json_mode:
            user_content = f"{prompt}\n\nRespond with valid JSON only."
        
        messages.append({"role": "user", "content": user_content})
        
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
