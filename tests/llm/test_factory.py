import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import llm


class DummyProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class GetLlmProviderFactoryTest(unittest.TestCase):
    def test_get_llm_provider_dispatches_to_expected_class(self):
        cases = [
            ("openai", "OpenAIProvider", {"model": "gpt-4o", "api_key": "test-openai"}),
            (
                "anthropic",
                "AnthropicProvider",
                {"model": "claude-3-sonnet-20240229", "api_key": "test-anthropic"},
            ),
            ("ollama", "OllamaProvider", {"model": "mistral", "base_url": "http://localhost:11434"}),
        ]

        for provider_type, attribute_name, kwargs in cases:
            with self.subTest(provider_type=provider_type):
                with patch.object(llm, attribute_name, DummyProvider):
                    provider = llm.get_llm_provider(provider_type, **kwargs)

                self.assertIsInstance(provider, DummyProvider)
                self.assertEqual(provider.kwargs, kwargs)

    def test_get_llm_provider_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            llm.get_llm_provider("unknown")


if __name__ == "__main__":
    unittest.main()
