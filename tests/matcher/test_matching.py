"""Independent tests for each Matcher strategy in src/matcher/.

Cases are derived from the alias-heavy edge cases discussed for this project
(jupyter/ipynb, python/py, GLM/GBM-style abbreviations) so each matcher's
behavior is validated against realistic examples, not just toy strings.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import LLMProvider
from matcher import EmbeddingCache, ExactAliasMatcher, LLMGroundingMatcher, SemanticMatcher, SkillRecord


SAMPLE_SKILLS = [
    SkillRecord(
        name="jupyter",
        aliases=("jupyter notebook", "jupyter lab", "ipynb"),
    ),
    SkillRecord(name="python", aliases=("py",)),
    SkillRecord(name="git", aliases=()),
    SkillRecord(name="glm", aliases=()),
    SkillRecord(name="gbm", aliases=()),
]


class ExactAliasMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = ExactAliasMatcher(SAMPLE_SKILLS)

    def test_matches_canonical_name_exactly(self) -> None:
        matches = self.matcher.match("python")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].canonical_name, "python")
        self.assertEqual(matches[0].match_type, "exact")

    def test_matches_alias(self) -> None:
        matches = self.matcher.match("ipynb")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].canonical_name, "jupyter")
        self.assertEqual(matches[0].match_type, "alias")

    def test_normalizes_case_and_whitespace(self) -> None:
        matches = self.matcher.match("  PyThOn  ")

        self.assertEqual(matches[0].canonical_name, "python")

    def test_returns_empty_for_unknown_term(self) -> None:
        self.assertEqual(self.matcher.match("kubernetes"), [])

    def test_does_not_split_compound_abbreviation(self) -> None:
        # This is exactly the gap SemanticMatcher exists to cover: exact/alias
        # lookup has no entry for the literal combined string "glm/gbm".
        self.assertEqual(self.matcher.match("glm/gbm"), [])

    def test_match_batch_matches_looping_behavior(self) -> None:
        results = self.matcher.match_batch(["python", "ipynb", "kubernetes"])

        self.assertEqual(results[0][0].canonical_name, "python")
        self.assertEqual(results[1][0].canonical_name, "jupyter")
        self.assertEqual(results[2], [])


class FakeEmbeddingProvider(LLMProvider):
    """Deterministic fake embeddings for fast, free, reproducible unit tests.

    Known text -> fixed vector so cosine similarity between semantically
    related pairs (ipynb/jupyter, glm-gbm/glm, glm-gbm/gbm) is high and
    unrelated pairs are low, without needing a real embeddings API call.
    """

    # dims: [jupyter, python, glm, gbm, degree]
    _KNOWN_VECTORS: Dict[str, List[float]] = {
        "jupyter": [1.0, 0.0, 0.0, 0.0, 0.0],
        "jupyter notebook": [1.0, 0.0, 0.0, 0.0, 0.0],
        "jupyter lab": [1.0, 0.0, 0.0, 0.0, 0.0],
        "ipynb": [0.97, 0.0, 0.0, 0.0, 0.0],
        "notebook": [0.9, 0.0, 0.0, 0.0, 0.0],
        "python": [0.0, 1.0, 0.0, 0.0, 0.0],
        "py": [0.0, 0.97, 0.0, 0.0, 0.0],
        "glm": [0.0, 0.0, 1.0, 0.0, 0.0],
        "gbm": [0.0, 0.0, 0.0, 1.0, 0.0],
        "glm/gbm": [0.0, 0.0, 0.9, 0.9, 0.0],
        "bachelor's degree": [0.0, 0.0, 0.0, 0.0, 1.0],
        "bsc": [0.0, 0.0, 0.0, 0.0, 0.95],
    }
    _UNKNOWN_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0]

    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def call_json(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._KNOWN_VECTORS.get(text.strip().lower(), list(self._UNKNOWN_VECTOR)) for text in texts]


class NoEmbeddingProvider(LLMProvider):
    """Simulates a provider without embedding support (e.g. Anthropic, Ollama)."""

    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def call_json(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    """Same deterministic vectors as FakeEmbeddingProvider, but counts embed() calls
    so tests can assert an EmbeddingCache actually avoids redundant API calls."""

    def __init__(self) -> None:
        self.embed_call_count = 0
        self.embedded_texts: List[str] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.embed_call_count += 1
        self.embedded_texts.extend(texts)
        return super().embed(texts)


class EmbeddingCacheTest(unittest.TestCase):
    def test_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "embeddings.json"

            cache = EmbeddingCache(cache_path)
            found, missing = cache.get_many("model-a", ["jupyter", "python"])
            self.assertEqual(found, {})
            self.assertEqual(missing, ["jupyter", "python"])

            cache.put_many("model-a", {"jupyter": [1.0, 0.0], "python": [0.0, 1.0]})
            cache.save()

            reloaded = EmbeddingCache(cache_path)
            found, missing = reloaded.get_many("model-a", ["jupyter", "python", "git"])
            self.assertEqual(found, {"jupyter": [1.0, 0.0], "python": [0.0, 1.0]})
            self.assertEqual(missing, ["git"])

    def test_separates_entries_by_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = EmbeddingCache(Path(tmp_dir) / "embeddings.json")
            cache.put_many("model-a", {"jupyter": [1.0, 0.0]})

            found, missing = cache.get_many("model-b", ["jupyter"])

            self.assertEqual(found, {})
            self.assertEqual(missing, ["jupyter"])

    def test_missing_or_corrupt_file_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "does_not_exist.json"
            cache = EmbeddingCache(cache_path)
            self.assertEqual(cache.get_many("model-a", ["jupyter"]), ({}, ["jupyter"]))

            cache_path.write_text("not valid json", encoding="utf-8")
            corrupt_cache = EmbeddingCache(cache_path)
            self.assertEqual(corrupt_cache.get_many("model-a", ["jupyter"]), ({}, ["jupyter"]))


class SemanticMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_skills = [
            SkillRecord(name="jupyter", aliases=("jupyter notebook", "jupyter lab")),
            SkillRecord(name="python", aliases=("py",)),
            SkillRecord(name="glm", aliases=()),
            SkillRecord(name="gbm", aliases=()),
            SkillRecord(name="bachelor's degree", aliases=()),
        ]

    def test_matches_alias_variant_not_in_cache_via_similarity(self) -> None:
        matcher = SemanticMatcher(self.embedding_skills, FakeEmbeddingProvider())

        matches = matcher.match("ipynb")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].canonical_name, "jupyter")
        self.assertEqual(matches[0].match_type, "semantic")
        self.assertGreaterEqual(matches[0].similarity, matcher.similarity_threshold)

    def test_resolves_compound_abbreviation_to_multiple_skills(self) -> None:
        # A lower threshold than the production default (0.80) is used here
        # deliberately: this fake embedding models "glm" and "gbm" as
        # orthogonal concepts, which caps an equal blend's cosine similarity
        # at ~0.71 regardless of scale. Real embeddings (validated separately
        # by the live OpenAI-gated test) place related terms much closer
        # together than that, since they share literal substrings and
        # co-occurrence context.
        matcher = SemanticMatcher(self.embedding_skills, FakeEmbeddingProvider(), similarity_threshold=0.65)

        matches = matcher.match("glm/gbm")

        canonical_names = {match.canonical_name for match in matches}
        self.assertEqual(canonical_names, {"glm", "gbm"})
        for match in matches:
            self.assertEqual(match.match_type, "semantic")

    def test_below_threshold_returns_no_match(self) -> None:
        matcher = SemanticMatcher(self.embedding_skills, FakeEmbeddingProvider())

        self.assertEqual(matcher.match("completely unrelated term"), [])

    def test_match_batch_processes_multiple_terms(self) -> None:
        matcher = SemanticMatcher(self.embedding_skills, FakeEmbeddingProvider())

        results = matcher.match_batch(["ipynb", "py", "completely unrelated term"])

        self.assertEqual(results[0][0].canonical_name, "jupyter")
        self.assertEqual(results[1][0].canonical_name, "python")
        self.assertEqual(results[2], [])

    def test_raises_not_implemented_for_provider_without_embeddings(self) -> None:
        with self.assertRaises(NotImplementedError):
            SemanticMatcher(self.embedding_skills, NoEmbeddingProvider())

    def test_embedding_cache_avoids_redundant_embed_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "embeddings.json"
            provider = CountingEmbeddingProvider()

            first_cache = EmbeddingCache(cache_path)
            first_matcher = SemanticMatcher(self.embedding_skills, provider, embedding_cache=first_cache)
            calls_after_first_construction = provider.embed_call_count
            self.assertGreater(calls_after_first_construction, 0)

            # A fresh SemanticMatcher backed by a fresh EmbeddingCache instance
            # pointed at the same file should find everything already cached
            # on disk and make no further embed() calls for reference texts.
            second_cache = EmbeddingCache(cache_path)
            SemanticMatcher(self.embedding_skills, provider, embedding_cache=second_cache)

            self.assertEqual(provider.embed_call_count, calls_after_first_construction)

            # Sanity check the second matcher still resolves correctly.
            matches = first_matcher.match("ipynb")
            self.assertEqual(matches[0].canonical_name, "jupyter")


class LLMGroundingMatcherTest(unittest.TestCase):
    class GroundingFakeLLMProvider(LLMProvider):
        def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
            raise NotImplementedError

        def call_json(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            prompt_lower = prompt.lower()
            if "skill canonical name: jupyter" in prompt_lower and "ipynb" in prompt_lower:
                return {"is_grounded": True, "reason": "ipynb is a jupyter notebook format"}
            return {"is_grounded": False, "reason": "Not supported"}

    class ExplodingLLMProvider(LLMProvider):
        def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
            raise NotImplementedError

        def call_json(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    def test_confirms_grounding_for_known_edge_case(self) -> None:
        matcher = LLMGroundingMatcher(self.GroundingFakeLLMProvider())

        result = matcher.confirm_grounding(
            posting_text="Experience working with .ipynb files for analysis.",
            canonical_name="jupyter",
            aliases=["jupyter notebook", "jupyter lab", "ipynb"],
            raw_term="ipynb",
            evidence=".ipynb files",
        )

        self.assertTrue(result)

    def test_rejects_unsupported_grounding(self) -> None:
        matcher = LLMGroundingMatcher(self.GroundingFakeLLMProvider())

        result = matcher.confirm_grounding(
            posting_text="We value strong communication skills.",
            canonical_name="python",
            aliases=["py"],
            raw_term="communication",
            evidence="strong communication skills",
        )

        self.assertFalse(result)

    def test_returns_false_when_llm_call_fails(self) -> None:
        matcher = LLMGroundingMatcher(self.ExplodingLLMProvider())

        result = matcher.confirm_grounding(
            posting_text="Some posting text.",
            canonical_name="python",
            aliases=[],
            raw_term="python",
            evidence="posting text",
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
