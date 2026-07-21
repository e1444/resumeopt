"""Embedding-based cosine-similarity matching against the skill cache.

So a skill doesn't need every phrasing variant (ipynb/jupyter, BSc/
undergraduate degree, GLM/GBM style abbreviations) hand-enumerated as a
cache alias to be matched. Still deterministic given a fixed embedding model
and a fixed cache: the same input text always produces the same vector and
therefore the same match decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from llm import LLMProvider

from .base import Matcher, MatchCandidate
from .embedding_cache import EmbeddingCache
from .models import SkillRecord


class SemanticMatcher(Matcher):
    """Embedding-based cosine-similarity matching against the skill cache.

    Raises NotImplementedError at construction time if the provided
    LLMProvider doesn't support embed() (e.g. Anthropic, Ollama today), so
    callers can catch it and disable semantic matching gracefully instead of
    failing the whole parser.

    Threshold calibration note: bare short terms (e.g. the literal string
    "ipynb") don't carry enough signal in isolation for `text-embedding-3-small`
    to reliably beat unrelated cache entries - in a manual check, bare "ipynb"
    scored *lower* against "jupyter" (0.44) than against "python" (0.47), a
    wrong answer. Embedding the raw term together with its surrounding
    posting-chunk `context` fixed this (0.51 for jupyter, python no longer in
    the top 3) and gave a healthy margin over unrelated pairs (observed
    ceiling ~0.32 for genuinely unrelated terms with equivalent context vs.
    ~0.50-0.56 for true matches). `context` should therefore be passed
    whenever it's available; DEFAULT_SIMILARITY_THRESHOLD is calibrated for
    the with-context case.

    DO NOT reuse DEFAULT_SIMILARITY_THRESHOLD as a default for a new
    matching task without re-validating on that task's own real data (see
    AGENTS.md's "LLM Scoring Rubric Design" section). This exact mistake was
    made and caught once already: `tailoring.retrieval` initially reused this
    0.45 default for project-fact-pool retrieval (matching long free-form
    fact/requirement text, not short skill names) and it caused a full
    precision collapse - a posting fully misaligned with a project
    semantically "matched" every one of its facts via vague shared
    vocabulary (scores 0.45-0.63). The one genuine match found there scored
    0.7375. `tailoring.retrieval` now defines and validates its own,
    higher, task-specific threshold instead of inheriting this one.

    Cache reference-text embeddings (canonical name + aliases
    for every skill) are looked up in an optional `EmbeddingCache` first, so a
    stable skills cache only pays the embedding cost once, not on every parser
    construction/run; only new/changed reference texts get embedded and the
    cache is updated in place.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.45

    def __init__(
        self,
        skills: Sequence[SkillRecord],
        llm_provider: LLMProvider,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        self.llm = llm_provider
        self.similarity_threshold = similarity_threshold
        self.embedding_cache = embedding_cache
        self._canonical_names: List[str] = []
        self._reference_texts: List[str] = []
        self._reference_owner_index: List[int] = []

        for skill_index, record in enumerate(skills):
            self._canonical_names.append(record.name)
            for text in (record.name, *record.aliases):
                cleaned = text.strip()
                if not cleaned:
                    continue
                self._reference_texts.append(cleaned)
                self._reference_owner_index.append(skill_index)

        self._reference_embeddings: Optional[np.ndarray] = None
        if self._reference_texts:
            embedded_by_text = self._embed_reference_texts(self._reference_texts)
            self._reference_embeddings = np.array(
                [embedded_by_text[text] for text in self._reference_texts], dtype=float
            )

    def _embed_reference_texts(self, texts: List[str]) -> Dict[str, List[float]]:
        model_key = getattr(self.llm, "embedding_model", "default")

        if self.embedding_cache is None:
            # Raises NotImplementedError here if llm_provider doesn't support
            # embeddings; callers construct this inside a try/except.
            vectors = self.llm.embed(texts)
            return dict(zip(texts, vectors))

        found, missing = self.embedding_cache.get_many(model_key, texts)
        if missing:
            # Raises NotImplementedError here if llm_provider doesn't support
            # embeddings; callers construct this inside a try/except.
            new_vectors = self.llm.embed(missing)
            new_entries = dict(zip(missing, new_vectors))
            self.embedding_cache.put_many(model_key, new_entries)
            self.embedding_cache.save()
            found.update(new_entries)
        return found

    def match(self, raw_term: str, context: str = "") -> List[MatchCandidate]:
        return self.match_batch([raw_term], contexts=[context])[0]

    def match_batch(
        self,
        raw_terms: Sequence[str],
        contexts: Optional[Sequence[str]] = None,
    ) -> List[List[MatchCandidate]]:
        cleaned_terms = [str(term).strip() for term in raw_terms]
        if not cleaned_terms or self._reference_embeddings is None:
            return [[] for _ in raw_terms]

        contexts = list(contexts) if contexts is not None else ["" for _ in cleaned_terms]
        embeddable_texts = [
            f"{term} ({context.strip()})" if context and context.strip() else term
            for term, context in zip(cleaned_terms, contexts)
        ]

        non_empty_indices = [index for index, term in enumerate(cleaned_terms) if term]
        results: List[List[MatchCandidate]] = [[] for _ in cleaned_terms]
        if not non_empty_indices:
            return results

        candidate_embeddings = np.array(
            self.llm.embed([embeddable_texts[index] for index in non_empty_indices]), dtype=float
        )
        similarities = _cosine_similarity_matrix(candidate_embeddings, self._reference_embeddings)

        for row_index, term_index in enumerate(non_empty_indices):
            row = similarities[row_index]

            # A raw term can legitimately correspond to more than one cache
            # skill (e.g. "GLM/GBM" should resolve to both "glm" and "gbm"),
            # so this keeps every skill whose best reference-text similarity
            # clears the threshold, not just the single closest one.
            best_similarity_per_skill: Dict[int, float] = {}
            for reference_index, similarity in enumerate(row):
                skill_index = self._reference_owner_index[reference_index]
                if float(similarity) > best_similarity_per_skill.get(skill_index, -1.0):
                    best_similarity_per_skill[skill_index] = float(similarity)

            matches = [
                MatchCandidate(
                    canonical_name=self._canonical_names[skill_index],
                    match_type="semantic",
                    confidence=round(min(1.0, similarity), 4),
                    similarity=round(similarity, 4),
                )
                for skill_index, similarity in best_similarity_per_skill.items()
                if similarity >= self.similarity_threshold
            ]
            matches.sort(key=lambda candidate: -(candidate.similarity or 0.0))
            results[term_index] = matches

        return results


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between every row of `a` and every row of `b`."""

    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = np.divide(a, a_norms, out=np.zeros_like(a), where=a_norms != 0)
    b_normalized = np.divide(b, b_norms, out=np.zeros_like(b), where=b_norms != 0)
    return a_normalized @ b_normalized.T
