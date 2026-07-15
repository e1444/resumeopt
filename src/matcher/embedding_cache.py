"""Persistent on-disk cache for skill-cache reference-text embeddings.

Keyed by (embedding model name, text), so adding or changing a single skill's
name/alias/related term only requires embedding that one new text, not
re-embedding the entire skills cache. Without this, SemanticMatcher would
re-embed every reference text (canonical name + aliases + related terms for
every cache entry) on every parser construction/run, which is wasteful cost
and latency once the cache stabilizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class EmbeddingCache:
    """JSON-backed cache of {model_name: {text: embedding_vector}}.

    Not safe for concurrent writers (no file locking) - fine for this
    project's single-process CLI usage, but worth knowing if this is ever
    reused in a concurrent/service context.
    """

    def __init__(self, cache_path: Path = Path("build/cache/skill_embeddings_cache.json")):
        self.cache_path = Path(cache_path)
        self._store: Dict[str, Dict[str, List[float]]] = self._load()

    def _load(self) -> Dict[str, Dict[str, List[float]]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def get_many(self, model: str, texts: Sequence[str]) -> Tuple[Dict[str, List[float]], List[str]]:
        """Return (found: {text: vector}, missing: [text, ...]) for the given model."""

        model_store = self._store.get(model, {})
        found: Dict[str, List[float]] = {}
        missing: List[str] = []
        for text in texts:
            if text in model_store:
                found[text] = model_store[text]
            else:
                missing.append(text)
        return found, missing

    def put_many(self, model: str, entries: Dict[str, List[float]]) -> None:
        """Merge newly computed embeddings into the in-memory store (call save() to persist)."""

        model_store = self._store.setdefault(model, {})
        model_store.update(entries)

    def save(self) -> None:
        """Persist the in-memory store to disk as JSON."""

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self._store, handle)
