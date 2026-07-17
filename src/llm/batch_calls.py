"""Shared async LLM-call batching/retry plumbing, used by both `chunker`
(LLM-based chunking) and `parser` (extraction/categorization/redundancy/
atomicity stages) - lives in `llm` (the lowest-level package) specifically to
avoid a circular dependency between `chunker` and `parser`.

All calls here go through `LLMProvider.call_json`, which is synchronous -
`asyncio.to_thread` runs each call off the event loop so many candidate
batches can be in flight concurrently via `asyncio.gather`. These are plain
I/O-bound API calls with no CPU-bound work between them, so this is all the
concurrency needed here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from .base import LLMProvider

_logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_BATCH_SIZE = 20
"""Judging many candidates in a single call risks degenerate, collapsed
responses (observed empirically earlier in this project); smaller batches
keep genuine per-candidate judgment."""

DEFAULT_MAX_CONCURRENCY = 8
"""Caps how many batch calls are in flight at once (via an asyncio.Semaphore)
- a safety margin against rate limits on very large candidate lists, not a
real parallelism concern."""


async def call_json_with_retry_async(
    llm_provider: LLMProvider,
    call_label: str,
    **call_json_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Async wrapper around `LLMProvider.call_json` with a couple of retries.

    Logs and returns `None` on final failure instead of raising, so one
    failed batch doesn't crash the whole `asyncio.gather`.
    """

    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(llm_provider.call_json, **call_json_kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, logged below
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS:
                _logger.warning(
                    "%s call failed (attempt %d/%d): %s - retrying",
                    call_label,
                    attempt,
                    _RETRY_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
    _logger.warning("%s call failed after %d attempts, giving up: %s", call_label, _RETRY_ATTEMPTS, last_exc)
    return None


def batch_list(items: Sequence[str], batch_size: int) -> List[List[str]]:
    """Split a sequence into ordered sub-lists of at most `batch_size` items each."""

    items = list(items)
    if not items:
        return []
    if batch_size <= 0 or batch_size >= len(items):
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
