"""Thin Gemini wrapper. Agents call these two functions and nothing else, so
tests mock here instead of stubbing the SDK.
"""

import json
import time
from collections.abc import Callable
from functools import cache
from typing import TypeVar

from google import genai
from google.genai import errors, types

from app import config

T = TypeVar("T")

# Flash models 503 intermittently under load; a sync that gives up on the first
# one falls back to template wording for the whole batch.
_RETRY_CODES = {429, 500, 502, 503, 504}
_ATTEMPTS = 4


@cache
def _client() -> genai.Client:
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _with_retry(call: Callable[[], T]) -> T:
    for attempt in range(_ATTEMPTS):
        try:
            return call()
        except errors.APIError as exc:
            if exc.code not in _RETRY_CODES or attempt == _ATTEMPTS - 1:
                raise
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def generate_text(prompt: str) -> str:
    response = _with_retry(
        lambda: _client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            # Phrasing a game prompt does not need deliberation, and the batch
            # call sits in the sync request path -- default thinking costs ~60s.
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
    )
    return response.text or ""


def embed_batch(texts: list[str]) -> list[list[float]]:
    """One API call for the whole list -- never loop this per item."""
    if not texts:
        return []
    response = _with_retry(
        lambda: _client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                # Truncated from 3072 to keep the sync payload small.
                # Unnormalized is fine: retrieval compares by cosine, which
                # normalizes anyway.
                output_dimensionality=config.GEMINI_EMBED_DIM,
            ),
        )
    )
    return [e.values for e in response.embeddings or []]


def parse_json(raw: str):
    """Gemini wraps JSON in ```json fences often enough to be worth one place."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(cleaned)
