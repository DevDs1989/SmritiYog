"""Hybrid retrieval over the request-scoped rag_delta: structured filter first,
cosine rerank only when the round type needs fuzzy matching.
"""

import math


def filter_structured(
    rag_delta: list[dict],
    kind: str | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Exact-match filter on `kind` and on `metadata` keys.

    A list-valued metadata field matches if the wanted value is in it (so
    `{"tags": "birthday"}` matches an item tagged `["birthday", "family"]`).
    """
    out = []
    for item in rag_delta:
        if kind is not None and item.get("kind") != kind:
            continue
        meta = item.get("metadata") or {}
        for key, wanted in (metadata_filter or {}).items():
            actual = meta.get(key)
            if isinstance(actual, (list, tuple, set)):
                if wanted not in actual:
                    break
            elif actual != wanted:
                break
        else:
            out.append(item)
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def retrieve_top_k(
    query_embedding: list[float],
    candidates: list[dict],
    k: int = 3,
    min_score: float = 0.5,
) -> list[dict]:
    """Candidates must already carry an `embedding` key (content_node embeds
    the whole rag_delta up front in one batched call).

    Returns [] when nothing clears `min_score` -- callers fall back to a
    non-personalized round rather than forcing a weak match.
    """
    scored = [
        (cosine_similarity(query_embedding, c.get("embedding") or []), c)
        for c in candidates
    ]
    hits = [c for score, c in sorted(scored, key=lambda s: -s[0]) if score >= min_score]
    return hits[:k]
