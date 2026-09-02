"""Planner Agent: pick game_type + difficulty. Rule-based; Gemini only breaks
a genuine tie between two domains that map to different games.
"""

import logging

from app.agents.state import PatientState
from app.services import gemini

log = logging.getLogger(__name__)

DOMAIN_GAME = {
    "memory": "memory_voice",
    "emotional": "relationship",
    "attention": "cognitive_visual",
    "pattern": "cognitive_visual",
}

TIE_EPSILON = 0.02


def _difficulty(score: float) -> str:
    if score < 0.4:
        return "easy"
    if score < 0.75:
        return "medium"
    return "hard"


def _break_tie(candidates: list[str], scores: dict) -> str:
    """Two near-identical weak domains -> ask Gemini once. Cheap and rare."""
    prompt = (
        "A dementia patient's cognitive domain scores (0-1, lower is weaker):\n"
        + "\n".join(f"- {d}: {scores[d]:.2f}" for d in scores)
        + f"\n\nThese domains are tied for weakest: {', '.join(candidates)}."
        " Which single domain should today's exercise target?"
        " Reply with only the domain name."
    )
    try:
        answer = gemini.generate_text(prompt).strip().lower()
    except Exception:
        log.warning("tie-break call failed; taking weakest domain", exc_info=True)
        return candidates[0]
    return next((d for d in candidates if d in answer), candidates[0])


def planner_node(state: PatientState) -> dict:
    scores = dict(state["domain_scores"])
    ranked = sorted(scores, key=lambda d: scores[d])

    weakest = ranked[0]
    tied = [d for d in ranked if scores[d] - scores[weakest] < TIE_EPSILON]
    if len(tied) > 1 and len({DOMAIN_GAME[d] for d in tied}) > 1:
        weakest = _break_tie(tied, scores)

    game_type = DOMAIN_GAME[weakest]

    # Don't serve the same game twice in a row -- fall to the next weakest
    # domain that plays differently.
    history = state.get("session_history") or []
    last_game = history[-1].get("game_type") if history else None
    if game_type == last_game:
        alt = next((d for d in ranked if DOMAIN_GAME[d] != last_game), None)
        if alt:
            weakest, game_type = alt, DOMAIN_GAME[alt]

    return {"game_type": game_type, "difficulty": _difficulty(scores[weakest])}
