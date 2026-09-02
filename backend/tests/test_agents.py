from datetime import datetime, timedelta, timezone

import pytest

from app.agents.content import content_node
from app.agents.monitor import monitor_node
from app.agents.planner import planner_node
from app.db.models import SessionRecord
from app.db.session import SessionLocal
from app.schemas.sync import (
    CognitiveVisualMetadata,
    MemoryVoiceMetadata,
    RelationshipMetadata,
)
from app.services import gemini

RAG = [
    {
        "id": "p1",
        "kind": "person",
        "text": "Meera, daughter",
        "metadata": {"relation": "daughter", "notes": "brings jasmine every Sunday"},
    },
    {
        "id": "ph1",
        "kind": "photo",
        "text": "Diwali 2019",
        "metadata": {"person_name": "Meera", "closeness": "close"},
    },
    {
        "id": "e1",
        "kind": "event",
        "text": "BP tablet after breakfast",
        "metadata": {"tags": ["medicine"], "answer": "after breakfast"},
    },
]


def state(**over) -> dict:
    base = {
        "patient_id": "p-test",
        "session_history": [],
        "rag_delta": [],
        "domain_scores": {
            "emotional": 0.9,
            "memory": 0.3,
            "attention": 0.8,
            "pattern": 0.7,
        },
        "game_type": None,
        "difficulty": None,
        "retrieved_context": [],
        "generated_rounds": [],
        "embeddings": [],
        "alerts": [],
    }
    return base | over


# --- planner ---------------------------------------------------------------


def test_planner_targets_weakest_domain():
    out = planner_node(state())
    assert out == {"game_type": "memory_voice", "difficulty": "easy"}


def test_planner_difficulty_scales_with_score():
    scores = {"emotional": 0.9, "memory": 0.6, "attention": 0.8, "pattern": 0.95}
    assert planner_node(state(domain_scores=scores))["difficulty"] == "medium"
    scores["memory"] = 0.85
    assert planner_node(state(domain_scores=scores))["difficulty"] == "hard"


def test_planner_avoids_repeating_last_game():
    history = [{"game_type": "memory_voice", "domain": "memory"}]
    out = planner_node(state(session_history=history))
    assert out["game_type"] != "memory_voice"


def test_planner_asks_gemini_only_on_a_tie(monkeypatch):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return "emotional"

    monkeypatch.setattr(gemini, "generate_text", fake)

    planner_node(state())
    assert calls == []  # clear winner -> no API call

    tied = {"emotional": 0.30, "memory": 0.31, "attention": 0.9, "pattern": 0.9}
    assert planner_node(state(domain_scores=tied))["game_type"] == "relationship"
    assert len(calls) == 1


# --- content ---------------------------------------------------------------


def test_content_cognitive_visual_metadata_is_valid():
    out = content_node(state(game_type="cognitive_visual", difficulty="hard"))
    assert len(out["generated_rounds"]) >= 10
    for round_ in out["generated_rounds"]:
        meta = CognitiveVisualMetadata.model_validate(round_["metadata"])
        assert meta.similarity_level == "high"
        assert len(meta.items) == 6
        assert meta.items[meta.odd_one_out_index] == round_["correct_answer"]


def test_content_embeds_rag_delta_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gemini,
        "embed_batch",
        lambda texts: calls.append(texts) or [[1.0] * 8 for _ in texts],
    )
    out = content_node(state(game_type="memory_voice", difficulty="easy", rag_delta=RAG))
    assert len(calls) == 1
    assert [e["id"] for e in out["embeddings"]] == ["p1", "ph1", "e1"]


def test_content_hard_memory_voice_serves_photo_recall():
    out = content_node(state(game_type="memory_voice", difficulty="hard", rag_delta=RAG))
    recalls = [
        r for r in out["generated_rounds"] if r["metadata"]["subtype"] == "photo_recall"
    ]
    assert recalls
    # Answer always comes verbatim from the retrieved photo, never invented.
    assert all(r["correct_answer"] == "Meera" for r in recalls)
    assert all(r["metadata"]["photo_ref"] == "ph1" for r in recalls)
    for round_ in out["generated_rounds"]:
        MemoryVoiceMetadata.model_validate(round_["metadata"])


def test_content_falls_back_to_generic_when_nothing_retrieved():
    out = content_node(state(game_type="memory_voice", difficulty="hard", rag_delta=[]))
    subtypes = {r["metadata"]["subtype"] for r in out["generated_rounds"]}
    assert subtypes <= {"favorite_recall", "sequence_from_sound"}
    assert "photo_recall" not in subtypes


def test_content_relationship_needs_rag_and_never_invents_a_person():
    out = content_node(state(game_type="relationship", difficulty="easy", rag_delta=RAG))
    for round_ in out["generated_rounds"]:
        meta = RelationshipMetadata.model_validate(round_["metadata"])
        assert meta.photo_ref == "ph1"
        assert round_["correct_answer"] == "Meera"

    empty = content_node(state(game_type="relationship", difficulty="easy", rag_delta=[]))
    assert all(r["game_type"] == "cognitive_visual" for r in empty["generated_rounds"])


def test_content_hard_relationship_generates_a_voice_clue():
    out = content_node(state(game_type="relationship", difficulty="hard", rag_delta=RAG))
    for round_ in out["generated_rounds"]:
        meta = RelationshipMetadata.model_validate(round_["metadata"])
        assert meta.mode == "guess_relation_from_voice"
        assert meta.voice_clue_text
        assert round_["correct_answer"] == "daughter"


# --- monitor ---------------------------------------------------------------


async def _seed(accuracies: list[float], domain: str = "memory") -> None:
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    async with SessionLocal() as db:
        for i, acc in enumerate(accuracies):
            db.add(
                SessionRecord(
                    patient_id="p-test",
                    game_type="memory_voice",
                    domain=domain,
                    accuracy=acc,
                    response_time_ms=5000,
                    date=start + timedelta(days=i),
                )
            )
        await db.commit()


async def test_monitor_flags_a_clear_decline():
    await _seed([0.9, 0.88, 0.92, 0.5, 0.48, 0.45])
    out = await monitor_node(state())
    assert [a["domain"] for a in out["alerts"]] == ["memory"]

    from sqlalchemy import select

    from app.db.models import Alert

    async with SessionLocal() as db:
        rows = (await db.execute(select(Alert))).scalars().all()
    assert len(rows) == 1 and rows[0].patient_id == "p-test"


async def test_monitor_ignores_noise():
    await _seed([0.80, 0.82, 0.79, 0.80, 0.78, 0.81])
    assert (await monitor_node(state()))["alerts"] == []


async def test_monitor_needs_enough_history():
    await _seed([0.9, 0.4, 0.3])
    assert (await monitor_node(state()))["alerts"] == []


async def test_monitor_asks_gemini_when_borderline(monkeypatch):
    monkeypatch.setattr(gemini, "generate_text", lambda prompt: "memory")
    await _seed([0.80, 0.82, 0.81, 0.70, 0.71, 0.70])
    assert [a["domain"] for a in (await monitor_node(state()))["alerts"]] == ["memory"]
