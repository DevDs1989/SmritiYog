import pytest
from sqlalchemy import func, select

from app.db.models import Alert, SessionRecord
from app.db.session import SessionLocal
from app.schemas.sync import SyncResponse
from tests.conftest import AUTH

PAYLOAD = {
    "patient_id": "patient-001",
    "session_history": [
        {
            "game_type": "cognitive_visual",
            "domain": "attention",
            "accuracy": 0.82,
            "response_time_ms": 4200,
            "date": "2026-08-30T09:00:00+00:00",
        },
        {
            "game_type": "memory_voice",
            "domain": "memory",
            "accuracy": 0.41,
            "response_time_ms": 7100,
            "date": "2026-08-31T09:00:00+00:00",
        },
    ],
    "rag_delta": [
        {
            "id": "p1",
            "kind": "person",
            "text": "Meera, daughter, visits every Sunday",
            "metadata": {"relation": "daughter", "notes": "brings jasmine flowers"},
        },
        {
            "id": "ph1",
            "kind": "photo",
            "text": "Family photo at Diwali 2019",
            "metadata": {"person_name": "Meera", "closeness": "close"},
        },
        {
            "id": "e1",
            "kind": "event",
            "text": "Takes blood pressure tablet after breakfast",
            "metadata": {"tags": ["medicine"], "answer": "after breakfast"},
        },
    ],
    "domain_scores": {
        "emotional": 0.72,
        "memory": 0.38,
        "attention": 0.80,
        "pattern": 0.65,
    },
}


async def test_sync_returns_valid_response_and_persists_history(client):
    resp = await client.post("/sync", json=PAYLOAD, headers=AUTH)
    assert resp.status_code == 200, resp.text

    body = SyncResponse.model_validate(resp.json())
    assert body.batch, "expected a pre-generated round batch"
    # Weakest domain is memory, but memory_voice was the last game played,
    # so the planner rotates to the next-weakest domain instead.
    assert {r.game_type for r in body.batch} == {"cognitive_visual"}
    assert all(r.prompt_text for r in body.batch)
    assert {e.id for e in body.embeddings} == {"p1", "ph1", "e1"}

    async with SessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(SessionRecord).where(
                SessionRecord.patient_id == "patient-001"
            )
        )
    assert count == 2


async def test_sync_does_not_duplicate_history_on_resync(client):
    await client.post("/sync", json=PAYLOAD, headers=AUTH)
    await client.post("/sync", json=PAYLOAD, headers=AUTH)

    async with SessionLocal() as db:
        count = await db.scalar(select(func.count()).select_from(SessionRecord))
    assert count == 2


async def test_sync_requires_token(client):
    assert (await client.post("/sync", json=PAYLOAD)).status_code == 401


async def test_dashboard_summarises_domains(client):
    await client.post("/sync", json=PAYLOAD, headers=AUTH)
    resp = await client.get("/dashboard/patient-001", headers=AUTH)
    assert resp.status_code == 200

    body = resp.json()
    assert body["total_sessions"] == 2
    assert body["domains"]["memory"]["accuracy_overall"] == 0.41
    assert isinstance(body["alerts"], list)


async def test_voice_clone_stub(client):
    resp = await client.post(
        "/onboard/voice-clone",
        json={
            "patient_id": "patient-001",
            "speaker_label": "daughter",
            "audio_base64": "aGVsbG8=",
        },
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.json()["voice_model_ref"].startswith("mock-voice://")


async def test_voice_clone_rejects_empty_audio(client):
    resp = await client.post(
        "/onboard/voice-clone",
        json={"patient_id": "p", "speaker_label": "son", "audio_base64": ""},
        headers=AUTH,
    )
    assert resp.status_code == 422


RECALL = {
    "patient_id": "patient-001",
    "photo_ref": "ph1",
    "expected_answer": "Meera",
    "photo_caption": "Family photo at Diwali 2019",
    "transcript": "That is Meera, my daughter. She brings jasmine every Sunday.",
}


async def test_recall_parse_extracts_facts(client, monkeypatch):
    from app.services import gemini

    monkeypatch.setattr(
        gemini,
        "generate_text",
        lambda prompt: """```json
        {"answer_matched": true,
         "facts": [{"kind": "person", "text": "Meera brings jasmine every Sunday",
                    "metadata": {"relation": "daughter"}}]}
        ```""",
    )
    resp = await client.post("/recall/parse", json=RECALL, headers=AUTH)
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer_matched"] is True
    assert len(body["facts"]) == 1
    fact = body["facts"][0]
    assert fact["text"] == "Meera brings jasmine every Sunday"
    assert fact["metadata"]["source"] == "photo_recall"
    assert fact["metadata"]["photo_ref"] == "ph1"
    assert fact["id"]  # server-assigned, ready to store as rag_delta


async def test_recall_parse_falls_back_when_gemini_fails(client, monkeypatch):
    from app.services import gemini

    def boom(prompt):
        raise RuntimeError("503")

    monkeypatch.setattr(gemini, "generate_text", boom)
    resp = await client.post("/recall/parse", json=RECALL, headers=AUTH)
    assert resp.status_code == 200
    # Substring match still scores the round; no facts invented.
    assert resp.json() == {"answer_matched": True, "facts": []}


async def test_recall_parse_requires_token(client):
    assert (await client.post("/recall/parse", json=RECALL)).status_code == 401
