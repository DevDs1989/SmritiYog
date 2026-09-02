from typing import TypedDict


class PatientState(TypedDict):
    patient_id: str
    session_history: list[dict]
    rag_delta: list[dict]
    domain_scores: dict
    game_type: str | None
    difficulty: str | None
    retrieved_context: list[dict]
    generated_rounds: list[dict]
    embeddings: list[dict]
    alerts: list[dict]
