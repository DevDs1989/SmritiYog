from typing import Literal

from pydantic import BaseModel

GameType = Literal["cognitive_visual", "memory_voice", "relationship"]
Domain = Literal["emotional", "memory", "attention", "pattern"]
Difficulty = Literal["easy", "medium", "hard"]


class SessionHistoryItem(BaseModel):
    game_type: GameType
    domain: Domain
    accuracy: float
    response_time_ms: int
    date: str  # ISO 8601


class RagDeltaItem(BaseModel):
    id: str
    kind: Literal["person", "event", "photo"]
    text: str
    metadata: dict


class DomainScores(BaseModel):
    emotional: float
    memory: float
    attention: float
    pattern: float


class SyncRequest(BaseModel):
    patient_id: str
    session_history: list[SessionHistoryItem]
    rag_delta: list[RagDeltaItem]
    domain_scores: DomainScores


# --- per-game metadata shapes (documentation of what `RoundOutput.metadata`
# carries; the app renders off these keys) ---


class CognitiveVisualMetadata(BaseModel):
    items: list[str]
    odd_one_out_index: int
    similarity_level: Literal["low", "medium", "high"]


class MemoryVoiceMetadata(BaseModel):
    subtype: Literal[
        "favorite_recall", "sequence_recall", "photo_recall", "sequence_from_sound"
    ]
    sequence: list[str] | None = None
    audio_cues: list[str] | None = None
    photo_ref: str | None = None  # photo_recall: rag_delta item id of kind="photo"


class RelationshipMetadata(BaseModel):
    mode: Literal["guess_person_from_photo", "guess_relation_from_voice"]
    photo_ref: str | None = None
    voice_clue_text: str | None = None


class RoundOutput(BaseModel):
    game_type: str
    difficulty: Difficulty
    prompt_text: str
    correct_answer: str
    metadata: dict


class EmbeddingOutput(BaseModel):
    id: str
    embedding: list[float]


class AlertOutput(BaseModel):
    domain: str
    message: str


class SyncResponse(BaseModel):
    batch: list[RoundOutput]
    embeddings: list[EmbeddingOutput]
    alerts: list[AlertOutput]
