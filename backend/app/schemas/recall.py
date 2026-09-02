from pydantic import BaseModel

from app.schemas.sync import RagDeltaItem


class RecallParseRequest(BaseModel):
    """One finished `photo_recall` round: what the patient said about the photo.

    The app holds the RAG store, so it sends the round's own context back up --
    the backend has no memory of the photo between requests.
    """

    patient_id: str
    photo_ref: str
    expected_answer: str
    transcript: str
    photo_caption: str = ""


class RecallParseResponse(BaseModel):
    answer_matched: bool
    facts: list[RagDeltaItem]  # new, ready to store; send back as rag_delta next sync
