from pydantic import BaseModel


class VoiceCloneRequest(BaseModel):
    """Base64 upload -- keeps the mobile client on plain JSON everywhere."""

    patient_id: str
    speaker_label: str  # e.g. "daughter", used as the clue voice identity
    audio_base64: str
    mime_type: str = "audio/wav"


class VoiceCloneResponse(BaseModel):
    patient_id: str
    speaker_label: str
    voice_model_ref: str
    status: str
