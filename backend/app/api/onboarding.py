import base64
import hashlib

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import require_token, verify_patient_scope
from app.schemas.onboarding import VoiceCloneRequest, VoiceCloneResponse

router = APIRouter()


@router.post("/onboard/voice-clone", response_model=VoiceCloneResponse)
async def voice_clone(
    payload: VoiceCloneRequest,
    token: str = Depends(require_token),
) -> VoiceCloneResponse:
    """Base64 audio in, a voice model reference out.

    TODO: swap the stub for the real cloning call. Everything downstream only
    needs `voice_model_ref` to be stable per (patient, speaker), so the rest of
    the flow can be built and tested against this.
    """
    verify_patient_scope(payload.patient_id, token)
    try:
        sample = base64.b64decode(payload.audio_base64, validate=True)
    except Exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "audio_base64 invalid")
    if not sample:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty audio sample")

    digest = hashlib.sha256(
        f"{payload.patient_id}:{payload.speaker_label}".encode()
    ).hexdigest()[:16]
    return VoiceCloneResponse(
        patient_id=payload.patient_id,
        speaker_label=payload.speaker_label,
        voice_model_ref=f"mock-voice://{digest}",
        status="stubbed",
    )
