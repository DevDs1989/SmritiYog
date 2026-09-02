"""Parse what a patient said during a `photo_recall` round.

The round is open-ended: the patient names the person *and* volunteers whatever
else they remember. One Gemini call pulls both out of the transcript -- whether
they got the person right, and any new facts worth keeping. The facts come back
rag_delta-shaped for the app to store; nothing is persisted here.
"""

import logging
import uuid

from fastapi import APIRouter, Depends

from app.core.security import require_token, verify_patient_scope
from app.schemas.recall import RecallParseRequest, RecallParseResponse
from app.schemas.sync import RagDeltaItem
from app.services import gemini

router = APIRouter()
log = logging.getLogger(__name__)

_PARSE_PROMPT = """An elderly patient was shown a photo of a family member and
asked who it is and what they remember about them. Read what they said.

Photo: {caption}
Who it actually is: {expected}
What the patient said: {transcript}

Return ONLY JSON, no prose:
{{"answer_matched": true or false,
  "facts": [{{"kind": "person" or "event",
              "text": "<one short third-person fact>",
              "metadata": {{"relation": "<if known>", "tags": ["<if any>"]}}}}]}}

answer_matched is true if they named the right person or the right relation,
even loosely, by nickname, or mid-sentence. Put in "facts" only details they
volunteered about that person -- not the identification itself, and nothing
they did not actually say. Empty list is fine.
"""


@router.post("/recall/parse", response_model=RecallParseResponse)
async def parse_recall(
    payload: RecallParseRequest,
    token: str = Depends(require_token),
) -> RecallParseResponse:
    verify_patient_scope(payload.patient_id, token)

    matched = payload.expected_answer.strip().lower() in payload.transcript.lower()
    facts: list[RagDeltaItem] = []

    try:
        parsed = gemini.parse_json(
            gemini.generate_text(
                _PARSE_PROMPT.format(
                    caption=payload.photo_caption or payload.photo_ref,
                    expected=payload.expected_answer,
                    transcript=payload.transcript,
                )
            )
        )
        matched = bool(parsed.get("answer_matched", matched))
        for fact in parsed.get("facts") or []:
            text = (fact.get("text") or "").strip()
            if not text:
                continue
            facts.append(
                RagDeltaItem(
                    id=str(uuid.uuid4()),
                    kind=fact.get("kind") if fact.get("kind") in ("person", "event") else "person",
                    text=text,
                    metadata={
                        **(fact.get("metadata") or {}),
                        "source": "photo_recall",
                        "photo_ref": payload.photo_ref,
                    },
                )
            )
    except Exception:
        log.warning("recall parse failed; falling back to substring match", exc_info=True)

    return RecallParseResponse(answer_matched=matched, facts=facts)
