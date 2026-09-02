from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import graph
from app.core.security import require_token, verify_patient_scope
from app.db.models import Patient, SessionRecord
from app.db.session import get_session
from app.schemas.sync import SyncRequest, SyncResponse

router = APIRouter()


@router.post("/sync", response_model=SyncResponse)
async def sync(
    payload: SyncRequest,
    db: AsyncSession = Depends(get_session),
    token: str = Depends(require_token),
) -> SyncResponse:
    verify_patient_scope(payload.patient_id, token)

    if await db.get(Patient, payload.patient_id) is None:
        db.add(Patient(id=payload.patient_id, consent_flag=True))
        await db.flush()

    # The app resends its local history; only take what we haven't stored, so
    # repeated syncs don't skew the trend windows.
    latest = await db.scalar(
        select(SessionRecord.date)
        .where(SessionRecord.patient_id == payload.patient_id)
        .order_by(SessionRecord.date.desc())
        .limit(1)
    )
    for item in payload.session_history:
        date = datetime.fromisoformat(item.date)
        if latest is not None and date <= latest.replace(tzinfo=date.tzinfo):
            continue
        db.add(
            SessionRecord(
                patient_id=payload.patient_id,
                game_type=item.game_type,
                domain=item.domain,
                accuracy=item.accuracy,
                response_time_ms=item.response_time_ms,
                date=date,
            )
        )
    await db.commit()

    result = await graph.ainvoke(
        {
            "patient_id": payload.patient_id,
            "session_history": [i.model_dump() for i in payload.session_history],
            "rag_delta": [i.model_dump() for i in payload.rag_delta],
            "domain_scores": payload.domain_scores.model_dump(),
            "game_type": None,
            "difficulty": None,
            "retrieved_context": [],
            "generated_rounds": [],
            "embeddings": [],
            "alerts": [],
        }
    )

    return SyncResponse(
        batch=result["generated_rounds"],
        embeddings=result["embeddings"],
        alerts=result["alerts"],
    )
