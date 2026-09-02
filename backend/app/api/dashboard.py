from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.monitor import WINDOW
from app.core.security import require_token, verify_patient_scope
from app.db.models import Alert, SessionRecord
from app.db.session import get_session

router = APIRouter()


@router.get("/dashboard/{patient_id}")
async def dashboard(
    patient_id: str,
    db: AsyncSession = Depends(get_session),
    token: str = Depends(require_token),
) -> dict:
    verify_patient_scope(patient_id, token)

    rows = (
        await db.execute(
            select(SessionRecord)
            .where(SessionRecord.patient_id == patient_id)
            .order_by(SessionRecord.date)
        )
    ).scalars().all()

    by_domain: dict[str, list[SessionRecord]] = {}
    for row in rows:
        by_domain.setdefault(row.domain, []).append(row)

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    domains = {
        domain: {
            "sessions": len(items),
            "accuracy_overall": avg([i.accuracy for i in items]),
            "accuracy_recent": avg([i.accuracy for i in items[-WINDOW:]]),
            "avg_response_time_ms": round(avg([i.response_time_ms for i in items])),
            "last_played": items[-1].date.isoformat(),
        }
        for domain, items in by_domain.items()
    }

    alerts = (
        await db.execute(
            select(Alert)
            .where(Alert.patient_id == patient_id, Alert.resolved.is_(False))
            .order_by(Alert.created_at.desc())
        )
    ).scalars().all()

    return {
        "patient_id": patient_id,
        "total_sessions": len(rows),
        "domains": domains,
        "alerts": [
            {
                "id": a.id,
                "domain": a.domain,
                "message": a.message,
                "created_at": a.created_at.isoformat(),
            }
            for a in alerts
        ],
    }
