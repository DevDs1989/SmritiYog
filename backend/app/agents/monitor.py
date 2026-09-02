"""Monitor Agent: trend check over the *persisted* history (not just this
request's payload), then raise + persist caretaker alerts.

Rule-based first. Gemini is asked one batched question, and only about the
domains whose drop is borderline enough to be noise.
"""

import logging

from sqlalchemy import select

from app.agents.state import PatientState
from app.db.models import Alert, SessionRecord
from app.db.session import SessionLocal
from app.services import gemini

log = logging.getLogger(__name__)

WINDOW = 3          # sessions per rolling average
CLEAR_DROP = 0.15   # unambiguous decline
NOISE_DROP = 0.07   # below this, ignore


def _drop(accuracies: list[float]) -> float | None:
    """Prior-window average minus recent-window average. None if too few."""
    if len(accuracies) < WINDOW * 2:
        return None
    recent = accuracies[-WINDOW:]
    prior = accuracies[-WINDOW * 2 : -WINDOW]
    return sum(prior) / WINDOW - sum(recent) / WINDOW


def _judge(borderline: dict[str, float]) -> set[str]:
    """Real decline or noise? One call for every borderline domain."""
    prompt = (
        "A dementia patient's game accuracy dropped slightly in these cognitive"
        " domains, comparing the last 3 sessions to the 3 before:\n"
        + "\n".join(f"- {d}: dropped {v:.0%}" for d, v in borderline.items())
        + "\n\nWhich of these are real decline rather than normal day-to-day"
        " variation? Reply with only the domain names, comma separated, or the"
        " word none."
    )
    try:
        answer = gemini.generate_text(prompt).strip().lower()
    except Exception:
        log.warning("decline judgment call failed; not alerting", exc_info=True)
        return set()  # borderline + no judgment available -> don't alarm anyone
    return {d for d in borderline if d in answer}


async def monitor_node(state: PatientState) -> dict:
    patient_id = state["patient_id"]

    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(SessionRecord)
                .where(SessionRecord.patient_id == patient_id)
                .order_by(SessionRecord.date)
            )
        ).scalars().all()

        by_domain: dict[str, list[float]] = {}
        for row in rows:
            by_domain.setdefault(row.domain, []).append(row.accuracy)

        declining: dict[str, float] = {}
        borderline: dict[str, float] = {}
        for domain, accuracies in by_domain.items():
            drop = _drop(accuracies)
            if drop is None or drop <= NOISE_DROP:
                continue
            (declining if drop > CLEAR_DROP else borderline)[domain] = drop

        if borderline:
            declining.update(
                {d: v for d, v in borderline.items() if d in _judge(borderline)}
            )

        alerts = [
            {
                "domain": domain,
                "message": (
                    f"{domain.capitalize()} performance dropped {drop:.0%} over the"
                    " last few sessions. Worth a check-in with the caretaker."
                ),
            }
            for domain, drop in sorted(declining.items())
        ]

        for alert in alerts:
            db.add(Alert(patient_id=patient_id, **alert))
        await db.commit()

    return {"alerts": alerts}
