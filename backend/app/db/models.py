"""ORM models. Kept Postgres-compatible: no SQLite-only types or defaults, so
moving to Neon is a DATABASE_URL + driver change.

Deliberately no table for rag_delta -- family RAG data is request-scoped only.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    consent_flag: Mapped[bool] = mapped_column(Boolean, default=False)


class SessionRecord(Base):
    """Server-side copy of the app's session history, for cross-sync trends."""

    __tablename__ = "session_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), index=True
    )
    game_type: Mapped[str] = mapped_column(String(32))
    domain: Mapped[str] = mapped_column(String(32), index=True)
    accuracy: Mapped[float] = mapped_column(Float)
    response_time_ms: Mapped[int] = mapped_column(Integer)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.id"), index=True
    )
    domain: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
