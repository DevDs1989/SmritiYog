import os
import pathlib

# Must be set before anything imports app.config.
_TEST_DB = pathlib.Path(__file__).parent / "test_smritiyog.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB}"
os.environ["AUTH_SECRET"] = "test-secret"
os.environ["GEMINI_API_KEY"] = "fake-key"

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.session import SessionLocal, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import gemini  # noqa: E402

AUTH = {"Authorization": "Bearer test-secret"}


@pytest.fixture(autouse=True)
async def fresh_db():
    _TEST_DB.unlink(missing_ok=True)
    await init_db()
    yield
    await engine.dispose()
    _TEST_DB.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def no_gemini(monkeypatch):
    """Never hit the real API in tests. Empty JSON array from generate_text
    exercises the template fallback path in the content agent."""
    monkeypatch.setattr(gemini, "generate_text", lambda prompt: "[]")
    monkeypatch.setattr(gemini, "embed_batch", lambda texts: [[1.0] * 8 for _ in texts])


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db_session():
    return SessionLocal
