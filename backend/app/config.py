"""Settings read straight from the environment.

No dotenv loading in-app: run with `uv run --env-file .env ...`.
"""

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./smritiyog.db")
AUTH_SECRET = os.getenv("AUTH_SECRET", "")
ENV = os.getenv("ENV", "development")

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_DIM = 768

# Rounds pre-generated per /sync call.
BATCH_SIZE = 12
