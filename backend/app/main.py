from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import dashboard, onboarding, recall, sync
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="SmritiYog Backend", lifespan=lifespan)
app.include_router(sync.router, tags=["sync"])
app.include_router(onboarding.router, tags=["onboarding"])
app.include_router(recall.router, tags=["recall"])
app.include_router(dashboard.router, tags=["dashboard"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
