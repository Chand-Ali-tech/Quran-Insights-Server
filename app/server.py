from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.database import get_session, init_db
from app.routes.ayah import router as ayah_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()   # runs once when the server starts
    yield


app = FastAPI(
    title="Quran Insights API",
    description="API for Quran verse details, tafsir, and RAG context retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ayah_router)


# ── Core routes ───────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def read_root():
    return {"status": "online", "message": "Welcome to Quran Insights API"}


@app.get("/health", tags=["General"])
async def health_check(session: AsyncSession = Depends(get_session)):
    return {"status": "connected"}