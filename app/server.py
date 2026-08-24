from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.database import get_session, init_db
from app.services.cache_service import init_ayah_cache
from app.routes.ayah import router as ayah_router
from app.routes.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize database tables
    await init_db()
    # 2. Warm up in-memory Ayah cache for sub-second RAG lookups
    await init_ayah_cache()
    yield


app = FastAPI(
    title="Quran Insights API",
    description="API for Quran verse details, tafsir, and high-performance AI-powered RAG Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(ayah_router)


# ── Core routes ───────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def read_root():
    return {
        "status": "online",
        "message": "Welcome to Quran Insights API",
        "endpoints": {
            "chat_json": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "ayah": "GET /ayah/{surah_no}/{ayah_no}",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["General"])
async def health_check(session: AsyncSession = Depends(get_session)):
    return {"status": "connected"}