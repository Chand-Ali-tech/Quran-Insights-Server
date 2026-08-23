from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.database import get_session, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()   # runs once when the server starts
    yield

app = FastAPI(
    title="Quran Insights API",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check(session: AsyncSession = Depends(get_session)):
    return {"status": "connected"}


@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to Quran Insights API"}