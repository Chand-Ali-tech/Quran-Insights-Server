import ssl
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.config.settings import settings

db_url = settings.DATABASE_URL

# Normalize URL scheme: asyncpg requires postgresql+asyncpg://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Strip ?sslmode=require from the URL — asyncpg does NOT accept it as a query param.
# SSL is instead passed via connect_args using Python's ssl module.
parsed = urlparse(db_url)
query_params = parse_qs(parsed.query)
query_params.pop("sslmode", None)

clean_query = urlencode({k: v[0] for k, v in query_params.items()})
db_url = urlunparse(parsed._replace(query=clean_query))

# Build SSL context for Neon (requires TLS)
ssl_context = ssl.create_default_context()

# echo=True logs SQL queries — turn off in production
# pool_pre_ping=True prevents disconnect errors on Neon serverless endpoints
engine = create_async_engine(
    db_url,
    echo=True,
    # future=True,
    # pool_pre_ping=True,
    # pool_recycle=300,
    # connect_args={"ssl": ssl_context},
)

async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create tables on startup if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    """FastAPI dependency — yields a DB session per request."""
    async with async_session_maker() as session:
        yield session
