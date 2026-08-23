"""
seed.py — Loads quran_dataset_final.json into the Neon PostgreSQL database.

Run from the project root (server/) with:
    python -m app.seed

What it does:
    1. Creates all tables (surahs, ayahs, ayah_insights) if they don't exist.
    2. Reads quran_dataset_final.json (18,450 records).
    3. Inserts Surahs first (114 unique chapters).
    4. Inserts Ayahs next   (6,236 unique verses).
    5. Inserts AyahInsights (18,450 rows, one per verse × audience_group).
    6. Skips records that are already in the database (safe to re-run).
"""

import asyncio
import json
import time
from pathlib import Path

from sqlalchemy import text
from sqlmodel import SQLModel

from app.config.database import async_session_maker, engine

# ── Import models so their tables get registered with SQLModel.metadata ──────
from app.model.models import Ayah, AyahInsight, Surah  # noqa: F401

# ── Path to the dataset ───────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "quran_dataset_final.json"

# ── How many rows to INSERT per database round-trip (tune if needed) ─────────
BATCH_SIZE = 200


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    """Read and return the JSON dataset as a list of records."""
    print(f"📂  Loading dataset from {DATASET_PATH} …")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅  {len(data):,} records loaded.\n")
    return data


def build_surahs(records: list[dict]) -> list[dict]:
    """Deduplicate and return one dict per unique Surah."""
    seen = {}
    for r in records:
        n = r["surah_no"]
        if n not in seen:
            seen[n] = {
                "number": n,
                "name_arabic": r["surah_name_ar"],
                "name_english": r["surah_name_en"],
                "name_roman": r["surah_name_roman"],
                "place_of_revelation": r["place_of_revelation"],
            }
    return list(seen.values())


def build_ayahs(records: list[dict]) -> list[dict]:
    """Deduplicate and return one dict per unique verse_id."""
    seen = {}
    for r in records:
        vid = r["verse_id"]
        if vid not in seen:
            seen[vid] = {
                "verse_id": vid,
                "surah_number": r["surah_no"],   # used to look up surah_id later
                "ayah_number": r["ayah_no_surah"],
                "text_arabic": r["ayah_ar"],
                "text_english": r["ayah_en"],
                "text_urdu": r.get("ayah_ur"),
                "main_themes": r.get("main_themes"),
            }
    return list(seen.values())


def build_insights(records: list[dict]) -> list[dict]:
    """Return one dict per record (each is a unique verse × audience_group row)."""
    return [
        {
            "verse_id": r["verse_id"],           # used to look up ayah_id later
            "audience_group": r["audience_group"],
            "tafsir": r["tafsir"],
        }
        for r in records
    ]


async def create_tables() -> None:
    """Create all SQLModel tables if they don't already exist."""
    print("🏗️   Creating tables …")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    print("✅  Tables ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  INSERT HELPERS  (batch insert with ON CONFLICT DO NOTHING)
# ─────────────────────────────────────────────────────────────────────────────

async def insert_surahs(session, surah_rows: list[dict]) -> dict[int, int]:
    """
    Insert Surahs and return a mapping  {surah_number → surah.id}.
    Uses ON CONFLICT DO NOTHING so re-runs are safe.
    """
    print(f"📖  Inserting {len(surah_rows)} Surahs …")

    for i in range(0, len(surah_rows), BATCH_SIZE):
        batch = surah_rows[i : i + BATCH_SIZE]
        await session.execute(
            text("""
                INSERT INTO surahs (number, name_arabic, name_english, name_roman, place_of_revelation, created_at)
                VALUES (:number, :name_arabic, :name_english, :name_roman, :place_of_revelation, NOW())
                ON CONFLICT (number) DO NOTHING
            """),
            batch,
        )

    await session.commit()

    # Fetch id mapping
    result = await session.execute(text("SELECT number, id FROM surahs"))
    mapping = {row.number: row.id for row in result}
    print(f"✅  {len(mapping)} Surahs in database.\n")
    return mapping


async def insert_ayahs(session, ayah_rows: list[dict], surah_id_map: dict[int, int]) -> dict[str, int]:
    """
    Insert Ayahs and return a mapping  {verse_id → ayah.id}.
    Uses ON CONFLICT DO NOTHING so re-runs are safe.
    """
    print(f"📜  Inserting {len(ayah_rows):,} Ayahs …")

    # Attach surah_id to each row
    prepared = []
    for row in ayah_rows:
        surah_id = surah_id_map.get(row["surah_number"])
        if surah_id is None:
            print(f"  ⚠️  Skipping ayah {row['verse_id']} — surah {row['surah_number']} not found.")
            continue
        prepared.append({
            "verse_id": row["verse_id"],
            "ayah_number": row["ayah_number"],
            "text_arabic": row["text_arabic"],
            "text_english": row["text_english"],
            "text_urdu": row.get("text_urdu"),
            "main_themes": row.get("main_themes"),
            "surah_id": surah_id,
        })

    for i in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[i : i + BATCH_SIZE]
        await session.execute(
            text("""
                INSERT INTO ayahs (verse_id, ayah_number, text_arabic, text_english, text_urdu, main_themes, surah_id, created_at)
                VALUES (:verse_id, :ayah_number, :text_arabic, :text_english, :text_urdu, :main_themes, :surah_id, NOW())
                ON CONFLICT (verse_id) DO NOTHING
            """),
            batch,
        )
        inserted = i + len(batch)
        print(f"    … {inserted:,} / {len(prepared):,}", end="\r")

    await session.commit()

    # Fetch id mapping
    result = await session.execute(text("SELECT verse_id, id FROM ayahs"))
    mapping = {row.verse_id: row.id for row in result}
    print(f"\n✅  {len(mapping):,} Ayahs in database.\n")
    return mapping


async def insert_insights(session, insight_rows: list[dict], ayah_id_map: dict[str, int]) -> None:
    """
    Insert AyahInsights.
    Uses ON CONFLICT DO NOTHING so re-runs are safe.
    Requires a unique constraint on (ayah_id, audience_group) — added via raw DDL below.
    """
    print(f"💡  Inserting {len(insight_rows):,} AyahInsights …")

    # Ensure the unique constraint exists (idempotent)
    await session.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_ayah_insights_ayah_audience'
            ) THEN
                ALTER TABLE ayah_insights
                ADD CONSTRAINT uq_ayah_insights_ayah_audience
                UNIQUE (ayah_id, audience_group);
            END IF;
        END$$;
    """))
    await session.commit()

    prepared = []
    skipped = 0
    for row in insight_rows:
        ayah_id = ayah_id_map.get(row["verse_id"])
        if ayah_id is None:
            skipped += 1
            continue
        prepared.append({
            "ayah_id": ayah_id,
            "audience_group": row["audience_group"],
            "tafsir": row["tafsir"],
        })

    if skipped:
        print(f"  ⚠️  Skipped {skipped} insight rows with missing verse_id.")

    for i in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[i : i + BATCH_SIZE]
        await session.execute(
            text("""
                INSERT INTO ayah_insights (ayah_id, audience_group, tafsir, created_at)
                VALUES (:ayah_id, :audience_group, :tafsir, NOW())
                ON CONFLICT ON CONSTRAINT uq_ayah_insights_ayah_audience DO NOTHING
            """),
            batch,
        )
        inserted = i + len(batch)
        print(f"    … {inserted:,} / {len(prepared):,}", end="\r")

    await session.commit()
    print(f"\n✅  AyahInsights seeded.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def seed() -> None:
    start = time.perf_counter()

    # 1. Ensure tables exist
    await create_tables()

    # 2. Load JSON
    records = load_dataset()

    # 3. Derive rows for each table
    surah_rows = build_surahs(records)
    ayah_rows = build_ayahs(records)
    insight_rows = build_insights(records)

    # 4. Insert in order: Surah → Ayah → AyahInsight
    async with async_session_maker() as session:
        surah_id_map = await insert_surahs(session, surah_rows)
        ayah_id_map = await insert_ayahs(session, ayah_rows, surah_id_map)
        await insert_insights(session, insight_rows, ayah_id_map)

    elapsed = time.perf_counter() - start
    print(f"🎉  Seeding complete in {elapsed:.1f}s")
    print(f"    Surahs    : {len(surah_id_map):>6,}")
    print(f"    Ayahs     : {len(ayah_id_map):>6,}")
    print(f"    Insights  : {len(insight_rows):>6,}")


if __name__ == "__main__":
    asyncio.run(seed())

