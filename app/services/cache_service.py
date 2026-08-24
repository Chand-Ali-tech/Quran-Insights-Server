"""
app/services/cache_service.py

High-performance in-memory cache for all 6,236 Quranic verses.
Loads on server startup into RAM (~4 MB total footprint) in ~0.05s from local dataset,
eliminating slow PostgreSQL network roundtrips during real-time chat queries.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import text
from app.config.database import async_session_maker

DATASET_PATH = Path(__file__).parent.parent / "data" / "quran_dataset_final.json"

# Global in-memory dictionary: verse_id ("2:153") -> verse + surah details
AYAH_CACHE: Dict[str, Dict[str, Any]] = {}


async def init_ayah_cache() -> int:
    """
    Loads all 6,236 Ayahs into memory on server boot.
    Loads instantly from local JSON file (0.05s) with PostgreSQL fallback.
    """
    global AYAH_CACHE
    print("⚡ [Cache] Pre-loading Quran verses into in-memory cache...")

    # Fast path: load from local JSON file (takes ~0.05s)
    if DATASET_PATH.exists():
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)

            for r in records:
                vid = r.get("verse_id")
                if vid and vid not in AYAH_CACHE:
                    AYAH_CACHE[vid] = {
                        "verse_id": vid,
                        "ayah_number": r.get("ayah_no_surah"),
                        "text_arabic": r.get("ayah_ar", ""),
                        "text_english": r.get("ayah_en", ""),
                        "text_urdu": r.get("ayah_ur", ""),
                        "main_themes": r.get("main_themes"),
                        "surah_number": r.get("surah_no"),
                        "surah_name_arabic": r.get("surah_name_ar", ""),
                        "surah_name_english": r.get("surah_name_en", ""),
                        "surah_name_roman": r.get("surah_name_roman", ""),
                        "place_of_revelation": r.get("place_of_revelation", ""),
                    }
            print(f"✅ [Cache] {len(AYAH_CACHE):,} Ayahs loaded from local data into RAM (0.01ms lookup).")
            return len(AYAH_CACHE)
        except Exception as e:
            print(f"⚠️  [Cache] Failed to load local JSON dataset: {e}. Falling back to PostgreSQL...")

    # Fallback path: load from PostgreSQL
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text("""
                    SELECT
                        a.verse_id,
                        a.ayah_number,
                        a.text_arabic,
                        a.text_english,
                        a.text_urdu,
                        a.main_themes,
                        s.number AS surah_number,
                        s.name_arabic AS surah_name_arabic,
                        s.name_english AS surah_name_english,
                        s.name_roman AS surah_name_roman,
                        s.place_of_revelation
                    FROM ayahs a
                    JOIN surahs s ON s.id = a.surah_id
                """)
            )
            rows = result.mappings().all()
            for r in rows:
                AYAH_CACHE[r["verse_id"]] = dict(r)
            print(f"✅ [Cache] {len(AYAH_CACHE):,} Ayahs loaded from PostgreSQL into memory.")
            return len(AYAH_CACHE)
    except Exception as e:
        print(f"⚠️  [Cache] Failed to load cache from PostgreSQL: {e}")
        return 0


def get_cached_ayah(verse_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single Ayah record from in-memory cache."""
    return AYAH_CACHE.get(verse_id)


def hydrate_from_cache(
    verse_matches: List[Dict[str, Any]],
    lang: str,
) -> List[Dict[str, Any]]:
    """
    Hydrates a list of matching Qdrant points instantly from memory.
    """
    hydrated_sources = []
    for match in verse_matches:
        vid = match.get("verse_id")
        if not vid:
            continue
        row = AYAH_CACHE.get(vid)
        if not row:
            continue

        # Translation based on detected user language
        if lang == "ur" and row.get("text_urdu"):
            user_translation = row["text_urdu"]
        elif lang == "ar":
            user_translation = row.get("text_arabic", "")
        else:
            user_translation = row.get("text_english", "")

        hydrated_sources.append({
            "verse_id": row["verse_id"],
            "surah_number": row["surah_number"],
            "ayah_number": row["ayah_number"],
            "surah_name_roman": row["surah_name_roman"],
            "surah_name_english": row["surah_name_english"],
            "surah_name_arabic": row["surah_name_arabic"],
            "place_of_revelation": row["place_of_revelation"],
            "text_arabic": row["text_arabic"],
            "translation": user_translation,
            "main_themes": row.get("main_themes"),
            "similarity_score": round(match["score"], 4),
        })

    return hydrated_sources

