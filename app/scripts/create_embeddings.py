"""
scripts/create_embeddings.py
────────────────────────────
Creates OpenAI text-embedding-3-large embeddings for every unique Ayah
in 3 languages (Arabic, English, Urdu) and uploads them to Qdrant.

Each Ayah → 3 Qdrant points:
    verse_id="1:1", lang="ar"  →  point id: uuid5("1:1_ar")
    verse_id="1:1", lang="en"  →  point id: uuid5("1:1_en")
    verse_id="1:1", lang="ur"  →  point id: uuid5("1:1_ur")

Total points: 6,236 unique Ayahs × 3 = 18,708

Features:
  - Checkpoint file: tracks completed verse_ids so re-runs skip already
    embedded ayahs (saves OpenAI API cost after interruption).
  - Retry logic: Qdrant upsert retried up to 5× with backoff.
  - Configurable timeout on Qdrant client (default 60s).

Run from the server/ directory:
    python -m app.scripts.create_embeddings
"""

import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# ── Load env vars ─────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_SIZE     = 3072          # text-embedding-3-large output dimensions
COLLECTION_NAME = "quran_ayahs"

DATASET_PATH    = Path(__file__).parent.parent / "data" / "quran_dataset_final.json"

# Checkpoint file — stores verse_ids that have been successfully uploaded.
# Allows the script to resume after a crash without re-embedding.
CHECKPOINT_PATH = Path(__file__).parent / ".embed_checkpoint.json"

EMBED_BATCH  = 10   # ayahs per OpenAI API call (× 3 langs = 30 texts per call)
UPLOAD_BATCH = 10   # Qdrant points per upsert (small = safer with large vectors)
RETRY_DELAY  = 5    # base seconds for retry backoff

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# timeout=60: allow up to 60s per Qdrant request (3072-dim vectors are large)
qdrant = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT  (resume after crash)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Return set of verse_ids already successfully uploaded."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        return set(data.get("done", []))
    return set()


def save_checkpoint(done: set[str]) -> None:
    """Persist the set of completed verse_ids to disk."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"done": list(done)}, f)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_point_id(verse_id: str, lang: str) -> str:
    """
    Deterministic UUID for a (verse_id, lang) pair.
    Same input always gives the same UUID — makes re-runs safe.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{verse_id}_{lang}"))


def build_embed_texts(ayah: dict) -> dict[str, str]:
    """
    Build the 3 text strings to be converted into vectors.
    Surah name + themes are included so the vector captures full context.
    """
    surah_roman  = ayah["surah_name_roman"]
    surah_en     = ayah["surah_name_en"]
    ayah_no      = ayah["ayah_no_surah"]
    themes       = ayah.get("main_themes", "")
    arabic_text  = ayah.get("ayah_ar", "")
    english_text = ayah.get("ayah_en", "")
    urdu_text    = ayah.get("ayah_ur", "")

    return {
        "ar": f"{arabic_text} الموضوعات: {themes}",
        "en": (
            f"Surah {surah_roman} ({surah_en}), Ayah {ayah_no}. "
            f"Translation: {english_text} Themes: {themes}"
        ),
        "ur": (
            f"سورۃ {surah_roman}، آیت {ayah_no}۔ "
            f"ترجمہ: {urdu_text} موضوعات: {themes}"
        ),
    }


def build_payload(ayah: dict, lang: str) -> dict:
    """
    Minimal Qdrant payload — just enough to identify the verse after search.
    Full verse text is fetched from PostgreSQL using verse_id.
    """
    return {
        "lang":                lang,
        "verse_id":            ayah["verse_id"],
        "ayah_number":         ayah["ayah_no_surah"],
        "surah_number":        ayah["surah_no"],
        "surah_name_english":  ayah["surah_name_en"],
        "surah_name_arabic":   ayah["surah_name_ar"],
        "surah_name_roman":    ayah["surah_name_roman"],
        "place_of_revelation": ayah["place_of_revelation"],
    }


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI embeddings API with retry on rate-limit errors.
    Returns a list of vectors in the same order as texts.
    """
    for attempt in range(5):
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"\n  ⚠️  OpenAI rate limit. Waiting {wait}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI embeddings failed after 5 retries.")


def upsert_with_retry(points: list[PointStruct]) -> None:
    """Upload points to Qdrant with exponential backoff retry."""
    for attempt in range(5):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            return  # success
        except Exception as e:
            if attempt < 4:
                wait = RETRY_DELAY * (attempt + 1)
                print(
                    f"\n  ⚠️  Qdrant error (attempt {attempt + 1}/5): "
                    f"{type(e).__name__}. Retrying in {wait}s …"
                )
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
#  QDRANT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"✅  Collection '{COLLECTION_NAME}' already exists.\n")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(
        f"🏗️   Created Qdrant collection '{COLLECTION_NAME}' "
        f"(dims={VECTOR_SIZE}, metric=cosine).\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Quran Insights — Embedding Script")
    print(f"  Model  : {EMBEDDING_MODEL}  ({VECTOR_SIZE} dims)")
    print(f"  Target : Qdrant → '{COLLECTION_NAME}'")
    print("=" * 60, "\n")

    # 1. Ensure Qdrant collection
    ensure_collection()

    # 2. Load dataset
    print("📂  Loading dataset …")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"✅  {len(records):,} records loaded.\n")

    # 3. Deduplicate to unique Ayahs
    seen: dict[str, dict] = {}
    for r in records:
        vid = r["verse_id"]
        if vid not in seen:
            seen[vid] = r
    unique_ayahs = list(seen.values())
    print(f"📜  {len(unique_ayahs):,} unique Ayahs found.")

    # 4. Load checkpoint — skip already-processed ayahs
    completed = load_checkpoint()
    remaining = [a for a in unique_ayahs if a["verse_id"] not in completed]
    skipped = len(unique_ayahs) - len(remaining)
    if skipped:
        print(f"⏭️   Skipping {skipped:,} already-embedded Ayahs (checkpoint).")
    print(f"⚡  Embedding {len(remaining):,} remaining Ayahs × 3 languages = "
          f"{len(remaining) * 3:,} vectors.\n")

    if not remaining:
        print("🎉  All Ayahs already embedded. Nothing to do!")
        return

    LANGS         = ["ar", "en", "ur"]
    points_buffer : list[PointStruct] = []
    done          = 0
    start         = time.perf_counter()

    for i in range(0, len(remaining), EMBED_BATCH):
        batch_ayahs = remaining[i : i + EMBED_BATCH]

        # Build flat list of texts: [a0_ar, a0_en, a0_ur, a1_ar, a1_en, a1_ur, ...]
        texts_flat: list[str] = []
        for ayah in batch_ayahs:
            embed_texts = build_embed_texts(ayah)
            for lang in LANGS:
                texts_flat.append(embed_texts[lang])

        # Embed with OpenAI
        vectors = get_embeddings(texts_flat)

        # Build Qdrant points
        for j, ayah in enumerate(batch_ayahs):
            for k, lang in enumerate(LANGS):
                vector = vectors[j * len(LANGS) + k]
                points_buffer.append(
                    PointStruct(
                        id=make_point_id(ayah["verse_id"], lang),
                        vector=vector,
                        payload=build_payload(ayah, lang),
                    )
                )

        done += len(batch_ayahs)

        # Upload when buffer is full or at the end
        if len(points_buffer) >= UPLOAD_BATCH or i + EMBED_BATCH >= len(remaining):
            upsert_with_retry(points_buffer)
            points_buffer.clear()

            # Save checkpoint after each successful upload
            for ayah in batch_ayahs:
                completed.add(ayah["verse_id"])
            save_checkpoint(completed)

        # Progress
        elapsed = time.perf_counter() - start
        pct     = (done / len(remaining)) * 100
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (len(remaining) - done) / rate if rate > 0 else 0
        print(
            f"  [{pct:5.1f}%]  {done:,}/{len(remaining):,} ayahs  |  "
            f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s",
            end="\r",
        )

    elapsed = time.perf_counter() - start
    print(f"\n\n🎉  Done in {elapsed:.1f}s")
    print(f"    Ayahs embedded  : {len(unique_ayahs):,}")
    print(f"    Total vectors   : {len(unique_ayahs) * 3:,}")
    print(f"    Collection      : '{COLLECTION_NAME}'")
    print(f"    Model           : {EMBEDDING_MODEL}")

    # Clean up checkpoint on success
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("    Checkpoint file : deleted (run complete)")


if __name__ == "__main__":
    main()
