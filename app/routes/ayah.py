"""
routes/ayah.py

GET /ayah/{surah_no}/{ayah_no}
    Returns the core details of a single Ayah — surah info, verse texts,
    and main themes. Tafsir/insights excluded for now to keep responses
    lean and fast for the RAG pipeline.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.database import get_session

router = APIRouter(prefix="/ayah", tags=["Ayah"])


# ─────────────────────────────────────────────────────────────────────────────
#  RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class SurahInfo(BaseModel):
    """Surah (chapter) metadata embedded in the Ayah response."""
    number: int
    name_arabic: str
    name_english: str
    name_roman: str
    place_of_revelation: str


class AyahDetailResponse(BaseModel):
    """
    Core detail response for a single Ayah.
    Includes everything needed for RAG context except tafsir
    (tafsir can be added back later via a dedicated endpoint).
    """
    verse_id: str                  # e.g. "2:255"
    ayah_number: int               # position within the Surah
    text_arabic: str               # original Arabic text
    text_english: str              # English translation
    text_urdu: Optional[str]       # Urdu translation
    main_themes: Optional[str]     # e.g. "['Faith', 'Guidance']"
    surah: SurahInfo               # parent Surah details


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{surah_no}/{ayah_no}",
    response_model=AyahDetailResponse,
    summary="Get Ayah details by Surah and Ayah number",
    description=(
        "Returns the core details of a single Ayah: Arabic text, "
        "English and Urdu translations, main themes, and parent Surah info. "
        "Ideal as lightweight context for a RAG pipeline."
    ),
)
async def get_ayah_detail(
    surah_no: int,
    ayah_no: int,
    session: AsyncSession = Depends(get_session),
) -> AyahDetailResponse:
    """
    Fetch a single Ayah with its Surah details.

    Path parameters:
      - surah_no: Surah (chapter) number  (1 – 114)
      - ayah_no:  Ayah (verse) number within that Surah
    """

    result = await session.execute(
        text("""
            SELECT
                a.verse_id,
                a.ayah_number,
                a.text_arabic,
                a.text_english,
                a.text_urdu,
                a.main_themes,

                s.number        AS surah_number,
                s.name_arabic   AS surah_name_arabic,
                s.name_english  AS surah_name_english,
                s.name_roman    AS surah_name_roman,
                s.place_of_revelation
            FROM ayahs a
            JOIN surahs s ON s.id = a.surah_id
            WHERE s.number = :surah_no
              AND a.ayah_number = :ayah_no
        """),
        {"surah_no": surah_no, "ayah_no": ayah_no},
    )

    row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ayah {surah_no}:{ayah_no} not found. "
                   f"Check that the Surah number (1–114) and Ayah number are valid.",
        )

    return AyahDetailResponse(
        verse_id=row["verse_id"],
        ayah_number=row["ayah_number"],
        text_arabic=row["text_arabic"],
        text_english=row["text_english"],
        text_urdu=row["text_urdu"],
        main_themes=row["main_themes"],
        surah=SurahInfo(
            number=row["surah_number"],
            name_arabic=row["surah_name_arabic"],
            name_english=row["surah_name_english"],
            name_roman=row["surah_name_roman"],
            place_of_revelation=row["place_of_revelation"],
        ),
    )
