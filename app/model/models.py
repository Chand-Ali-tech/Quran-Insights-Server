"""
Database models for Quran Insights.

Dataset shape (quran_dataset_final.json):
- 18,450 total records
- 6,236 unique verses (verse_id like "1:1", "2:255" etc.)
- Each verse can have multiple records — one per audience_group (up to 4)
- Fields per record:
    surah_no, surah_name_en, surah_name_ar, surah_name_roman,
    ayah_no_surah, ayah_ar, ayah_en, place_of_revelation, ayah_ur,
    main_themes, tafsir, audience_group, verse_id

Schema design:
    Surah  (1) ──< (many)  Ayah  (1) ──< (many)  AyahInsight
"""

from datetime import datetime
from typing import List, Optional  # noqa: UP035

from sqlmodel import Field, Relationship, SQLModel


# ─────────────────────────────────────────
#  SURAH  (114 rows total)
# ─────────────────────────────────────────
class Surah(SQLModel, table=True):
    __tablename__ = "surahs"

    id: Optional[int] = Field(default=None, primary_key=True)

    number: int = Field(index=True, unique=True)  # e.g. 1, 2 … 114
    name_arabic: str  # e.g. "الفاتحة"
    name_english: str  # e.g. "The Opener"
    name_roman: str  # e.g. "Al-Fatihah"
    place_of_revelation: str  # "Meccan" or "Medinan"

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship: a Surah has many Ayahs
    ayahs: List["Ayah"] = Relationship(back_populates="surah")


# ─────────────────────────────────────────
#  AYAH  (6,236 unique verses)
# ─────────────────────────────────────────
class Ayah(SQLModel, table=True):
    __tablename__ = "ayahs"

    id: Optional[int] = Field(default=None, primary_key=True)

    verse_id: str = Field(index=True, unique=True)  # e.g. "1:1"
    ayah_number: int  # position inside Surah, e.g. 1
    text_arabic: str  # Arabic text of the verse
    text_english: str  # English translation
    text_urdu: Optional[str] = Field(default=None)  # Urdu translation
    main_themes: Optional[str] = Field(
        default=None
    )  # stored as stringified list e.g. "['Faith', 'Worship']"

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Foreign key → Surah
    surah_id: int = Field(foreign_key="surahs.id", index=True)
    surah: Optional[Surah] = Relationship(back_populates="ayahs")

    # Relationship: an Ayah has many AyahInsights (one per audience_group)
    insights: List["AyahInsight"] = Relationship(back_populates="ayah")


# ─────────────────────────────────────────
#  AYAH INSIGHT  (18,450 rows — ~3 per verse)
# ─────────────────────────────────────────
class AyahInsight(SQLModel, table=True):
    __tablename__ = "ayah_insights"

    id: Optional[int] = Field(default=None, primary_key=True)

    audience_group: str = Field(index=True)  # e.g. "Non-Muslims"
    tafsir: str  # Long-form explanation / tafsir text

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Foreign key → Ayah
    ayah_id: int = Field(foreign_key="ayahs.id", index=True)
    ayah: Optional[Ayah] = Relationship(back_populates="insights")


"""
SURAH's table :-
    One row per Surah (chapter) of the Quran.
    There are 114 Surahs total.
    

AYAH's table :- 
    One row per unique Ayah (verse) across all 114 Surahs.
    verse_id is the human-readable key e.g. "1:1", "2:255".
    

AYAH INSIGHT's table :- 
    One row per (Ayah × audience_group) combination.
    Each verse has multiple tafsir/insights tailored to different audiences.

    audience_group values in the dataset:
        'General Muslim Community', 'Non-Muslims', 'Scholars & Academics',
        'Spiritual Seekers', 'Students & Learners', 'Self-Improvement Seekers',
        'Facing Adversity/Hardship', 'Anxiety & Mental Health',
        'Inner Conflict & Peace', 'Existential Questions',
        'Struggling with Doubt/Faith', 'Truth & Meaning Seekers',
        'Moral & Ethical Guidance', 'Leaders & Decision Makers',
        'Social Justice & Community', 'Muslim Gender-Specific',
        'Islamic Studies Specific', 'Specific Professions',
        'Universal/All Humanity', 'Other'
"""
