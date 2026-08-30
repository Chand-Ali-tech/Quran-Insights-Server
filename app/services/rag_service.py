import os
import re
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.settings import settings
from app.services.cache_service import AYAH_CACHE, hydrate_from_cache

logger = logging.getLogger(__name__)

# ── LangSmith Observability Setup ─────────────────────────────────────────────
# Uses ONLY the lightweight `langsmith` package — no LangChain required.
# Enable by setting LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGCHAIN_API_KEY)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT)
os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGCHAIN_ENDPOINT)

_tracing_enabled = (
    settings.LANGCHAIN_TRACING_V2.lower() == "true"
    and bool(settings.LANGCHAIN_API_KEY)
)

if _tracing_enabled:
    try:
        from langsmith import traceable
        from langsmith.wrappers import wrap_openai
        # wrap_openai patches the client so every embeddings.create() and
        # chat.completions.create() call is automatically traced in LangSmith
        # — no other code changes needed.
        openai_async = wrap_openai(AsyncOpenAI(api_key=settings.OPENAI_API_KEY))
        logger.info(
            "✅ LangSmith tracing ENABLED — dashboard: https://smith.langchain.com"
            " | project: '%s'", settings.LANGCHAIN_PROJECT
        )
    except ImportError:
        logger.warning(
            "⚠️  langsmith package not found. Install it: pip install langsmith"
        )
        def traceable(**kw):           # no-op fallback
            return lambda f: f
        openai_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
else:
    # Tracing disabled — plain client, zero overhead in production
    def traceable(**kw):               # no-op — decorator does nothing
        return lambda f: f
    openai_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info(
        "ℹ️  LangSmith tracing DISABLED "
        "(set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env to enable)"
    )

# ── Qdrant Client (Reused async connection pool) ──────────────────────────────
qdrant_async = AsyncQdrantClient(
    url=settings.QDRANT_CLUSTER_ENDPOINT,
    api_key=settings.QDRANT_API_KEY,
    timeout=15,
)

# ── Language Detection Constants ──────────────────────────────────────────────
URDU_SPECIFIC_CHARS = set("ٹڈڑںےھگپچژ")

URDU_GRAMMAR_WORDS = {
    "کیا", "کیوں", "کیسے", "کب", "کہاں", "کون", "کونسا", "کونسی",
    "ہے", "ہیں", "ہوں", "ہو", "تھا", "تھی", "تھے", "گا", "گی", "گے",
    "کا", "کی", "کے", "کو", "سے", "پر", "میں", "تک", "اور", "نہیں",
    "نہ", "یہ", "وہ", "آپ", "تم", "ہم", "مجھے", "ہمارا", "میرا",
    "بارے", "بتائیں", "بتاؤ", "کریں", "کرو", "چاہئے", "والا", "والی",
    "والے", "شکریہ", "معلومات", "کچھ",
}

# ── Greeting / Small Talk Patterns ───────────────────────────────────────────
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))\b",
    r"^\s*(assalam|assalamu|salam|salaam)(\s+alaikum|\s+o\s+alaikum|\s+u\s+alaikum|\s+o\s+alekom)?\b",
    r"^\s*(how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name)\b",
    r"^\s*(thanks|thank\s+you|thx|bye|goodbye|cya)\b",
    r"^\s*(السلام\s+عليكم|سلام|مرحبا|أهلا|اهلا|كيف\s+حالك|من\s+أنت|من\s+انت|شكرا|مع\s+السلامة|صباح\s+الخير|مساء\s+الخير)\b",
    r"^\s*(اسلام\s+علیکم|السلام\s+علیکم|سلام\s+علیکم|کیسے\s+ہو|کیسے\s+ہیں|آپ\s+کون\s+ہیں|شکریہ|اللہ\s+حافظ|خدا\s+حافظ)\b",
]


def detect_language(query: str) -> str:
    cleaned = query.strip()
    arabic_script_chars = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", cleaned
    )
    total_alpha = len(re.findall(r"\w", cleaned))

    if not total_alpha or (len(arabic_script_chars) / max(1, total_alpha)) < 0.3:
        return "en"

    if any(c in URDU_SPECIFIC_CHARS for c in cleaned):
        return "ur"

    words = set(re.findall(r"[\u0600-\u06FF]+", cleaned))
    if words & URDU_GRAMMAR_WORDS:
        return "ur"

    return "ar"


def is_greeting(query: str) -> bool:
    cleaned = query.strip()
    words = cleaned.split()
    if len(words) > 6:
        return False

    for pattern in GREETING_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True
    return False


@traceable(name="quran-embed-query", run_type="embedding")
async def get_query_embedding(query: str) -> List[float]:
    """Step 2 — Generate OpenAI embedding for the user query.
    LangSmith traces: model, input text, latency, token count.
    """
    print(
        f"🔢 [Step 2] Generating embedding for query using '{settings.EMBEDDING_MODEL}'..."
    )
    response = await openai_async.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=query.strip(),
    )
    vector = response.data[0].embedding
    print(f"   ↳ Embedding generated successfully (dimensions: {len(vector)})")
    return vector


@traceable(name="quran-vector-search", run_type="retriever")
async def search_qdrant(
    query_vector: List[float],
    lang: str,
    limit: int = 10,
    threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """Step 3 — Cosine similarity search in Qdrant Cloud.
    LangSmith traces: language filter, results count, similarity scores.
    """
    print(
        f"🔍 [Step 3] Searching Qdrant (Collection: '{settings.COLLECTION_NAME}', lang: '{lang}', limit: {limit}, threshold: {threshold})..."
    )
    try:
        search_res = await qdrant_async.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="lang", match=MatchValue(value=lang))]
            ),
            limit=limit,
            with_payload=True,
        )

        print(f"   ↳ Qdrant returned {len(search_res.points)} raw matching points:")
        for idx, p in enumerate(search_res.points, 1):
            vid = p.payload.get("verse_id")
            surah = p.payload.get("surah_name_roman")
            passed = (
                "✅ PASS" if p.score >= threshold else "❌ FILTERED (below threshold)"
            )
            print(
                f"      {idx:>2}. Verse: {vid:<7} ({surah:<14}) | Score: {p.score:.4f} -> {passed}"
            )

        # Filter by similarity threshold
        filtered_points = [
            {
                "verse_id": p.payload.get("verse_id"),
                "score": float(p.score),
                "payload": p.payload,
            }
            for p in search_res.points
            if p.score >= threshold
        ]

        top_candidates = filtered_points[:5]
        print(
            f"📊 [Step 4] Candidates passing threshold (>= {threshold}): {len(filtered_points)} (Kept top {len(top_candidates)})"
        )
        return top_candidates
    except Exception as e:
        print(f"❌ [Error] Qdrant search failed: {e}")
        logger.error(f"Error querying Qdrant: {e}")
        return []


async def hydrate_ayahs(
    verse_matches: List[Dict[str, Any]],
    session: Optional[AsyncSession],
    lang: str,
) -> List[Dict[str, Any]]:
    """Step 4 — Hydrate verse metadata from in-memory cache or PostgreSQL.
    Hydrates verses from fast In-Memory Cache (0.01ms).
    Falls back to PostgreSQL if cache is not yet warmed up.
    """
    if not verse_matches:
        return []

    # Try fast In-Memory Cache first
    if AYAH_CACHE:
        print(
            f"⚡ [Step 5] Instant in-memory hydration from cache for {len(verse_matches)} verses (0.01ms)..."
        )
        hydrated = hydrate_from_cache(verse_matches, lang)
        if hydrated:
            for idx, s in enumerate(hydrated, 1):
                print(
                    f"      {idx}. [{s['surah_name_roman']} {s['verse_id']}] (Score: {s['similarity_score']})"
                )
            return hydrated

    # Fallback to PostgreSQL
    if session:
        vids = [m["verse_id"] for m in verse_matches if m.get("verse_id")]
        print(f"🐘 [Step 5] Cache miss — querying PostgreSQL for verse_ids: {vids}...")
        result = await session.execute(
            text("""
                SELECT
                    a.verse_id, a.ayah_number, a.text_arabic, a.text_english, a.text_urdu, a.main_themes,
                    s.number AS surah_number, s.name_arabic AS surah_name_arabic, s.name_english AS surah_name_english,
                    s.name_roman AS surah_name_roman, s.place_of_revelation
                FROM ayahs a
                JOIN surahs s ON s.id = a.surah_id
                WHERE a.verse_id = ANY(:vids)
            """),
            {"vids": vids},
        )
        rows_by_vid = {r["verse_id"]: dict(r) for r in result.mappings().all()}
        hydrated_sources = []
        for match in verse_matches:
            vid = match["verse_id"]
            row = rows_by_vid.get(vid)
            if not row:
                continue
            user_translation = (
                row["text_urdu"]
                if lang == "ur" and row.get("text_urdu")
                else row.get("text_english", "")
            )
            hydrated_sources.append(
                {
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
                }
            )
        return hydrated_sources

    return []


def build_system_prompt(lang: str) -> str:
    lang_instruction = {
        "ur": "You MUST respond strictly in clean, authentic URDU (اردو).",
        "ar": "You MUST respond strictly in classical ARABIC (العربية).",
        "en": "You MUST respond in clear, easy-to-understand ENGLISH.",
    }.get(lang, "You MUST respond in clear, easy-to-understand English.")

    return f"""You are 'Quran Insights Assistant', a wise, authentic, and clear Islamic AI guide.

LANGUAGE REQUIREMENT:
{lang_instruction}

FORMAT & STRUCTURE GUIDELINES:
1. Present your explanation in clear, structured bullet points (•) or numbered key takeaways so it is effortless for the user to read and understand.
2. Bold (**important words**) such as core Quranic concepts, virtues (e.g. **Sabr (Patience)**, **Tawakkul (Trust in Allah)**, **Dhikr (Remembrance)**), key rulings, and spiritual benefits to make them visually prominent and easy to scan.
3. You may begin with a single brief introductory sentence, followed directly by concise, thematic bullet points.
4. Ground each point in the provided Quranic verses from the context, citing the Surah name and Verse ID (e.g. [Surah Al-Baqarah 2:153] or [سورۃ البقرہ 2:153]).
5. Avoid dense walls of paragraph text. Keep each bullet point focused, impactful, and easy to digest.
6. If no specific verses reached the relevance threshold, offer brief, respectful guidance in concise bullet points.
"""


def build_search_query(
    query: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """If follow-up query is brief or refers to prior turns, combine with previous topic for higher vector recall."""
    cleaned = query.strip()
    if not history:
        return cleaned

    words = cleaned.split()
    pronouns = {
        "it", "this", "that", "these", "those", "they", "them",
        "earlier", "previous", "above", "mentioned", "second", "first",
        "last", "more", "explain", "detail", "tell", "what", "how", "why", "about",
        "اس", "ان", "یہ", "وہ", "مزید", "پہلی", "دوسری", "بارے", "بتائیں", "وضاحت",
        "ذلك", "هذا", "هذه", "تلك", "المذكورة", "السابقة", "المزيد", "وضح", "اشرح",
    }
    has_pronoun = any(w.lower().strip("?,.!") in pronouns for w in words)

    if (len(words) <= 8 and has_pronoun) or len(words) <= 3:
        last_user = next(
            (h["content"] for h in reversed(history) if h.get("role") == "user"), None
        )
        if last_user and last_user.strip() != cleaned:
            enriched = f"{last_user.strip()} — {cleaned}"
            return enriched
    return cleaned


def _prepare_prompts(
    query: str,
    lang: str,
    sources: List[Dict[str, Any]],
    is_greeting_query: bool,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system_prompt = build_system_prompt(lang)

    if is_greeting_query:
        user_prompt = f"The user sent: '{query}'. Reply with a brief, warm greeting as Quran Insights Assistant in the same language and invite them to explore Quranic verses."
    elif sources:
        context_blocks = []
        for s in sources:
            context_blocks.append(
                f"- [{s['surah_name_roman']} {s['verse_id']}] {s['text_arabic']} | Translation: {s['translation']}"
            )
        context_text = "\n".join(context_blocks)
        user_prompt = f"User Question: {query}\n\nQuranic Verses Context:\n{context_text}\n\nProvide a structured, easy-to-read explanation in clear bullet points based on the verses above:"
    else:
        user_prompt = f"User Question: {query}\n\nNote: No specific verses reached the relevance threshold. Please provide respectful, concise guidance in bullet points based on the conversation context."

    prompt_messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # Append up to last 6 prior conversation turns for tight multi-turn context
    if history:
        valid_turns = [
            {"role": t["role"], "content": t["content"]}
            for t in history
            if isinstance(t, dict)
            and t.get("role") in ("user", "assistant")
            and t.get("content")
        ]
        prompt_messages.extend(valid_turns[-6:])

    prompt_messages.append({"role": "user", "content": user_prompt})
    return prompt_messages


@traceable(name="quran-generate-answer", run_type="llm")
async def generate_llm_answer(
    query: str,
    lang: str,
    sources: List[Dict[str, Any]],
    is_greeting_query: bool,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Non-streaming generation with multi-turn history and max_tokens cap.
    LangSmith traces: full prompt, model, temperature, token usage, cost, latency.
    """
    messages = _prepare_prompts(query, lang, sources, is_greeting_query, history)
    print(
        f"🤖 [Step 6] Generating concise answer with '{settings.CHAT_MODEL}' (turns={len(messages)})..."
    )

    response = await openai_async.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=550,
    )

    answer = response.choices[0].message.content.strip()
    print(f"   ↳ LLM generated response ({len(answer)} chars).")
    return answer


@traceable(name="quran-stream-answer", run_type="llm")
async def generate_llm_answer_stream(
    query: str,
    lang: str,
    sources: List[Dict[str, Any]],
    is_greeting_query: bool,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    """Streaming generator yielding text chunks with multi-turn history support.
    LangSmith traces: full prompt, model, first-token latency, total tokens streamed.
    """
    messages = _prepare_prompts(query, lang, sources, is_greeting_query, history)
    print(
        f"⚡ [Step 6] Streaming response with '{settings.CHAT_MODEL}' in real-time (turns={len(messages)})..."
    )

    stream = await openai_async.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=550,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
