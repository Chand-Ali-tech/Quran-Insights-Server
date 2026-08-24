"""
routes/chat.py

High-performance RAG endpoints:
- POST /chat        → Fast JSON response (~2s) using in-memory cache & capped tokens.
- POST /chat/stream → Real-time streaming response (first token in ~0.8s) via SSE.
"""

import json
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.database import get_session
from app.config.settings import settings
from app.services.rag_service import (
    detect_language,
    is_greeting,
    build_search_query,
    get_query_embedding,
    search_qdrant,
    hydrate_ayahs,
    generate_llm_answer,
    generate_llm_answer_stream,
)

router = APIRouter(prefix="/chat", tags=["Chat & Q&A"])


# ─────────────────────────────────────────────────────────────────────────────
#  REQUEST & RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessageHistory(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="The user's question or message in English, Urdu, or Arabic",
        examples=["What does the Quran say about patience and prayer?"],
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Optional similarity cutoff (0.0 to 1.0). Defaults to server configuration (0.70).",
        ge=0.0,
        le=1.0,
    )
    history: Optional[List[ChatMessageHistory]] = Field(
        default=None,
        description="Prior conversation history turns for multi-turn conversational context.",
    )


class SourceAyah(BaseModel):
    verse_id: str
    surah_number: int
    ayah_number: int
    surah_name_roman: str
    surah_name_english: str
    surah_name_arabic: str
    place_of_revelation: str
    text_arabic: str
    translation: str
    similarity_score: float


class ChatResponse(BaseModel):
    query: str
    detected_language: str
    is_greeting: bool
    answer: str
    sources: List[SourceAyah]


# ─────────────────────────────────────────────────────────────────────────────
#  1. STANDARD JSON ENDPOINT (Fast ~2s Response)
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    summary="Fast JSON response for Quran Q&A",
    description="Returns complete structured response (~2–3s) using in-memory cache and optimized token generation.",
)
async def chat_endpoint(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    start_time = time.perf_counter()
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    print("\n" + "=" * 70)
    print(f'📥 [RAG JSON] Incoming query: "{query_text}"')

    lang = detect_language(query_text)
    greeting = is_greeting(query_text)
    history_dicts = (
        [h.model_dump() for h in request.history] if request.history else None
    )
    print(
        f"🌐 [Step 1] Lang: '{lang}' | is_greeting: {greeting} | History turns: {len(history_dicts) if history_dicts else 0}"
    )

    sources: List[SourceAyah] = []

    if greeting:
        print("⚡ [Fast-Path] Greeting detected — bypassing vector DB.")
        answer = await generate_llm_answer(
            query=query_text,
            lang=lang,
            sources=[],
            is_greeting_query=True,
            history=history_dicts,
        )
    else:
        try:
            search_text = build_search_query(query_text, history_dicts)
            query_vector = await get_query_embedding(search_text)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to generate embedding: {str(e)}",
            )

        threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )
        qdrant_matches = await search_qdrant(
            query_vector=query_vector,
            lang=lang,
            limit=10,
            threshold=threshold,
        )

        if qdrant_matches:
            hydrated = await hydrate_ayahs(
                verse_matches=qdrant_matches,
                session=session,
                lang=lang,
            )
            sources = [SourceAyah(**item) for item in hydrated]

        answer = await generate_llm_answer(
            query=query_text,
            lang=lang,
            sources=[s.model_dump() for s in sources],
            is_greeting_query=False,
            history=history_dicts,
        )

    elapsed = time.perf_counter() - start_time
    print(f"⏱️  [JSON Completed] Total time: {elapsed:.2f}s | Sources: {len(sources)}")
    print("=" * 70 + "\n")

    return ChatResponse(
        query=query_text,
        detected_language=lang,
        is_greeting=greeting,
        answer=answer,
        sources=sources,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  2. REAL-TIME STREAMING ENDPOINT (First Token in ~0.8s)
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/stream",
    summary="Real-time Server-Sent Events (SSE) streaming endpoint",
    description=(
        "Streams response chunks word-by-word in real-time.\n"
        "Emits:\n"
        "- `sources`: Metadata and citations event\n"
        "- `token`: Content delta chunks\n"
        "- `done`: Stream complete event"
    ),
)
async def chat_stream_endpoint(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    lang = detect_language(query_text)
    greeting = is_greeting(query_text)
    history_dicts = (
        [h.model_dump() for h in request.history] if request.history else None
    )

    sources_list: List[dict] = []

    if not greeting:
        search_text = build_search_query(query_text, history_dicts)
        query_vector = await get_query_embedding(search_text)
        threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )
        qdrant_matches = await search_qdrant(
            query_vector=query_vector,
            lang=lang,
            limit=10,
            threshold=threshold,
        )
        if qdrant_matches:
            sources_list = await hydrate_ayahs(
                verse_matches=qdrant_matches,
                session=session,
                lang=lang,
            )

    async def event_generator():
        # 1. Send metadata & sources as the first SSE event
        meta_event = {
            "type": "metadata",
            "query": query_text,
            "detected_language": lang,
            "is_greeting": greeting,
            "sources": sources_list,
        }
        yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

        # 2. Stream tokens chunk-by-chunk
        async for token in generate_llm_answer_stream(
            query=query_text,
            lang=lang,
            sources=sources_list,
            is_greeting_query=greeting,
            history=history_dicts,
        ):
            token_event = {"type": "token", "content": token}
            yield f"data: {json.dumps(token_event, ensure_ascii=False)}\n\n"

        # 3. Final completion event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
