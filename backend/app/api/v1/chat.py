from fastapi import APIRouter, status

from ...schemas import ChatRequest, ChatResponse, CitationSchema
from ..deps import GenerationServiceDep

chat_router = APIRouter(prefix="/chat", tags=["Chat"])


@chat_router.post(
    "",
    status_code=status.HTTP_200_OK,
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    generation_service: GenerationServiceDep,
) -> ChatResponse:
    """RAG chat: retrieve context, build the prompt, ask the LLM, return the answer.

    Pipeline: query -> retrieval -> PromptBuilder -> Groq -> answer + citations.
    Streaming, memory, multi-turn, history, sessions, and tool calling are
    explicitly out of scope for this endpoint (Sprint 9).
    """
    result = await generation_service.generate(request.query, top_k=request.top_k)

    citations = [
        CitationSchema(
            document_id=citation.document_id,
            chunk_id=citation.chunk_id,
            page_number=citation.page_number,
            score=citation.score,
        )
        for citation in result.citations
    ]

    return ChatResponse(
        query=request.query,
        answer=result.answer,
        model=result.model,
        total_citations=len(citations),
        citations=citations,
    )
