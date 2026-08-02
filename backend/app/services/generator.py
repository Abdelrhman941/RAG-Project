"""
GenerationService — orchestration only.

Flow:
    question -> retrieve() -> PromptBuilder -> provider.generate() -> GenerationResult

Hard rules enforced here:
- No HTTP concerns (no FastAPI imports, no status codes, no request/response models).
- No Groq SDK / provider internals — only BaseLLMProvider.
- No retrieval logic — delegates fully to RetrievalServicePort.
- No prompt template logic — delegates fully to PromptBuilderPort.
- Chunk order from retrieval is preserved end to end (citations mirror it).
- Citations are built only from fields the retrieved chunk actually has
    (document_id, chunk_id, page_number, score) — nothing invented.
- Vectors are never touched/exposed (RetrievedChunk protocol doesn't expose one).

Phase 4 changes
----------------
1. `generate()` now accepts an optional per-call `top_k` override.
    `/chat` lets clients pick `top_k` per request (mirroring the
    retrieval endpoint); the constructor default is just a fallback for
    callers that don't care.

2. `_to_provider_messages()` — bug fix. `PromptBuilder.build()` returns
    `generation.models.ChatMessage` (frozen Pydantic models), but
    `BaseLLMProvider.generate()` — and therefore `GroqProvider` — expects
    `llms.base.ChatMessage` (a plain TypedDict). Handing Pydantic model
    instances straight to the Groq SDK breaks at JSON-serialization time
    (a BaseModel isn't a dict). This is exactly the kind of mismatch that
    stays invisible until PromptBuilder and GroqProvider are actually
    wired together end to end — which only happens now, in Phase 4.
    The conversion also accepts plain dicts unchanged, so existing test
    doubles that already hand back `{"role": ..., "content": ...}` keep
    working with no changes required.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import LLMProviderError
from ..generation import Citation, GenerationResult
from ..generation.ports import PromptBuilderPort, RetrievalServicePort, RetrievedChunk
from ..llms.base import BaseLLMProvider
from ..llms.base import ChatMessage as ProviderChatMessage

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        retrieval_service: RetrievalServicePort,
        prompt_builder: PromptBuilderPort,
        llm_provider: BaseLLMProvider,
        top_k: int = 5,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._top_k = top_k

    async def generate(
        self, question: str, *, top_k: int | None = None
    ) -> GenerationResult:
        effective_top_k = top_k if top_k is not None else self._top_k
        chunks = await self._retrieval_service.retrieve(question, top_k=effective_top_k)
        if not chunks:
            logger.info("No chunks retrieved for question: %r", question)
        messages = self._prompt_builder.build(question, chunks=chunks)
        provider_messages = self._to_provider_messages(messages)
        try:
            answer = await self._llm_provider.generate(provider_messages)
        except LLMProviderError:
            logger.exception("LLM generation failed for question: %r", question)
            raise
        return GenerationResult(
            answer=answer,
            citations=self._build_citations(chunks),
            model=self._llm_provider.model_name,
        )

    @staticmethod
    def _build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
        """One citation per retrieved chunk, in the exact order retrieval gave us."""
        return [
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                score=chunk.score,
            )
            for chunk in chunks
        ]

    @staticmethod
    def _to_provider_messages(messages: Sequence[Any]) -> list[ProviderChatMessage]:
        """Bridge domain `ChatMessage` (Pydantic) -> provider TypedDict shape.

        Accepts either domain `ChatMessage` instances (real PromptBuilder
        output) or plain mappings (test doubles) — normalizes both to
        `{"role": str, "content": str}`.
        """
        provider_messages: list[ProviderChatMessage] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message["role"]
                content = message["content"]
            else:
                role = message.role
                content = message.content

            role_value = role.value if hasattr(role, "value") else role
            provider_messages.append({"role": role_value, "content": content})

        return provider_messages
