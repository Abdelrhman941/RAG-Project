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
"""

import logging

from ..core import LLMProviderError
from ..generation import (
    Citation,
    GenerationResult,
    PromptBuilderPort,
    RetrievalServicePort,
    RetrievedChunk,
)
from ..llms import BaseLLMProvider

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

    async def generate(self, question: str) -> GenerationResult:
        chunks = await self._retrieval_service.retrieve(question, top_k=self._top_k)

        if not chunks:
            logger.info("No chunks retrieved for question: %r", question)

        messages = self._prompt_builder.build(question, chunks=chunks)

        try:
            answer = await self._llm_provider.generate(messages)
        except LLMProviderError:
            logger.exception("LLM generation failed for question: %r", question)
            raise

        return GenerationResult(
            answer=answer,
            citations=self._build_citations(chunks),
            model=self._llm_provider.model_name,
        )

    @staticmethod
    def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
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
