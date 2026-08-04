from __future__ import annotations

import logging
from collections.abc import Sequence

from ..core import LLMProviderError
from ..generation import (
    Citation,
    GenerationResult,
    PromptBuilderPort,
    RetrievalServicePort,
)
from ..llms import BaseLLMProvider, to_provider_messages
from ..retrieval import SearchResult

logger = logging.getLogger(__name__)


class GenerationService:
    """Orchestrates retrieval -> prompt building -> LLM generation."""

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
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Run the complete RAG generation pipeline."""
        resolved_top_k = top_k if top_k is not None else self._top_k

        chunks = await self._retrieval_service.retrieve(
            question,
            top_k=resolved_top_k,
        )

        if not chunks:
            logger.info("No context retrieved for question: %r", question)

        messages = self._prompt_builder.build(
            question,
            chunks=chunks,
        )
        provider_messages = to_provider_messages(messages)

        try:
            answer = await self._llm_provider.generate(provider_messages)
        except LLMProviderError:
            logger.exception(
                "LLM generation failed for question: %r",
                question,
            )
            raise

        return GenerationResult(
            answer=answer,
            citations=self._build_citations(chunks),
            model=self._llm_provider.model_name,
        )

    @staticmethod
    def _build_citations(
        chunks: Sequence[SearchResult],
    ) -> list[Citation]:
        """Build citations preserving retrieval order."""
        return [
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                score=chunk.score,
            )
            for chunk in chunks
        ]
