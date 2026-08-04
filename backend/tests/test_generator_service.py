"""
Unit tests for GenerationService.

All three collaborators (retrieval, prompt builder, llm provider) are
mocked — this file tests ORCHESTRATION only, nothing else.

FakeChunk mirrors the real retrieval payload shape:
document_id, chunk_id, chunk_index, page_number, score, content.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.core.exceptions import LLMProviderError
from app.generation.models import GenerationResult
from backend.app.services.generation_service import GenerationService

DOC_A_ID = UUID("11111111-1111-4111-8111-111111111111")
DOC_B_ID = UUID("22222222-2222-4222-8222-222222222222")
CHUNK_1_ID = UUID("33333333-3333-4333-8333-333333333333")
CHUNK_2_ID = UUID("44444444-4444-4444-8444-444444444444")


@dataclass
class FakeChunk:
    """Minimal stand-in that satisfies the RetrievedChunk protocol."""

    document_id: UUID
    chunk_id: UUID
    chunk_index: int
    page_number: int
    score: float
    content: str


def _make_chunks() -> list[FakeChunk]:
    return [
        FakeChunk(
            document_id=DOC_A_ID,
            chunk_id=CHUNK_1_ID,
            chunk_index=0,
            page_number=3,
            score=0.91,
            content="Paris is the capital of France.",
        ),
        FakeChunk(
            document_id=DOC_B_ID,
            chunk_id=CHUNK_2_ID,
            chunk_index=4,
            page_number=12,
            score=0.77,
            content="France is in Western Europe.",
        ),
    ]


@pytest.fixture
def retrieval_service() -> AsyncMock:
    mock = AsyncMock()
    mock.retrieve.return_value = _make_chunks()
    return mock


@pytest.fixture
def prompt_builder() -> MagicMock:
    mock = MagicMock()
    mock.build.return_value = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    return mock


@pytest.fixture
def llm_provider() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = "The capital of France is Paris."
    mock.model_name = "llama-3.3-70b-versatile"
    return mock


@pytest.fixture
def service(
    retrieval_service: AsyncMock, prompt_builder: MagicMock, llm_provider: AsyncMock
) -> GenerationService:
    return GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        top_k=5,
    )


async def test_service_returns_answer(service: GenerationService) -> None:
    result = await service.generate("What is the capital of France?")

    assert isinstance(result, GenerationResult)
    assert result.answer == "The capital of France is Paris."
    assert result.model == "llama-3.3-70b-versatile"


async def test_service_passes_retrieved_chunks_into_prompt_builder(
    service: GenerationService, retrieval_service: AsyncMock, prompt_builder: MagicMock
) -> None:
    question = "What is the capital of France?"
    expected_chunks = _make_chunks()

    await service.generate(question)

    prompt_builder.build.assert_called_once()
    _, kwargs = prompt_builder.build.call_args
    assert kwargs["chunks"] == expected_chunks


async def test_service_calls_provider_exactly_once(
    service: GenerationService, llm_provider: AsyncMock
) -> None:
    await service.generate("What is the capital of France?")

    llm_provider.generate.assert_awaited_once()


async def test_service_builds_citations_correctly(service: GenerationService) -> None:
    result = await service.generate("What is the capital of France?")

    assert [c.chunk_id for c in result.citations] == [CHUNK_1_ID, CHUNK_2_ID]
    assert result.citations[0].document_id == DOC_A_ID
    assert result.citations[0].page_number == 3
    assert result.citations[0].score == 0.91


async def test_service_handles_empty_retrieval_gracefully(
    service: GenerationService, retrieval_service: AsyncMock, prompt_builder: MagicMock
) -> None:
    retrieval_service.retrieve.return_value = []

    result = await service.generate("An unanswerable question")

    prompt_builder.build.assert_called_once_with("An unanswerable question", chunks=[])
    assert result.citations == []
    assert result.answer  # provider still ran, just with no context


async def test_service_handles_llm_failure_cleanly(
    service: GenerationService, llm_provider: AsyncMock
) -> None:
    llm_provider.generate.side_effect = LLMProviderError("Groq request failed")

    with pytest.raises(LLMProviderError):
        await service.generate("What is the capital of France?")
