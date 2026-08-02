from functools import lru_cache
from typing import Annotated, cast

from fastapi import Depends

from ..core import Settings, get_settings
from ..embedders import BaseEmbeddingProvider, get_embedding_provider
from ..generation import PromptBuilder, PromptBuilderPort, RetrievalServicePort
from ..llms import BaseLLMProvider, get_llm_provider
from ..services import GenerationService, RetrievalServiceAdapter
from ..vectorstores import BaseVectorStore, get_vector_store

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_current_embedding_provider(
    settings: SettingsDep,
) -> BaseEmbeddingProvider:
    return get_embedding_provider(settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)


EmbeddingProviderDep = Annotated[
    BaseEmbeddingProvider, Depends(get_current_embedding_provider)
]


def get_current_vector_store(settings: SettingsDep) -> BaseVectorStore:
    return get_vector_store(
        provider=settings.VECTOR_STORE_PROVIDER,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=settings.QDRANT_PREFER_GRPC,
        api_key=settings.QDRANT_API_KEY,
    )


VectorStoreDep = Annotated[BaseVectorStore, Depends(get_current_vector_store)]


# ---------------------------------------------------------------------------
# /chat dependencies
# ---------------------------------------------------------------------------
@lru_cache
def get_prompt_builder() -> PromptBuilder:
    """Stateless, deterministic — safe to share across requests."""
    return PromptBuilder()


PromptBuilderDep = Annotated[PromptBuilder, Depends(get_prompt_builder)]

# `get_llm_provider` is already `lru_cache`d inside `llms/factory.py`
LLMProviderDep = Annotated[BaseLLMProvider, Depends(get_llm_provider)]


def get_retrieval_service(
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
    vector_store: VectorStoreDep,
) -> RetrievalServiceAdapter:
    """Binds retrieval's collaborators once, exposing only
    `retrieve(query, top_k)` to the generation layer — see
    `RetrievalServiceAdapter` docstring for why this exists.
    """
    return RetrievalServiceAdapter(
        provider=provider,
        vector_store=vector_store,
        collection_name=settings.QDRANT_COLLECTION,
        min_score=settings.MIN_SCORE,
    )


RetrievalServiceDep = Annotated[RetrievalServiceAdapter, Depends(get_retrieval_service)]


def get_generation_service(
    settings: SettingsDep,
    retrieval_service: RetrievalServiceDep,
    prompt_builder: PromptBuilderDep,
    llm_provider: LLMProviderDep,
) -> GenerationService:
    return GenerationService(
        retrieval_service=cast(RetrievalServicePort, retrieval_service),
        prompt_builder=cast(PromptBuilderPort, prompt_builder),
        llm_provider=llm_provider,
        top_k=settings.DEFAULT_TOP_K,
    )


GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]
