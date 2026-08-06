from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..core import Settings, get_settings
from ..embedders import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    BaseSparseEmbeddingProvider,
    get_embedding_provider,
    get_reranker,
    get_sparse_provider,
)
from ..generation import PromptBuilder
from ..llms import BaseLLMProvider, get_llm_provider
from ..services import (
    DocumentChunkerService,
    DocumentEmbedderService,
    DocumentIndexerService,
    DocumentParserService,
    FileStorageService,
    GenerationService,
    RetrievalServiceAdapter,
)
from ..vectorstores import BaseVectorStore, get_vector_store

SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingProviderDep = Annotated[BaseEmbeddingProvider, Depends(get_embedding_provider)]
RerankerDep = Annotated[BaseRerankerProvider | None, Depends(get_reranker)]
SparseProviderDep = Annotated[
    BaseSparseEmbeddingProvider | None, Depends(get_sparse_provider)
]
VectorStoreDep = Annotated[BaseVectorStore, Depends(get_vector_store)]


# ---------------------------------------------------------------------------
# /chat dependencies
# ---------------------------------------------------------------------------
@lru_cache
def get_prompt_builder() -> PromptBuilder:
    """Stateless, deterministic — safe to share across requests."""
    return PromptBuilder()


PromptBuilderDep = Annotated[PromptBuilder, Depends(get_prompt_builder)]
LLMProviderDep = Annotated[BaseLLMProvider, Depends(get_llm_provider)]


def get_retrieval_service(
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
    vector_store: VectorStoreDep,
    reranker: RerankerDep,
    sparse_provider: SparseProviderDep,
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
        fetch_k=settings.RETRIEVAL_FETCH_K,
        rerank_min_score=settings.RERANK_MIN_SCORE,
        reranker=reranker,
        sparse_provider=sparse_provider,
    )


RetrievalServiceDep = Annotated[RetrievalServiceAdapter, Depends(get_retrieval_service)]


def get_generation_service(
    settings: SettingsDep,
    retrieval_service: RetrievalServiceDep,
    prompt_builder: PromptBuilderDep,
    llm_provider: LLMProviderDep,
) -> GenerationService:
    return GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        top_k=settings.DEFAULT_TOP_K,
    )


GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]

# ---------------------------------------------------------------------------
# /ingestion dependencies
# ---------------------------------------------------------------------------


def get_file_storage_service(settings: SettingsDep) -> FileStorageService:
    return FileStorageService(max_size_bytes=settings.MAX_FILE_SIZE_BYTES)


FileStorageServiceDep = Annotated[FileStorageService, Depends(get_file_storage_service)]


def get_document_parser_service(settings: SettingsDep) -> DocumentParserService:
    return DocumentParserService(upload_dir=settings.UPLOAD_DIR)


DocumentParserServiceDep = Annotated[
    DocumentParserService, Depends(get_document_parser_service)
]


def get_document_chunker_service(
    settings: SettingsDep,
    parser: DocumentParserServiceDep,
) -> DocumentChunkerService:
    return DocumentChunkerService(
        parser=parser,
        min_chunk_chars=settings.MIN_CHUNK_CHARS,
        dedup_similarity_threshold=settings.DEDUP_SIMILARITY_THRESHOLD,
    )


DocumentChunkerServiceDep = Annotated[
    DocumentChunkerService, Depends(get_document_chunker_service)
]


def get_document_embedder_service(
    provider: EmbeddingProviderDep,
) -> DocumentEmbedderService:
    return DocumentEmbedderService(provider=provider)


DocumentEmbedderServiceDep = Annotated[
    DocumentEmbedderService, Depends(get_document_embedder_service)
]


def get_document_indexer_service(
    settings: SettingsDep,
    chunker: DocumentChunkerServiceDep,
    embedder: DocumentEmbedderServiceDep,
    provider: EmbeddingProviderDep,
    vector_store: VectorStoreDep,
    sparse_provider: SparseProviderDep,
) -> DocumentIndexerService:
    return DocumentIndexerService(
        chunker=chunker,
        embedder=embedder,
        provider=provider,
        vector_store=vector_store,
        sparse_provider=sparse_provider,
        ingestion_batch_size=settings.INGESTION_BATCH_SIZE,
    )


DocumentIndexerServiceDep = Annotated[
    DocumentIndexerService, Depends(get_document_indexer_service)
]
