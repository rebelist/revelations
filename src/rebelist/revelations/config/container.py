from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

from atlassian import Confluence
from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Singleton
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient

from rebelist.revelations.application.use_cases import DataFetchUseCase, DataVectorizeUseCase
from rebelist.revelations.application.use_cases.semantic_search import SemanticSearchUseCase
from rebelist.revelations.config.settings import load_settings
from rebelist.revelations.infrastructure.confluence.adapters import ConfluenceGateway
from rebelist.revelations.infrastructure.mongo.repositories import MongoDocumentRepository
from rebelist.revelations.infrastructure.ollama.adapters import OllamaAdapter
from rebelist.revelations.infrastructure.qdrant.adapters import QdrantContextReader, QdrantContextWriter


class Container(DeclarativeContainer):
    """Dependency injection container."""

    PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[4])

    @staticmethod
    def create() -> Container:
        """Factory method for creating a container instance."""
        container = Container()
        container.init_resources()

        return container

    @staticmethod
    def _get_mongo_database(client: MongoClient[Any]) -> Database[Mapping[str, Any]]:
        return client.get_default_database()

    ### Configuration ###

    wiring_config = WiringConfiguration(auto_wire=True)
    settings = Singleton(load_settings)

    ### Private Services ###

    __confluence_client = Singleton(
        Confluence, url=settings.provided.confluence.host, token=settings.provided.confluence.token
    )

    __embedding = Singleton(
        OllamaEmbeddings,
        model=settings.provided.rag.embedding_model,
        base_url=settings.provided.ollama.uri,
        # No need for model_kwargs={'device': 'cpu'} as Ollama handles device management
        # encode_kwargs={'normalize_embeddings': True} - Ollama embeddings are typically normalized by default
    )

    __document_splitter = Singleton(
        RecursiveCharacterTextSplitter,
        chunk_size=settings.provided.rag.embedding_chunk_size,
        chunk_overlap=settings.provided.rag.embedding_chunk_overlap,
    )

    ### Public Services ###

    mongo_client = Singleton(MongoClient, host=settings.provided.mongo.uri, tz_aware=True)

    qdrant_client = Singleton(QdrantClient, host=settings.provided.qdrant.host, port=settings.provided.qdrant.port)

    ollama_client = Singleton(Ollama, model=settings.provided.rag.llm_model, base_url=settings.provided.ollama.uri)

    ollama_adapter = Singleton(OllamaAdapter, ollama_client)

    context_writer = Singleton(
        QdrantContextWriter,
        qdrant_client,
        __embedding,
        __document_splitter,
        settings.provided.qdrant.context_collection,
    )

    context_reader = Singleton(
        QdrantContextReader, qdrant_client, __embedding, settings.provided.qdrant.context_collection
    )

    confluence_gateway = Singleton(ConfluenceGateway, __confluence_client, settings.provided.confluence.space)

    database = Singleton(_get_mongo_database, mongo_client)

    document_repository = Singleton(MongoDocumentRepository, database, settings.provided.mongo.source_collection)

    data_fetch_use_case = Singleton(DataFetchUseCase, confluence_gateway, document_repository)

    data_vectorize_use_case = Singleton(DataVectorizeUseCase, document_repository, context_writer)

    semantic_search_use_case = Singleton(SemanticSearchUseCase, context_reader, ollama_adapter)
