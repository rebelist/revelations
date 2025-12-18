from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final, Mapping, cast

import loguru
from atlassian import Confluence
from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Singleton
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, TextSplitter
from pymongo import MongoClient
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from rebelist.revelations.application.use_cases import DataEmbeddingUseCase, DataExtractionUseCase, InferenceUseCase
from rebelist.revelations.application.use_cases.benchmark import BenchmarkUseCase
from rebelist.revelations.config.prompts import benchmark_prompt_config, chat_prompt_config
from rebelist.revelations.config.settings import RagSettings, load_settings
from rebelist.revelations.domain import RetrievalEvaluator
from rebelist.revelations.infrastructure.confluence import ConfluenceGateway
from rebelist.revelations.infrastructure.logging import Logger
from rebelist.revelations.infrastructure.mongo import MongoDocumentRepository
from rebelist.revelations.infrastructure.mupdf.adapters import PdfConverter
from rebelist.revelations.infrastructure.ollama import OllamaMemoryChatAdapter
from rebelist.revelations.infrastructure.ollama.adapters import OllamaAnswerEvaluator, OllamaStatelessChatAdapter
from rebelist.revelations.infrastructure.qdrant import QdrantContextReader, QdrantContextWriter


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

    @staticmethod
    def _get_text_splitter(settings: RagSettings) -> TextSplitter:
        tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(settings.tokenizer_model_path))
        tokenizer.model_max_length = sys.maxsize
        return MarkdownTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    ### Configuration ###

    wiring_config = WiringConfiguration(auto_wire=True)
    settings = Singleton(load_settings)

    ### Private Services ###

    __confluence_client = Singleton(
        Confluence,
        url=settings.provided.confluence.host,
        token=settings.provided.confluence.token,
        backoff_and_retry=True,
        retry_status_codes=[413, 429, 500, 503],
        max_backoff_seconds=30,
        max_backoff_retries=3,
    )

    __embedding = Singleton(
        OllamaEmbeddings,
        model=settings.provided.rag.embedding_model,
        base_url=settings.provided.ollama.uri,
    )

    __document_splitter = Singleton(_get_text_splitter, settings.provided.rag)

    __ranker = Singleton(
        CrossEncoder,
        model_name_or_path=settings.provided.rag.ranker_model_path,
        local_files_only=True,
    )

    __pdf_converter = Singleton(PdfConverter)

    ### Public Services ###
    logger = Singleton(Logger, loguru.logger)

    mongo_client = Singleton(MongoClient, host=settings.provided.mongo.uri, tz_aware=True)

    qdrant_client = Singleton(QdrantClient, host=settings.provided.qdrant.host, port=settings.provided.qdrant.port)

    ollama_chat = Singleton(
        ChatOllama,
        model=settings.provided.rag.llm_model,
        base_url=settings.provided.ollama.uri,
        request_timeout=60.0,
        temperature=0.2,  # Lower temperature for more consistent responses
        num_ctx=4096,  # Limit context window to improve speed (adjust based on model)
        num_predict=512,  # Limit max tokens to generate for faster responses
        top_p=0.9,  # Nucleus sampling for faster decoding
        repeat_penalty=1.1,  # Reduce repetition
    )
    ollama_memory_chat_adapter = Singleton(OllamaMemoryChatAdapter, ollama_chat, chat_prompt_config)

    ollama_stateless_chat_adapter = Singleton(OllamaStatelessChatAdapter, ollama_chat, chat_prompt_config)

    retrieval_evaluator = Singleton(RetrievalEvaluator)

    ollama_answer_evaluator = Singleton(OllamaAnswerEvaluator, ollama_chat, benchmark_prompt_config)

    context_writer = Singleton(
        QdrantContextWriter,
        qdrant_client,
        __embedding,
        __document_splitter,
        settings.provided.qdrant.context_collection,
    )

    context_reader = Singleton(
        QdrantContextReader, qdrant_client, __embedding, settings.provided.qdrant.context_collection, __ranker
    )

    confluence_gateway = Singleton(ConfluenceGateway, __confluence_client, settings.provided.confluence, logger)

    database = Singleton(_get_mongo_database, mongo_client)

    document_repository = Singleton(MongoDocumentRepository, database, settings.provided.mongo.source_collection)

    data_extraction_use_case = Singleton(
        DataExtractionUseCase, confluence_gateway, document_repository, __pdf_converter, logger
    )

    data_embedding_use_case = Singleton(DataEmbeddingUseCase, document_repository, context_writer, logger)

    inference_use_case = Singleton(
        InferenceUseCase, context_reader, ollama_memory_chat_adapter, settings.provided.rag, logger
    )

    benchmark_use_case = Singleton(
        BenchmarkUseCase,
        retrieval_evaluator,
        ollama_answer_evaluator,
        context_reader,
        ollama_stateless_chat_adapter,
        logger,
    )
