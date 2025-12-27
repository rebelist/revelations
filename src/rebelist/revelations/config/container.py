from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final, Mapping, cast

import loguru
from atlassian import Confluence
from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Callable, Singleton
from docling.document_converter import DocumentConverter as DoclingConverter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import MarkdownTextSplitter, TextSplitter
from pymongo import MongoClient
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from rebelist.revelations.application.use_cases import DataEmbeddingUseCase, DataExtractionUseCase, InferenceUseCase
from rebelist.revelations.application.use_cases.benchmark import BenchmarkUseCase
from rebelist.revelations.config.settings import RagSettings, load_settings
from rebelist.revelations.domain import AnswerEvaluatorPort, ChatAdapterPort, RetrievalEvaluator
from rebelist.revelations.infrastructure.confluence import ConfluenceGateway
from rebelist.revelations.infrastructure.docling.adapters import PdfConverter
from rebelist.revelations.infrastructure.filesystem import YamlPromptLoader
from rebelist.revelations.infrastructure.logging import Logger
from rebelist.revelations.infrastructure.mongo import MongoDocumentRepository
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

    __sparse_embedding = Singleton(
        FastEmbedSparse,
        model_name=settings.provided.qdrant.sparse_embedding,
    )

    __document_splitter = Singleton(_get_text_splitter, settings.provided.rag)

    __ranker = Singleton(
        CrossEncoder,
        model_name_or_path=settings.provided.rag.ranker_model_path,
        local_files_only=True,
    )

    __docling_converter = Singleton(DoclingConverter)

    __pdf_converter = Singleton(PdfConverter, __docling_converter)

    __prompt_loader = Singleton(
        YamlPromptLoader,
        f'{PROJECT_ROOT}/src/rebelist/revelations/config/prompts.yaml',
        namespaces={'ChatAdapterPort': ChatAdapterPort, 'AnswerEvaluatorPort': AnswerEvaluatorPort},
    )

    __chat_prompt = Callable(__prompt_loader().load, key='chat_prompt')

    __benchmark_prompt = Callable(__prompt_loader().load, key='benchmark_prompt')

    ### Public Services ###
    logger = Singleton(Logger, loguru.logger)

    mongo_client = Singleton(MongoClient, host=settings.provided.mongo.uri, tz_aware=True)

    qdrant_client = Singleton(QdrantClient, host=settings.provided.qdrant.host, port=settings.provided.qdrant.port)

    qdrant_vector_store = Singleton(
        QdrantVectorStore,
        client=qdrant_client,
        collection_name=settings.provided.qdrant.context_collection,
        embedding=__embedding,
        sparse_embedding=__sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=settings.provided.qdrant.vector_name,
        sparse_vector_name=settings.provided.qdrant.sparse_vector_name,
    )

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
    ollama_memory_chat_adapter = Singleton(OllamaMemoryChatAdapter, ollama_chat, __chat_prompt)

    ollama_stateless_chat_adapter = Singleton(OllamaStatelessChatAdapter, ollama_chat, __chat_prompt)

    retrieval_evaluator = Singleton(RetrievalEvaluator)

    ollama_answer_evaluator = Singleton(OllamaAnswerEvaluator, ollama_chat, __benchmark_prompt)

    context_writer = Singleton(QdrantContextWriter, qdrant_vector_store, __document_splitter)

    context_reader = Singleton(QdrantContextReader, qdrant_vector_store, __ranker)

    confluence_gateway = Singleton(ConfluenceGateway, __confluence_client, settings.provided.confluence, logger)

    database = Singleton(_get_mongo_database, mongo_client)

    document_repository = Singleton(MongoDocumentRepository, database, settings.provided.mongo.source_collection)

    data_extraction_use_case = Singleton(
        DataExtractionUseCase, confluence_gateway, document_repository, __pdf_converter, settings.provided.rag, logger
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
