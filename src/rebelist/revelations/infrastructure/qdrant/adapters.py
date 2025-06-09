from datetime import datetime
from typing import Iterable, cast

from langchain_core.documents import Document as InputDocument
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from rebelist.revelations.domain import ContextDocument, ContextReaderPort, ContextWriterPort, Document


class QdrantContextWriter(ContextWriterPort):
    """Vector writer adapter."""

    def __init__(
        self,
        client: QdrantClient,
        embedding: OllamaEmbeddings,
        splitter: RecursiveCharacterTextSplitter,
        collection: str,
    ):
        self.__store = QdrantVectorStore(client=client, collection_name=collection, embedding=embedding)
        self.__splitter = splitter

    def add(self, document: Document) -> None:
        """Saves a context document."""
        input_document = InputDocument(
            page_content=document.content,
            metadata={
                'id': str(document.id),
                'title': document.title,
                'modified_at': document.modified_at.isoformat(),
            },
        )

        chunks = self.__splitter.split_documents([input_document])
        self.__store.add_documents(chunks)


class QdrantContextReader(ContextReaderPort):
    """Vector reader adapter."""

    def __init__(self, client: QdrantClient, embedding: OllamaEmbeddings, collection: str, ranker: CrossEncoder):
        self.__store = QdrantVectorStore(client=client, collection_name=collection, embedding=embedding)
        self.__ranker = ranker

    def search(self, query: str, limit: int) -> Iterable[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        items = self.__store.similarity_search(query, k=limit)
        documents: list[ContextDocument] = []

        for item in items:
            title = cast(str, item.metadata.get('title', ''))
            modified_at = datetime.fromisoformat(cast(str, item.metadata.get('modified_at')))

            documents.append(
                ContextDocument(
                    title=title,
                    content=item.page_content,
                    modified_at=modified_at,
                )
            )

        documents = self.rerank(query, documents)

        return documents[:5]

    def rerank(self, query: str, documents: Iterable[ContextDocument]) -> list[ContextDocument]:
        """Re-ranks documents by relevance to the query using a cross-encoder model."""
        pairs = [(query, document.content) for document in documents]
        scores = self.__ranker.predict(pairs)
        ranked_documents = sorted(zip(scores, documents, strict=False), key=lambda x: x[0], reverse=True)

        return [document for _, document in ranked_documents]
