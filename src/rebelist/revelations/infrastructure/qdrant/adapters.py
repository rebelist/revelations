from datetime import datetime
from typing import Final, Iterable, cast

from langchain_core.documents import Document as InputDocument
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import TextSplitter
from qdrant_client.models import SearchParams
from sentence_transformers import CrossEncoder

from rebelist.revelations.domain import ContextDocument, ContextReaderPort, ContextWriterPort, Document


class QdrantContextWriter(ContextWriterPort):
    """Vector writer adapter."""

    def __init__(self, store: QdrantVectorStore, splitter: TextSplitter):
        self.__store = store
        self.__splitter = splitter

    def add(self, document: Document) -> None:
        """Saves a context document."""
        input_document = InputDocument(
            page_content=document.content,
            metadata={
                'id': str(document.id),
                'title': document.title,
                'modified_at': document.modified_at.isoformat(),
                'url': document.url,
            },
        )

        chunks = self.__splitter.split_documents([input_document])
        self.__store.add_documents(chunks)


class QdrantContextReader(ContextReaderPort):
    """Vector reader adapter."""

    SEARCH_EFFORT: Final[int] = 400

    def __init__(self, store: QdrantVectorStore, ranker: CrossEncoder):
        self.__store = store
        self.__ranker = ranker

    def search(self, query: str, limit: int) -> list[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        search_params = SearchParams(hnsw_ef=QdrantContextReader.SEARCH_EFFORT, exact=False)
        items = self.__store.similarity_search(query, k=limit, search_params=search_params)
        documents: list[ContextDocument] = []

        for item in items:
            title = cast(str, item.metadata.get('title', ''))
            url = cast(str, item.metadata.get('url', None))
            modified_at = datetime.fromisoformat(cast(str, item.metadata.get('modified_at')))

            documents.append(ContextDocument(title=title, content=item.page_content, modified_at=modified_at, url=url))

        if len(documents) > 1:
            documents = self.rerank(query, documents)

        return documents

    def rerank(self, query: str, documents: Iterable[ContextDocument]) -> list[ContextDocument]:
        """Re-ranks documents by relevance to the query using a cross-encoder model."""
        pairs = [(query, document.content) for document in documents]
        scores = self.__ranker.predict(pairs)
        ranked_documents = sorted(zip(scores, documents, strict=False), key=lambda x: x[0], reverse=True)

        return [document for _, document in ranked_documents]
