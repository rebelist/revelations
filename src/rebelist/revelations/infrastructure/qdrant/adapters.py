from datetime import datetime
from typing import Iterable, cast

from langchain_core.documents import Document as InputDocument
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from rebelist.revelations.domain import ContextDocument, ContextReaderPort, ContextWriterPort, Document
from rebelist.revelations.infrastructure.search_hybrid import HybridSearchStrategy


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
        self.__hybrid_search = HybridSearchStrategy()

    def search(self, query: str, limit: int) -> Iterable[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        # Use similarity search with score threshold for better quality
        items = self.__store.similarity_search_with_score(query, k=limit)
        documents: list[ContextDocument] = []

        for item, score in items:
            # Filter out very low similarity scores (optional quality filter)
            if score < 0.3:  # Adjust threshold based on your data
                continue
                
            title = cast(str, item.metadata.get('title', ''))
            modified_at = datetime.fromisoformat(cast(str, item.metadata.get('modified_at')))

            documents.append(
                ContextDocument(
                    title=title,
                    content=item.page_content,
                    modified_at=modified_at,
                )
            )

        # Apply hybrid search filtering and ranking
        documents = self.__hybrid_search.filter_and_rank(documents, query)
        documents = self.__hybrid_search.deduplicate_documents(documents)
        
        print(f"📊 Found {len(documents)} relevant documents after filtering")
        
        # Always rerank for better quality, but limit to reasonable number
        if len(documents) > 8:
            documents = self.rerank(query, documents)
            return documents[:10]
        
        return documents

    def rerank(self, query: str, documents: Iterable[ContextDocument]) -> list[ContextDocument]:
        """Re-ranks documents by relevance to the query using a cross-encoder model."""
        doc_list = list(documents)
        
        # Skip reranking only for very small sets (≤3 documents)
        if len(doc_list) <= 3:
            return doc_list
            
        pairs = [(query, document.content) for document in doc_list]
        scores = self.__ranker.predict(pairs)
        ranked_documents = sorted(zip(scores, doc_list, strict=False), key=lambda x: x[0], reverse=True)

        return [document for _, document in ranked_documents]
