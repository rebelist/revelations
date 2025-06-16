from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

import pytest

from rebelist.revelations.domain import ContextDocument, Document
from rebelist.revelations.infrastructure.qdrant.adapters import (
    QdrantContextReader,
    QdrantContextWriter,
)


@pytest.fixture
def sample_document() -> Document:
    """A fully populated Document fixture."""
    return Document(
        id=123,
        title='Test Document',
        content='This is a test document about AI and ML.',
        modified_at=datetime(2024, 2, 15, 10, 30, 0),
        raw='',
    )


@pytest.fixture
def sample_context_documents() -> List[ContextDocument]:
    """A list of sample ContextDocuments for reranking/search tests."""
    now = datetime(2024, 2, 15, 10, 30, 0)
    return [
        ContextDocument(title='Alpha', content='Alpha content', modified_at=now),
        ContextDocument(title='Beta', content='Beta content', modified_at=now),
    ]


class TestQdrantContextWriter:
    """Tests for QdrantContextWriter behavior."""

    @patch('rebelist.revelations.infrastructure.qdrant.adapters.QdrantVectorStore')
    def test_add_splits_and_adds_documents(
        self,
        mock_vector_store: Mock,
        sample_document: Document,
    ) -> None:
        """Should convert a Document to InputDocument, split it, and add to vector store."""
        mock_client = Mock()
        mock_embedding = Mock()
        mock_splitter = Mock()
        mock_chunks = [Mock()]
        mock_splitter.split_documents.return_value = mock_chunks

        store_mock = Mock()
        mock_vector_store.return_value = store_mock

        writer = QdrantContextWriter(
            client=mock_client, embedding=mock_embedding, splitter=mock_splitter, collection='docs'
        )

        writer.add(sample_document)

        mock_splitter.split_documents.assert_called_once()
        mock_vector_store.assert_called_once()
        store_mock.add_documents.assert_called_once_with(mock_chunks)


class TestQdrantContextReader:
    """Tests for QdrantContextReader behavior."""

    @patch('rebelist.revelations.infrastructure.qdrant.adapters.QdrantVectorStore')
    def test_search_invokes_similarity_and_reranking(
        self,
        mock_vector_store: Mock,
        sample_context_documents: List[ContextDocument],
    ) -> None:
        """Should return documents from similarity search and rerank them."""
        mock_client = Mock()
        mock_embed = Mock()
        mock_ranker = Mock()
        mock_ranker.predict.return_value = [0.6, 0.9]

        qdrant_docs = [
            Mock(page_content=doc.content, metadata={'title': doc.title, 'modified_at': doc.modified_at.isoformat()})
            for doc in sample_context_documents
        ]

        store_mock = Mock()
        store_mock.similarity_search.return_value = qdrant_docs
        mock_vector_store.return_value = store_mock

        reader = QdrantContextReader(
            client=mock_client, embedding=mock_embed, collection='collection', ranker=mock_ranker
        )

        results = list(reader.search('explain transformers', limit=2))

        assert len(results) == 2
        assert isinstance(results[0], ContextDocument)
        assert store_mock.similarity_search.call_count == 1
        assert mock_ranker.predict.call_count == 1

    @patch('rebelist.revelations.infrastructure.qdrant.adapters.QdrantVectorStore')
    def test_rerank_orders_documents_by_score(
        self, _mock_vector_store: Mock, sample_context_documents: List[ContextDocument]
    ) -> None:
        """Should sort ContextDocuments by descending predicted relevance score."""
        mock_client = Mock()
        mock_embed = Mock()
        mock_ranker = Mock()
        mock_ranker.predict.return_value = [0.1, 0.95]  # Beta scores highest

        reader = QdrantContextReader(mock_client, mock_embed, 'c', mock_ranker)
        reranked = reader.rerank('query text', sample_context_documents)

        assert reranked[0].title == 'Beta'
        assert reranked[1].title == 'Alpha'
        assert isinstance(reranked[0], ContextDocument)
