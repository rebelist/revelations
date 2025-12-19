from datetime import datetime
from typing import List
from unittest.mock import Mock

import pytest
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import TextSplitter
from pytest_mock import MockerFixture
from sentence_transformers import CrossEncoder

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
        url=None,
    )


@pytest.fixture
def sample_context_documents() -> List[ContextDocument]:
    """A list of sample ContextDocuments for reranking/search tests."""
    now = datetime(2024, 2, 15, 10, 30, 0)
    return [
        ContextDocument(title='1', content='1 content', modified_at=now),
        ContextDocument(title='2', content='2 content', modified_at=now),
        ContextDocument(title='3', content='3 content', modified_at=now),
        ContextDocument(title='4', content='4 content', modified_at=now),
        ContextDocument(title='5', content='5 content', modified_at=now),
        ContextDocument(title='6', content='6 content', modified_at=now),
    ]


class TestQdrantContextWriter:
    """Tests for QdrantContextWriter behavior."""

    def test_add_splits_and_adds_documents(
        self,
        mocker: MockerFixture,
        sample_document: Document,
    ) -> None:
        """Should convert a Document to InputDocument, split it, and add to vector store."""
        mock_qrant_vector_store = mocker.create_autospec(QdrantVectorStore, spec_set=True, instance=True)
        mock_splitter = mocker.create_autospec(TextSplitter, spec_set=True, instance=True)

        mock_chunks = [Mock()]
        mock_splitter.split_documents.return_value = mock_chunks

        writer = QdrantContextWriter(mock_qrant_vector_store, mock_splitter)
        writer.add(sample_document)

        mock_splitter.split_documents.assert_called_once()
        mock_qrant_vector_store.add_documents.assert_called_once_with(mock_chunks)


class TestQdrantContextReader:
    """Tests for QdrantContextReader behavior."""

    def test_search_invokes_similarity_and_reranking(
        self,
        mocker: MockerFixture,
        sample_context_documents: List[ContextDocument],
    ) -> None:
        """Should return documents from similarity search and rerank them."""
        qdrant_docs = [
            Mock(content=doc.content, metadata={'title': doc.title, 'modified_at': doc.modified_at.isoformat()})
            for doc in sample_context_documents
        ]

        mock_qrant_vector_store = mocker.create_autospec(QdrantVectorStore, spec_set=True, instance=True)
        mock_qrant_vector_store.similarity_search.return_value = qdrant_docs
        mock_ranker = mocker.create_autospec(CrossEncoder, spec_set=True, instance=True)
        mock_ranker.predict.return_value = [0.6, 0.9]

        reader = QdrantContextReader(mock_qrant_vector_store, mock_ranker)

        results = list(reader.search('explain transformers', limit=2))

        assert len(results) == 2
        assert isinstance(results[0], ContextDocument)
        assert mock_qrant_vector_store.similarity_search.call_count == 1
        assert mock_ranker.predict.call_count == 1

    def test_rerank_orders_documents_by_score(
        self,
        mocker: MockerFixture,
        sample_context_documents: List[ContextDocument],
    ) -> None:
        """Should sort ContextDocuments by descending predicted relevance score."""
        mock_qrant_vector_store = mocker.create_autospec(QdrantVectorStore, spec_set=True, instance=True)
        mock_ranker = mocker.create_autospec(CrossEncoder, spec_set=True, instance=True)
        mock_ranker.predict.return_value = [0.1, 0.95]

        reader = QdrantContextReader(mock_qrant_vector_store, mock_ranker)
        reranked = reader.rerank('query text', sample_context_documents)

        assert reranked[0].title == '2'
        assert reranked[1].title == '1'
        assert isinstance(reranked[0], ContextDocument)
