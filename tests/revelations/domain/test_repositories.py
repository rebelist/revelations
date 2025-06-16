from datetime import datetime
from typing import Iterable

import pytest

from rebelist.revelations.domain.models import Document
from rebelist.revelations.domain.repositories import DocumentRepositoryPort


class TestDocumentRepositoryPort:
    @pytest.fixture
    def mock_document(self) -> Document:
        """Creates a sample Document instance for testing with basic test data."""
        return Document(
            id=1, title='Test Document', content='Test Content', modified_at=datetime.now(), raw='Raw Content'
        )

    def test_find_all_abstract_method(self, mock_document: Document) -> None:
        """Tests that the find_all abstract method correctly returns an iterable of documents."""

        class MockDocumentRepository(DocumentRepositoryPort):
            def find_all(self) -> Iterable[Document]:
                return [mock_document]

            def save(self, document: Document) -> None:
                pass

        repository = MockDocumentRepository()
        result = list(repository.find_all())

        assert len(result) == 1
        assert result[0].id == mock_document.id
        assert result[0].title == mock_document.title
        assert result[0].content == mock_document.content

    def test_save_abstract_method(self, mock_document: Document) -> None:
        """Tests that the save abstract method can be called without raising exceptions."""

        class MockDocumentRepository(DocumentRepositoryPort):
            def find_all(self) -> Iterable[Document]:
                return []

            def save(self, document: Document) -> None:
                pass

        repository = MockDocumentRepository()
        repository.save(mock_document)
