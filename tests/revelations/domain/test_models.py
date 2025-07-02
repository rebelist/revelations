from datetime import datetime

import pytest

from rebelist.revelations.domain.models import ContextDocument, Document


class TestDocument:
    @pytest.fixture
    def mock_document(self) -> Document:
        """Creates a sample Document instance for testing with basic test data."""
        return Document(
            id=1, title='Test Document', content='Test Content', modified_at=datetime.now(), raw='Raw Content'
        )

    @pytest.fixture
    def mock_context_document(self) -> ContextDocument:
        """Creates a sample ContextDocument instance for testing with basic test data."""
        return ContextDocument(title='Test Context', content='Test Content', modified_at=datetime.now())

    def test_document_creation(self, mock_document: Document) -> None:
        """Test document creation."""
        assert mock_document.id == 1
        assert mock_document.title == 'Test Document'
        assert mock_document.content == 'Test Content'
        assert mock_document.raw == 'Raw Content'

    def test_document_as_dict(self, mock_document: Document) -> None:
        """Test document as dictionary."""
        doc_dict = mock_document.as_dict()

        assert doc_dict['id'] == 1
        assert doc_dict['title'] == 'Test Document'
        assert doc_dict['content'] == 'Test Content'
        assert doc_dict['raw'] == 'Raw Content'


class TestContextDocument:
    @pytest.fixture
    def mock_document(self) -> Document:
        """Creates a sample Document instance for testing with basic test data."""
        return Document(
            id=1, title='Test Document!', content='Test Content', modified_at=datetime.now(), raw='Raw Content'
        )

    @pytest.fixture
    def mock_context_document(self) -> ContextDocument:
        """Creates a sample ContextDocument instance for testing with basic test data."""
        return ContextDocument(title='Test Context!', content='Test Content', modified_at=datetime.now())

    def test_context_document_creation(self, mock_context_document: ContextDocument) -> None:
        """Tests that a ContextDocument can be created with the correct title and content."""
        assert mock_context_document.title == 'Test Context!'
        assert mock_context_document.content == 'Test Content'

    def test_context_document_with_modified_at(self) -> None:
        """Tests that ContextDocument can be created with a modified_at timestamp."""
        modified_at = datetime.now()
        document = ContextDocument(title='Test Context', content='Test Content', modified_at=modified_at)

        assert document.title == 'Test Context'
        assert document.content == 'Test Content'
        assert document.modified_at == modified_at
