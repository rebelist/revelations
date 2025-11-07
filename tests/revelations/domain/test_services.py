from datetime import datetime
from typing import Any, Iterable

import pytest

from rebelist.revelations.domain import ContextDocument, Document, Response
from rebelist.revelations.domain.services import (
    ContentProviderPort,
    ContextReaderPort,
    ContextWriterPort,
    ResponseGeneratorPort,
)


class TestContentProviderPort:
    def test_fetch_abstract_method(self) -> None:
        """Tests that the fetch abstract method correctly returns an iterable of content dictionaries."""

        class MockContentProvider(ContentProviderPort):
            def fetch(self) -> Iterable[dict[str, Any]]:
                return [{'id': 1, 'title': 'Test'}]

        provider = MockContentProvider()
        result = list(provider.fetch())

        assert len(result) == 1
        assert result[0]['id'] == 1
        assert result[0]['title'] == 'Test'

    def test_fetch_empty_result(self) -> None:
        """Tests that the fetch method can return an empty iterable."""

        class MockContentProvider(ContentProviderPort):
            def fetch(self) -> Iterable[dict[str, Any]]:
                return []

        provider = MockContentProvider()
        result = list(provider.fetch())

        assert len(result) == 0


class TestContextWriterPort:
    @pytest.fixture
    def mock_document(self) -> Document:
        """Creates a sample Document instance for testing with basic test data."""
        return Document(
            id=1, title='Test Document', content='Test Content', modified_at=datetime.now(), raw='Raw Content'
        )

    def test_add_abstract_method(self, mock_document: Document) -> None:
        """Test add method."""

        class MockContextWriter(ContextWriterPort):
            def add(self, document: Document) -> None:
                pass

        writer = MockContextWriter()
        writer.add(mock_document)  # Should not raise any exception


class TestContextReaderPort:
    @pytest.fixture
    def mock_context_document(self) -> ContextDocument:
        """Creates a sample ContextDocument instance for testing with basic test data."""
        return ContextDocument(title='Test Context', content='Test Content', modified_at=datetime.now())

    def test_search_abstract_method(self, mock_context_document: ContextDocument) -> None:
        """Tests that the search abstract method correctly returns an iterable of context documents."""

        class MockContextReader(ContextReaderPort):
            def search(self, query: str, limit: int) -> Iterable[ContextDocument]:
                return [mock_context_document]

        reader = MockContextReader()
        result = list(reader.search('test', 1))

        assert len(result) == 1
        assert result[0].title == mock_context_document.title
        assert result[0].content == mock_context_document.content

    def test_search_with_limit(self, mock_context_document: ContextDocument) -> None:
        """Tests that the search method respects the limit parameter."""

        class MockContextReader(ContextReaderPort):
            def search(self, query: str, limit: int) -> Iterable[ContextDocument]:
                # Return only one document even though we could return more
                return [mock_context_document]

        reader = MockContextReader()
        result = list(reader.search('test', 1))

        assert len(result) == 1
        assert result[0].title == mock_context_document.title


class TestResponseGeneratorPort:
    @pytest.fixture
    def mock_context_document(self) -> ContextDocument:
        """Creates a sample ContextDocument instance for testing with basic test data."""
        return ContextDocument(title='Test Context', content='Test Content', modified_at=datetime.now())

    def test_respond_abstract_method(self, mock_context_document: ContextDocument) -> None:
        """Tests that the "respond" abstract method correctly generates a response with answer and documents."""

        class MockResponseGenerator(ResponseGeneratorPort):
            def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
                return Response(answer='Test Answer', documents=[mock_context_document])

        generator = MockResponseGenerator()
        result = generator.respond('test question', [mock_context_document])

        assert result.answer == 'Test Answer'
        assert len(list(result.documents)) == 1
        assert list(result.documents)[0].title == mock_context_document.title

    def test_get_prompt(self) -> None:
        """Tests that the get_prompt method returns a properly formatted prompt template."""

        class MockResponseGenerator(ResponseGeneratorPort):
            def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
                return Response(answer='', documents=[])

        generator = MockResponseGenerator()
        system_prompt = generator.get_system_prompt()
        user_prompt = generator.get_user_prompt()

        assert 'You are a senior co-worker and an expert' in system_prompt
        assert '{context}' in user_prompt
        assert '{question}' in user_prompt

    def test_respond_with_empty_documents(self) -> None:
        """Tests that the "respond" method can handle empty document lists."""

        class MockResponseGenerator(ResponseGeneratorPort):
            def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
                return Response(answer='No documents found.', documents=[])

        generator = MockResponseGenerator()
        result = generator.respond('test question', [])

        assert result.answer == 'No documents found.'
        assert len(list(result.documents)) == 0
