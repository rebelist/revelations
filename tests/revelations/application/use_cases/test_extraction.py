from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.extraction import DataExtractionUseCase
from rebelist.revelations.config.settings import RagSettings
from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort, PdfConverterPort


class TestDataExtractionUseCase:
    @pytest.fixture
    def document_fixture(self) -> dict[str, Any]:
        """Provides a raw document payload returned by the content provider."""
        return {
            'id': 'abc-123',
            'title': 'Mocked Document',
            'content': '# This is a title',
            'modified_at': datetime.fromisoformat('2020-11-12T09:04:47.054+01:00'),
            'raw': '<p>Hello, world!</p>',
            'url': 'https://example.com',
        }

    @pytest.fixture
    def content_provider(self, mocker: MockerFixture, document_fixture: dict[str, Any]) -> MagicMock:
        """Mocks the content provider returning one document."""
        provider = mocker.create_autospec(ContentProviderPort, instance=True)
        provider.fetch.return_value = [document_fixture]
        return provider

    @pytest.fixture
    def repository(self, mocker: MockerFixture) -> MagicMock:
        """Mocks the document repository."""
        return mocker.create_autospec(DocumentRepositoryPort, instance=True)

    @pytest.fixture
    def pdf_converter(self, mocker: MockerFixture) -> MagicMock:
        """Mocks PDF conversion into Markdown."""
        converter = mocker.create_autospec(PdfConverterPort, instance=True)
        converter.pdf_to_markdown.return_value = '# This is a title'
        return converter

    @pytest.fixture
    def logger(self, mocker: MockerFixture) -> MagicMock:
        """Mocks the logger used by the use case."""
        return mocker.create_autospec(LoggerPort, instance=True)

    @pytest.fixture
    def settings(self) -> RagSettings:
        """Provides RAG settings with a minimum content length."""
        return RagSettings(min_content_length=5)

    @pytest.fixture
    def use_case(
        self,
        content_provider: MagicMock,
        repository: MagicMock,
        pdf_converter: MagicMock,
        settings: RagSettings,
        logger: MagicMock,
    ) -> DataExtractionUseCase:
        """Creates the DataExtractionUseCase with all dependencies wired."""
        return DataExtractionUseCase(
            content_provider=content_provider,
            repository=repository,
            converter=pdf_converter,
            settings=settings,
            logger=logger,
        )

    def test_document_is_fetched_converted_and_saved(
        self,
        use_case: DataExtractionUseCase,
        repository: MagicMock,
        document_fixture: dict[str, Any],
    ) -> None:
        """Ensures fetched documents are converted into domain Documents and persisted."""
        use_case()

        repository.save.assert_called_once()
        saved_document: Document = repository.save.call_args[0][0]

        assert saved_document.id == document_fixture['id']
        assert saved_document.title == document_fixture['title']
        assert saved_document.content == document_fixture['content']
        assert saved_document.raw == document_fixture['raw']
        assert saved_document.url == document_fixture['url']
        assert isinstance(saved_document.modified_at, datetime)

    def test_exception_in_content_provider_is_propagated(
        self,
        mocker: MockerFixture,
        repository: MagicMock,
        pdf_converter: MagicMock,
        settings: RagSettings,
        logger: MagicMock,
    ) -> None:
        """Ensures failures while fetching content are not swallowed."""
        provider = mocker.create_autospec(ContentProviderPort, instance=True)
        provider.fetch.side_effect = Exception('Provider error')

        use_case = DataExtractionUseCase(
            content_provider=provider,
            repository=repository,
            converter=pdf_converter,
            settings=settings,
            logger=logger,
        )

        with pytest.raises(Exception, match='Provider error'):
            use_case()

    def test_exception_while_saving_document_is_logged(
        self,
        use_case: DataExtractionUseCase,
        repository: MagicMock,
        logger: MagicMock,
    ) -> None:
        """Ensures repository failures are logged with document context."""
        repository.save.side_effect = Exception('Repository error')

        use_case()

        logger.error.assert_called_once_with('Error saving document. [id=abc-123] - Repository error')
