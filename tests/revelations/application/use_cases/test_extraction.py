from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.extraction import DataExtractionUseCase
from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort, PdfConverterPort


class TestDataExtractionUseCase:
    @pytest.fixture
    def document_fixture(self) -> dict[str, Any]:
        """Create a document fixture."""
        return {
            'id': 'abc-123',
            'title': 'Mocked Document',
            'content': '# This is a title',
            'modified_at': datetime.fromisoformat('2020-11-12T09:04:47.054+01:00'),
            'raw': '<p>Hello, world!</p>',
            'url': 'https://example.com',
        }

    @pytest.fixture
    def use_case_and_mocks(self, mocker: MockerFixture, document_fixture: dict[str, Any]) -> dict[str, Any]:
        """Sets up the use case along with mocked dependencies."""
        mock_provider: MagicMock = mocker.Mock(spec_set=ContentProviderPort)
        mock_repository: MagicMock = mocker.Mock(spec_set=DocumentRepositoryPort)
        pdf_converter: PdfConverterPort | MagicMock = mocker.Mock(spec_set=PdfConverterPort)
        mock_logger = mocker.create_autospec(LoggerPort)
        mock_provider.fetch.return_value = [document_fixture]
        pdf_converter.pdf_to_markdown.return_value = '# This is a title'

        use_case = DataExtractionUseCase(
            content_provider=mock_provider,
            repository=mock_repository,
            converter=pdf_converter,
            logger=mock_logger,
        )

        return {
            'use_case': use_case,
            'provider': mock_provider,
            'repository': mock_repository,
            'logger': mock_logger,
            'document': document_fixture,
        }

    def test_document_is_fetched_and_saved(self, use_case_and_mocks: dict[str, Any]) -> None:
        """Ensures the use case fetches documents, converts them to Documents, and saves them."""
        use_case = use_case_and_mocks['use_case']
        mock_repository = use_case_and_mocks['repository']
        document = use_case_and_mocks['document']

        # Act
        use_case()

        # Assert
        mock_repository.save.assert_called_once()
        saved_document: Document = mock_repository.save.call_args[0][0]

        assert saved_document.id == document['id']
        assert saved_document.title == document['title']
        assert saved_document.content == document['content']
        assert saved_document.raw == document['raw']
        assert saved_document.url == document['url']
        assert isinstance(saved_document.modified_at, datetime)

    def test_error_in_content_provider_is_handled(self, mocker: MockerFixture) -> None:
        """Ensures that exceptions in content_provider.fetch are caught and re-raised."""
        mock_provider = mocker.create_autospec(ContentProviderPort)
        mock_repository = mocker.create_autospec(DocumentRepositoryPort)
        pdf_converter = mocker.create_autospec(PdfConverterPort)
        mock_logger = mocker.create_autospec(LoggerPort)
        mock_provider.fetch.side_effect = Exception('Provider error')
        use_case = DataExtractionUseCase(
            content_provider=mock_provider, repository=mock_repository, converter=pdf_converter, logger=mock_logger
        )
        with pytest.raises(Exception, match='Provider error'):
            use_case()

    def test_error_in_repository_is_handled(self, use_case_and_mocks: dict[str, Any]) -> None:
        """Ensures that exceptions in repository.save are caught and re-raised."""
        use_case = use_case_and_mocks['use_case']
        mock_repository = use_case_and_mocks['repository']
        mock_logger = use_case_and_mocks['logger']
        mock_repository.save.side_effect = Exception('Repository error')

        use_case()

        mock_logger.error.assert_called_once_with("Failed fetching document. <class 'Exception'> - Repository error")
