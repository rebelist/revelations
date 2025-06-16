from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.data_fetcher import DataFetchUseCase
from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.infrastructure.confluence import XHTMLParser


class TestDataFetchUseCase:
    @pytest.fixture
    def page_fixture(self) -> dict[str, Any]:
        """Create a page fixture."""
        return {
            'id': 'abc-123',
            'title': 'Mocked Page',
            'content': '<p>Hello, world!</p>',
            'raw': '<p>Hello, world!</p>',
        }

    @pytest.fixture
    def use_case_and_mocks(self, mocker: MockerFixture, page_fixture: dict[str, Any]) -> dict[str, Any]:
        """Sets up the use case along with mocked dependencies."""
        mock_provider: MagicMock = mocker.Mock(spec_set=ContentProviderPort)
        mock_repository: MagicMock = mocker.Mock(spec_set=DocumentRepositoryPort)
        mock_provider.fetch.return_value = [page_fixture]

        use_case = DataFetchUseCase(
            content_provider=mock_provider,
            repository=mock_repository,
        )

        return {
            'use_case': use_case,
            'provider': mock_provider,
            'repository': mock_repository,
            'page': page_fixture,
        }

    def test_document_is_fetched_and_saved(self, use_case_and_mocks: dict[str, Any]) -> None:
        """Ensures the use case fetches pages, converts them to Documents, and saves them."""
        use_case = use_case_and_mocks['use_case']
        mock_repository = use_case_and_mocks['repository']
        page = use_case_and_mocks['page']

        # Act
        use_case()

        # Assert
        mock_repository.save.assert_called_once()
        saved_doc: Document = mock_repository.save.call_args[0][0]

        assert saved_doc.id == page['id']
        assert saved_doc.title == page['title']
        assert saved_doc.content == XHTMLParser(page['content']).text()
        assert saved_doc.raw == page['raw']
        assert isinstance(saved_doc.modified_at, datetime)
