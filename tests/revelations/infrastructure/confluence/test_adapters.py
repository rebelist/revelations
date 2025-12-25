from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call

import pytest
from atlassian import Confluence
from pytest_mock import MockerFixture

from rebelist.revelations.config.settings import ConfluenceSettings
from rebelist.revelations.infrastructure.confluence.adapters import ConfluenceGateway


class TestConfluenceGateway:
    """Tests for the ConfluenceGateway class."""

    @pytest.fixture
    def document_fixtures(self) -> list[dict[str, Any]]:
        """Document Fixtures."""
        return [
            {
                'id': '123',
                'title': 'Sample Page',
                'body': {
                    'export_view': {'value': 'Purple elephants often walk slowly while drinking refreshing green tea.'}
                },
                'history': {'lastUpdated': {'when': '2020-11-12T09:04:47.054+01:00'}},
                '_links': {'tinyui': '/1'},
            },
            {
                'id': '456',
                'title': 'Another Page',
                'body': {
                    'export_view': {'value': 'Purple lion often walk slowly while drinking refreshing green tea.'}
                },
                'history': {'lastUpdated': {'when': '2020-11-12T09:04:47.054+01:00'}},
                '_links': {'tinyui': '/2'},
            },
        ]

    @pytest.fixture
    def mock_client(self, mocker: MockerFixture, document_fixtures: list[dict[str, Any]]) -> MagicMock:
        """Mock client."""
        client = mocker.Mock(spec=Confluence)
        client.get_all_pages_from_space_as_generator.return_value = document_fixtures
        client.get_page_as_pdf.return_value = 'random'.encode('utf-8')
        client.url = 'https://example.com'
        return client

    def test_fetch_yields_transformed_documents(self, mock_client: MagicMock, document_fixtures: list[dict[str, Any]]):
        """Test fetch documents."""
        mock_logger = MagicMock()
        settings = ConfluenceSettings(spaces=('DOCS',), throttle_delay_seconds=0)
        gateway = ConfluenceGateway(client=mock_client, settings=settings, logger=mock_logger)
        results = list(gateway.fetch())

        assert len(results) == len(document_fixtures)

        for i, result in enumerate(results):
            expected = document_fixtures[i]
            assert result['id'] == expected['id']
            assert result['title'] == expected['title']
            assert result['content'] == 'random'.encode('utf-8')
            assert result['url'] == mock_client.url + expected['_links']['tinyui']
            assert result['raw'] == expected
            assert isinstance(result['modified_at'], datetime)
            assert result['modified_at'] == datetime.fromisoformat(expected['history']['lastUpdated']['when'])

        mock_client.get_page_as_pdf.assert_has_calls([call('123'), call('456')], any_order=True)
        mock_client.get_all_pages_from_space_as_generator.assert_called_once_with(
            'DOCS', expand='body.export_view,history.lastUpdated', status='current'
        )

    def test_fetch_with_corrupted_document(self, mock_client: MagicMock):
        """Test fetch documents."""
        mock_logger = MagicMock()
        settings = ConfluenceSettings(spaces=('DOCS',), throttle_delay_seconds=0)
        gateway = ConfluenceGateway(client=mock_client, settings=settings, logger=mock_logger)

        documents = [
            {
                'id': '123',
                'body': {
                    'export_view': {'value': 'Purple elephants often walk slowly while drinking refreshing green tea.'}
                },
                '_links': {'tinyui': '/1'},
            },
        ]

        mock_client.get_all_pages_from_space_as_generator.return_value = documents

        assert list(gateway.fetch()) == []

        mock_client.get_page_as_pdf.assert_called_once_with('123')
        mock_client.get_all_pages_from_space_as_generator.assert_called_once_with(
            'DOCS', expand='body.export_view,history.lastUpdated', status='current'
        )

        mock_logger.error.assert_called_once_with("Processing document failed [id=123] - 'title'")
