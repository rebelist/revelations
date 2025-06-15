from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from atlassian import Confluence
from pytest_mock import MockerFixture

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
                'body': {'storage': {'value': '<p>Sample content</p>'}},
            },
            {
                'id': '456',
                'title': 'Another Page',
                'body': {'storage': {'value': '<p>More content</p>'}},
            },
        ]

    @pytest.fixture
    def mock_client(self, mocker: MockerFixture, document_fixtures: list[dict[str, Any]]) -> MagicMock:
        """Mock client."""
        client = mocker.Mock(spec=Confluence)
        client.get_all_pages_from_space.return_value = document_fixtures
        return client

    def test_fetch_yields_transformed_documents(self, mock_client: MagicMock, document_fixtures: list[dict[str, Any]]):
        """Test fetch documents."""
        gateway = ConfluenceGateway(client=mock_client, space='DOCS')
        results = list(gateway.fetch())

        assert len(results) == len(document_fixtures)

        for i, result in enumerate(results):
            expected = document_fixtures[i]
            assert result['id'] == expected['id']
            assert result['title'] == expected['title']
            assert result['content'] == expected['body']['storage']['value']
            assert result['raw'] == expected
            assert isinstance(result['modified_at'], datetime)
            assert datetime.now() - result['modified_at'] < timedelta(seconds=5)

        mock_client.get_all_pages_from_space.assert_called_once_with(
            'DOCS', start=0, limit=20, expand='body.storage', status='current'
        )
