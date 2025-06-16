from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.data_vectorizer import DataVectorizeUseCase
from rebelist.revelations.domain import ContextWriterPort, Document
from rebelist.revelations.infrastructure.mongo import MongoDocumentRepository


class TestDataVectorizeUseCase:
    @pytest.fixture
    def document_fixtures(self) -> list[Document]:
        """Create document fixtures."""
        modified_at = datetime(2024, 2, 15, 10, 30, 0)
        return [
            Document(
                id=100,
                title='First Doc',
                content='Some processed content',
                modified_at=modified_at,
                raw='<p>raw</p>',
            ),
            Document(
                id=200,
                title='Second Doc',
                content='Another one',
                modified_at=modified_at,
                raw='<div>raw</div>',
            ),
        ]

    @pytest.fixture
    def use_case_and_mocks(self, mocker: MockerFixture, document_fixtures: list[Document]) -> dict[str, Any]:
        """Prepares the use case with mocked repository and context writer."""
        mock_repository: MagicMock = mocker.Mock(spec_set=MongoDocumentRepository)
        mock_writer: MagicMock = mocker.Mock(spec_set=ContextWriterPort)
        mock_repository.find_all.return_value = document_fixtures

        use_case = DataVectorizeUseCase(repository=mock_repository, context_writer=mock_writer)

        return {
            'use_case': use_case,
            'repository': mock_repository,
            'writer': mock_writer,
            'documents': document_fixtures,
        }

    def test_all_documents_are_vectorized(self, use_case_and_mocks: dict[str, Any]) -> None:
        """Ensures all retrieved documents are passed to the context writer for vectorization."""
        use_case = use_case_and_mocks['use_case']
        mock_writer = use_case_and_mocks['writer']
        documents = use_case_and_mocks['documents']

        # Act
        use_case()

        # Assert
        assert mock_writer.add.call_count == len(documents)
        for doc in documents:
            mock_writer.add.assert_any_call(doc)
