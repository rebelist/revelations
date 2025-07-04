from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.data_vectorizer import DataVectorizeUseCase
from rebelist.revelations.domain import ContextWriterPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort
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
        mock_logger = mocker.create_autospec(LoggerPort)
        mock_repository.find_all.return_value = document_fixtures

        use_case = DataVectorizeUseCase(repository=mock_repository, context_writer=mock_writer, logger=mock_logger)

        return {
            'use_case': use_case,
            'repository': mock_repository,
            'writer': mock_writer,
            'logging': mock_logger,
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

    def test_error_in_repository_is_handled(self, mocker: MockerFixture, document_fixtures: list[Document]) -> None:
        """Ensures that exceptions in repository.find_all are caught and re-raised."""
        mock_repository = mocker.create_autospec(DocumentRepositoryPort, instance=True)
        mock_writer = mocker.create_autospec(ContextWriterPort, instance=True)
        mock_logger = mocker.create_autospec(LoggerPort)
        mock_repository.find_all.side_effect = Exception('Repository error')
        use_case = DataVectorizeUseCase(repository=mock_repository, context_writer=mock_writer, logger=mock_logger)
        with pytest.raises(Exception, match='Repository error'):
            use_case()

    def test_error_in_context_writer_is_handled(self, mocker: MockerFixture, document_fixtures: list[Document]) -> None:
        """Ensures that exceptions in context_writer.add are caught and re-raised."""
        mock_repository = mocker.create_autospec(DocumentRepositoryPort, instance=True)
        mock_writer = mocker.create_autospec(ContextWriterPort, instance=True)
        mock_logger = mocker.create_autospec(LoggerPort)
        mock_repository.find_all.return_value = document_fixtures
        mock_writer.add.side_effect = Exception('Writer error')
        use_case = DataVectorizeUseCase(repository=mock_repository, context_writer=mock_writer, logger=mock_logger)
        with pytest.raises(Exception, match='Writer error'):
            use_case()
