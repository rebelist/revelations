from datetime import datetime
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.application.use_cases.embedding import DataEmbeddingUseCase
from rebelist.revelations.domain import ContextWriterPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort
from rebelist.revelations.infrastructure.mongo import MongoDocumentRepository


class TestDataEmbeddingUseCase:
    @pytest.fixture
    def documents(self) -> list[Document]:
        """Create document fixtures."""
        modified_at = datetime(2024, 2, 15, 10, 30, 0)
        return [
            Document(
                id=100,
                title='First Doc',
                content='Some processed content',
                modified_at=modified_at,
                raw='<p>raw</p>',
                url='https://example.com',
            ),
            Document(
                id=200,
                title='Second Doc',
                content='Another one',
                modified_at=modified_at,
                raw='<div>raw</div>',
                url='https://example.com',
            ),
        ]

    @pytest.fixture
    def repository(
        self,
        mocker: MockerFixture,
        documents: list[Document],
    ) -> MagicMock:
        """Create repository fixture."""
        repo: MagicMock = mocker.Mock(spec_set=MongoDocumentRepository)
        repo.find_all.return_value = documents
        return repo

    @pytest.fixture
    def context_writer(self, mocker: MockerFixture) -> MagicMock:
        """Create context writer fixture."""
        return mocker.create_autospec(ContextWriterPort, instance=True)

    @pytest.fixture
    def logger(self, mocker: MockerFixture) -> MagicMock:
        """Logger fixture."""
        return mocker.create_autospec(LoggerPort)

    @pytest.fixture
    def use_case(
        self,
        repository: MagicMock,
        context_writer: MagicMock,
        logger: MagicMock,
    ) -> DataEmbeddingUseCase:
        """Create usecase fixture."""
        return DataEmbeddingUseCase(
            repository=repository,
            context_writer=context_writer,
            logger=logger,
        )

    def test_all_documents_are_vectorized(
        self,
        use_case: DataEmbeddingUseCase,
        context_writer: MagicMock,
        documents: list[Document],
    ) -> None:
        """Ensures all retrieved documents are passed to the context writer for vectorization."""
        use_case()

        assert context_writer.add.call_count == len(documents)
        for document in documents:
            context_writer.add.assert_any_call(document)

    def test_error_in_repository_is_raised(
        self,
        mocker: MockerFixture,
        context_writer: MagicMock,
        logger: MagicMock,
    ) -> None:
        """Ensures exceptions in repository.find_all are propagated."""
        repository = mocker.create_autospec(DocumentRepositoryPort, instance=True)
        repository.find_all.side_effect = Exception('Repository error')

        use_case = DataEmbeddingUseCase(
            repository=repository,
            context_writer=context_writer,
            logger=logger,
        )

        with pytest.raises(Exception, match='Repository error'):
            use_case()

    def test_error_in_context_writer_is_logged(
        self,
        mocker: MockerFixture,
        documents: list[Document],
    ) -> None:
        """Ensures exceptions in context_writer.add are logged."""
        repository = mocker.create_autospec(DocumentRepositoryPort, instance=True)
        repository.find_all.return_value = documents

        context_writer = mocker.create_autospec(ContextWriterPort, instance=True)
        context_writer.add.side_effect = Exception('Writer error')

        logger = mocker.create_autospec(LoggerPort)

        use_case = DataEmbeddingUseCase(
            repository=repository,
            context_writer=context_writer,
            logger=logger,
        )

        use_case()

        logger.error.assert_called_with('Error saving document: Writer error - [id="200" - title="Second Doc"]')
