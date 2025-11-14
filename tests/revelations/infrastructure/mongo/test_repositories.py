from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock.plugin import MockerFixture

from rebelist.revelations.domain.models import Document
from rebelist.revelations.infrastructure.mongo.repositories import MongoDocumentRepository


@pytest.fixture
def document_fixture() -> Document:
    """Provides a sample Document instance."""
    return Document(
        id=123,
        title='Test Title',
        content='This is the content',
        modified_at=datetime(2024, 2, 15, 10, 30, 0),
        raw='',
        url=None,
    )


@pytest.fixture
def mock_collection(mocker: MockerFixture) -> MagicMock:
    """Mocks a MongoDB collection."""
    return mocker.MagicMock()


@pytest.fixture
def mock_database(mocker: MockerFixture, mock_collection: MagicMock) -> MagicMock:
    """Mocks a MongoDB database returning the mocked collection."""
    db = mocker.MagicMock()
    db.get_collection.return_value = mock_collection
    return db


class TestMongoDocumentRepository:
    def test_save_deletes_and_inserts_document(
        self,
        mock_database: MagicMock,
        mock_collection: MagicMock,
        document_fixture: Document,
    ) -> None:
        """It should delete existing document and insert the new one."""
        repo = MongoDocumentRepository(mock_database, 'test-collection')
        repo.save(document_fixture)

        mock_collection.delete_one.assert_called_once_with({'id': document_fixture.id})
        mock_collection.insert_one.assert_called_once_with(document_fixture.as_dict())

    def test_find_all_yields_documents(
        self,
        mock_database: MagicMock,
        mock_collection: MagicMock,
        document_fixture: Document,
        mocker: MockerFixture,
    ) -> None:
        """It should yield Document objects retrieved from MongoDB."""
        doc_dict: dict[str, Any] = document_fixture.as_dict()
        mock_cursor = [doc_dict]

        mock_cursor_obj = mocker.MagicMock()
        mock_cursor_obj.__iter__.return_value = iter(mock_cursor)
        mock_cursor_obj.close = mocker.MagicMock()

        mock_collection.find.return_value = mock_cursor_obj

        repo = MongoDocumentRepository(mock_database, 'test-collection')
        results = list(repo.find_all())

        assert len(results) == 1
        doc = results[0]
        assert isinstance(doc, Document)
        assert doc.id == document_fixture.id
        assert doc.title == document_fixture.title
        assert doc.content == document_fixture.content
        assert doc.modified_at == document_fixture.modified_at
        assert doc.raw == document_fixture.raw

        mock_collection.find.assert_called_once_with({})
        mock_cursor_obj.close.assert_called_once()
