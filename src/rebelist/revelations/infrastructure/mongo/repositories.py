from typing import Any, Generator, Mapping, TypeAlias

from pymongo.synchronous.collection import Collection as MongoCollection
from pymongo.synchronous.database import Database as MongoDatabase

from rebelist.revelations.domain.models import Document
from rebelist.revelations.domain.repositories import DocumentRepositoryPort

Collection: TypeAlias = MongoCollection[Mapping[str, Any]]
Database: TypeAlias = MongoDatabase[Mapping[str, Any]]


class MongoDocumentRepository(DocumentRepositoryPort):
    """Repository for accessing documents from Mongo."""

    def __init__(self, database: Database, collection_name: str) -> None:
        self.__collection: Collection = database.get_collection(collection_name)

    def save(self, document: Document) -> None:
        """Saves a document."""
        self.__collection.delete_one({'id': document.id})
        self.__collection.insert_one(document.as_dict())

    def find_all(self) -> Generator[Document, None, None]:
        """Finds all documents."""
        cursor = self.__collection.find({})

        try:
            for item in cursor:
                document = Document(
                    id=item['id'],
                    title=item['title'],
                    content=item['content'],
                    modified_at=item['modified_at'],
                    raw=item['raw'],
                    url=item['url'],
                )

                yield document
        finally:
            cursor.close()
