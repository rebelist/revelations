from rebelist.revelations.domain import ContextWriterPort
from rebelist.revelations.infrastructure.mongo import MongoDocumentRepository


class DataVectorizeUseCase:
    def __init__(self, repository: MongoDocumentRepository, context_writer: ContextWriterPort):
        self.__repository = repository
        self.__context_writer = context_writer

    def __call__(self) -> None:
        """Executes the command."""
        for document in self.__repository.find_all():
            self.__context_writer.add(document)
