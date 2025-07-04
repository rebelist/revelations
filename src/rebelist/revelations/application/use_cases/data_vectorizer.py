from rebelist.revelations.domain import ContextWriterPort, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort


class DataVectorizeUseCase:
    def __init__(self, repository: DocumentRepositoryPort, context_writer: ContextWriterPort, logger: LoggerPort):
        self.__repository = repository
        self.__context_writer = context_writer
        self.__logger = logger

    def __call__(self) -> None:
        """Executes the command."""
        try:
            for document in self.__repository.find_all():
                self.__context_writer.add(document)
        except Exception as e:
            self.__logger.error(f'Error in DataVectorizeUseCase: {e}')
            raise
