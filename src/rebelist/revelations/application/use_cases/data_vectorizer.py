from typing import Final

from rebelist.revelations.domain import ContextWriterPort, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort


class DataVectorizeUseCase:
    CONTENT_LENGTH_LIMIT: Final[int] = 200_000

    def __init__(self, repository: DocumentRepositoryPort, context_writer: ContextWriterPort, logger: LoggerPort):
        self.__repository = repository
        self.__context_writer = context_writer
        self.__logger = logger

    def __call__(self) -> None:
        """Executes the command."""
        try:
            for document in self.__repository.find_all():
                if len(document.content) > DataVectorizeUseCase.CONTENT_LENGTH_LIMIT:
                    self.__logger.warning(f'Skipping large document. id="{document.id}" - title="{document.title}"')
                    continue

                self.__context_writer.add(document)
        except Exception as e:
            self.__logger.error(f'Error in DataVectorizeUseCase: {e}')
            raise
