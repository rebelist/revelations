from typing import Final

from rebelist.revelations.domain import ContextWriterPort, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort


class DataEmbeddingUseCase:
    CHARACTERS_LENGTH_LIMIT: Final[int] = 500_000

    def __init__(self, repository: DocumentRepositoryPort, context_writer: ContextWriterPort, logger: LoggerPort):
        self.__repository = repository
        self.__context_writer = context_writer
        self.__logger = logger

    def __call__(self) -> None:
        """Executes the use case."""
        count = 0
        for document in self.__repository.find_all():
            if len(document.content) > DataEmbeddingUseCase.CHARACTERS_LENGTH_LIMIT:
                self.__logger.warning(f'Skipping large document. [id="{document.id}" - title="{document.title}"]')
                continue

            try:
                self.__context_writer.add(document)
                count += 1
            except Exception as error:
                # We don't let one document failure stop the batch
                self.__logger.error(f'Error saving document: {error} - [id="{document.id}" - title="{document.title}"]')

        self.__logger.info(f'Total documents processed successfully: {count}')
