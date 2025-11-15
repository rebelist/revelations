from datetime import datetime

from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort


class DataFetchUseCase:
    def __init__(
        self,
        content_provider: ContentProviderPort,
        repository: DocumentRepositoryPort,
        logger: LoggerPort,
    ):
        self.__content_provider = content_provider
        self.__repository = repository
        self.__logger = logger

    def __call__(self) -> None:
        """Executes the use case."""
        try:
            pages = self.__content_provider.fetch()
            for page in pages:
                document = Document(
                    id=page['id'],
                    title=page['title'],
                    content=page['content'],
                    modified_at=datetime.now(),
                    raw=page['raw'],
                    url=page['url'],
                )
                self.__repository.save(document)
        except Exception as e:
            self.__logger.error(f'Data fetch has failed: {e}')
            raise
