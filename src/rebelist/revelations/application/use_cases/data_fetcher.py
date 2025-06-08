from datetime import datetime

from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.infrastructure.confluence import XHTMLParser


class DataFetchUseCase:
    def __init__(
        self,
        content_provider: ContentProviderPort,
        repository: DocumentRepositoryPort,
    ):
        self.__content_provider = content_provider
        self.__repository = repository

    def __call__(self) -> None:
        """Executes the use case."""
        pages = self.__content_provider.fetch()

        for page in pages:
            document = Document(
                id=page['id'],
                title=page['title'],
                content=XHTMLParser(page['content']).text(),
                modified_at=datetime.now(),
                raw=page['raw'],
            )

            self.__repository.save(document)
