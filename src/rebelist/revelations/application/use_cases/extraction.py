from rebelist.revelations.domain import ContentProviderPort, Document, DocumentRepositoryPort
from rebelist.revelations.domain.services import LoggerPort, PdfConverterPort


class DataExtractionUseCase:
    def __init__(
        self,
        content_provider: ContentProviderPort,
        repository: DocumentRepositoryPort,
        converter: PdfConverterPort,
        logger: LoggerPort,
    ):
        self.__content_provider = content_provider
        self.__repository = repository
        self.__converter = converter
        self.__logger = logger

    def __call__(self) -> None:
        """Executes the use case."""
        documents = self.__content_provider.fetch()
        for raw_document in documents:
            try:
                document = Document(
                    id=raw_document['id'],
                    title=raw_document['title'],
                    content=self.__converter.pdf_to_markdown(raw_document['content']),
                    modified_at=raw_document['modified_at'],
                    raw=raw_document['raw'],
                    url=raw_document['url'],
                )

                self.__repository.save(document)

            except Exception as error:
                self.__logger.error(f'Failed fetching document. {type(error)} - {error}')
