from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import batched
from typing import Any, Final, Generator, TypeAlias, Union, cast

from atlassian import Confluence

from rebelist.revelations.domain import ContentProviderPort
from rebelist.revelations.domain.services import LoggerPort

Documents: TypeAlias = list[dict[str, Any]]
Document: TypeAlias = dict[str, Any]


class ConfluenceGateway(ContentProviderPort):
    """Confluence api gateway."""

    MIN_CONTENT_LENGTH: Final[int] = 200
    MAX_WORKERS: Final[int] = 5
    BATCH_SIZE: Final[int] = 500

    def __init__(self, client: Confluence, spaces: tuple[str, ...], logger: LoggerPort):
        self.__client = client
        self.__spaces = spaces
        self.__logger = logger

    def fetch(self) -> Generator[dict[str, Any], None, None]:
        """Finds content pages from confluence spaces."""
        for space in self.__spaces:
            yield from self.__fetch_from_space(space)

    def __fetch_from_space(self, space: str) -> Generator[dict[str, Any], None, None]:
        """Finds content pages from a confluence space using concurrent processing."""
        documents = cast(
            Documents,
            self.__client.get_all_pages_from_space_as_generator(
                space,
                expand='body.export_view,history.lastUpdated',
                status='current',
            ),
        )

        batches = batched(documents, ConfluenceGateway.BATCH_SIZE, strict=False)

        for batch in batches:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                results = executor.map(self.__process_page, batch)

                for result in results:
                    match result:
                        case dict() as page:
                            yield page
                        case None:
                            continue

    def __process_page(self, document: dict[str, Any]) -> Union[dict[str, Any], None]:
        """Worker method running in a separate thread."""
        try:
            body = document.get('body', {}).get('export_view', {})
            content = body.get('value', '')

            if len(content) <= ConfluenceGateway.MIN_CONTENT_LENGTH:
                self.__logger.info(f'Skipping short document. [id={document.get("id")}]')
                return None

            content_bytes = self.__client.get_page_as_pdf(document['id'])

            return {
                'id': document['id'],
                'title': document['title'],
                'content': content_bytes,
                'raw': document,
                'modified_at': datetime.fromisoformat(document['history']['lastUpdated']['when']),
                'url': self.__client.url + document['_links']['tinyui'],
            }

        except Exception as error:
            self.__logger.error(f'Processing document failed [id={document.get("id")}] - {error}')
            return None
