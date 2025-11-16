from datetime import datetime
from typing import Any, Final, Generator, TypeAlias, cast

import pypandoc
from atlassian import Confluence

from rebelist.revelations.domain import ContentProviderPort
from rebelist.revelations.domain.services import LoggerPort

Documents: TypeAlias = list[dict[str, Any]]
Document: TypeAlias = dict[str, Any]


class ConfluenceGateway(ContentProviderPort):
    """Confluence api gateway."""

    PANDOC_OPTIONS: Final[tuple[str, ...]] = ('--wrap=none', '--strip-comments')
    MIN_DOCUMENT_LENGTH: Final[int] = 50

    def __init__(self, client: Confluence, spaces: tuple[str, ...], logger: LoggerPort):
        self.__client = client
        self.__spaces = spaces
        self.__logger = logger

    def fetch(self) -> Generator[dict[str, Any], None, None]:
        """Finds content pages from confluence spaces."""
        for space in self.__spaces:
            yield from self.__fetch_from_space(space)

    def __fetch_from_space(self, space: str) -> Generator[dict[str, Any], None, None]:
        """Finds content pages from a confluence space."""
        documents = cast(
            Documents,
            self.__client.get_all_pages_from_space_as_generator(
                space,
                start=0,
                limit=20,
                expand='body.export_view',
                status='current',
            ),
        )

        for document in documents:
            try:
                html_content = document['body']['export_view']['value']
                markdown_content = pypandoc.convert_text(
                    html_content.strip(),
                    'gfm+hard_line_breaks-raw_html',
                    format='html',
                    extra_args=ConfluenceGateway.PANDOC_OPTIONS,
                ).strip()

                if len(markdown_content) > ConfluenceGateway.MIN_DOCUMENT_LENGTH:
                    page = {
                        'id': document['id'],
                        'title': document['title'],
                        'content': markdown_content,
                        'raw': document,
                        'modified_at': datetime.now(),
                        'url': self.__client.url + document['_links']['tinyui'],
                    }

                    yield page

            except (KeyError, RuntimeError) as e:
                self.__logger.error(f'Skipping page {document["id"]}, operation failed: ({type(e).__name__}) {e}.')
