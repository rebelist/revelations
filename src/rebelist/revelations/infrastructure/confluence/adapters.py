from datetime import datetime
from typing import Any, Generator, TypeAlias, cast

import pypandoc
from atlassian import Confluence

from rebelist.revelations.domain import ContentProviderPort

Documents: TypeAlias = list[dict[str, Any]]
Document: TypeAlias = dict[str, Any]


class ConfluenceGateway(ContentProviderPort):
    """Confluence api gateway."""

    def __init__(self, client: Confluence, space: str):
        self.__client = client
        self.__space = space

    def fetch(self) -> Generator[dict[str, Any], None, None]:
        """Finds content_provider pages in a given space."""
        documents = cast(
            Documents,
            self.__client.get_all_pages_from_space_as_generator(
                self.__space,
                start=0,
                limit=20,
                expand='body.view',
                status='current',
            ),
        )

        for document in documents:
            try:
                html_content = document['body']['view']['value']
                markdown_content = pypandoc.convert_text(html_content, 'gfm', format='html')

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
                # Handle pages that might be malformed or pandoc errors
                print(f'Skipping page {document.get("id")}: Could not parse HTML. Error: {e}')
