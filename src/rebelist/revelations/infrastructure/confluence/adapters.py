from datetime import datetime
from typing import Any, Generator, TypeAlias, cast

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
            self.__client.get_all_pages_from_space(
                self.__space, start=0, limit=20, expand='body.storage', status='current'
            ),
        )

        for document in documents:
            content = cast(Document, self.__client.convert_storage_to_view(document['body']['storage']['value']))

            page = {
                'id': document['id'],
                'title': document['title'],
                'content': content['value'],
                'raw': document,
                'modified_at': datetime.now(),
            }

            yield page
