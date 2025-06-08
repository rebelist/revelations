from abc import ABC, abstractmethod
from typing import Iterable

from rebelist.revelations.domain.models import Document


class DocumentRepositoryPort(ABC):
    """Abstract base class for documents repository."""

    @abstractmethod
    def find_all(self) -> Iterable[Document]:
        """Finds documents that were modified since specific date."""
        ...

    @abstractmethod
    def save(self, document: Document) -> None:
        """Saves a document."""
        ...
