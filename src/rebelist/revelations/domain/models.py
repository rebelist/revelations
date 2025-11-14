from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    content: str
    modified_at: datetime
    raw: str
    url: str | None

    def as_dict(self) -> dict[str, int | str | datetime]:
        """Converts the document to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ContextDocument:
    title: str
    content: str
    modified_at: datetime
    url: str | None = None


@dataclass(frozen=True)
class Response:
    answer: str
    documents: Iterable[ContextDocument]
