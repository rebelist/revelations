from abc import ABC, abstractmethod
from typing import Any, Iterable

from rebelist.revelations.domain import ContextDocument, Document, Response


class ContentProviderPort(ABC):
    @abstractmethod
    def fetch(self) -> Iterable[dict[str, Any]]:
        """Fetches raw content from the content source."""
        ...


class ContextWriterPort(ABC):
    @abstractmethod
    def add(self, document: Document) -> None:
        """Saves a context document."""
        ...


class ContextReaderPort(ABC):
    @abstractmethod
    def search(self, query: str, limit: int) -> Iterable[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        ...


class ResponseGeneratorPort(ABC):
    @abstractmethod
    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        ...

    @staticmethod
    def get_prompt(question: str, context: str) -> str:
        """Get the RAG prompt."""
        prompt = (
            f"You are an expert Q&A system for the internal documentation of a platform called 'evelin'.\n"
            f'Your task is to provide accurate answers based solely on the provided context.\n'
            f'If the answer is not found in the context, state that you cannot answer the question or ask to clarify.\n'
            f'Do not make up information.\n\n'
            f'Do not start the answer with something like: based on the provided context, just answer.\n\n'
            f'Context:\n{context}\n\n'
            f'Question: {question}\n\n'
            f'Provide the Answer in Markdown format:'
        )

        return prompt
