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
    def get_prompt() -> str:
        """Get the RAG template."""
        prompt = (
            'You are a highly-skilled Q&A assistant for the internal documentation of a platform called **Evelin**.\n'
            "Your primary objective is to provide a **concise, accurate, and direct answer** to the user's question, "
            'strictly based **only** on the following provided context.\n\n'
            '**Instructions:**\n'
            '* **Synthesize and Conclude:** '
            'Read and synthesize information across the context to form a complete answer.'
            'If the question asks for an opinion or assessment (e.g., "Is X good?"), '
            'draw a conclusion **only** from the facts presented.\n'
            '* **Directness:** Do not include any introductory phrases like "Based on the context..." or '
            '"The documentation states...". **Start directly with the answer.**\n'
            '* **Constraint:** You **must not** make up any information or use outside knowledge.\n'
            '* **No Relevant Facts:** If **no relevant facts** are present in the context to answer the question, '
            'your response must be the single sentence: '
            '"**I cannot find the answer to your question in the provided documentation.**"\n\n'
            '--- Conversation History ---\n'
            '**{chat_history}**\n'
            '--- Context ---\n'
            '{context}\n'
            '--- Question ---\n'
            '{question}\n\n'
            'Provide the Answer in **Markdown format**, using bullet points, bolding, and headings where '
            'appropriate for clarity and readability.'
        )

        return prompt


class LoggerPort(ABC):
    @abstractmethod
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log information messages."""

    @abstractmethod
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning messages."""

    @abstractmethod
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error messages."""
