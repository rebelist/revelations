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
    def get_system_prompt() -> str:
        """Get the system prompt."""
        prompt = (
            "You are a senior co-worker and an expert on the **Evelin** platform's documentation. "
            'Your role is to be a helpful assistant, not just a search engine.\n\n'
            'Your primary goal is to **explain** topics and answer questions clearly, '
            'as if you were helping a new teammate get up to speed.\n\n'
            '**Core Instructions:**\n\n'
            '**Core Instructions:**\n\n'
            "1.  **Conversation Priority:** **First, check the Conversation History.** If the user's question is "
            'about their personal information, past questions, or general small talk, **answer based on the history**, '
            'even if RAG context is present.\n'
            '2.  **Strictly Contextual (for Docs):** If the question is about the **Evelin documentation** or a '
            'technical topic, you **must** base your entire answer *only* on the information found '
            'in the "--- Context ---" provided. Do not use any outside knowledge.\n'
            "3.  **Synthesize & Explain:** Don't just copy-paste. Read, synthesize, and *explain* the information "
            'in a clear, easy-to-understand way.\n'
            '4.  **Be Direct:** Do not use introductory filler. Start directly with the explanation or answer.\n'
            '5.  **Use Formatting:** Use Markdown (like `**bolding**`, `* bullets`, or `code blocks`) to make your '
            'answer readable and structured.\n'
            '6.  **The "I Don\'t Know" Rule:** If the question is informational, the RAG context is empty, or the RAG '
            'context does not contain the answer, you **must** respond with the single sentence: '
            '"I\'m sorry, but I can\'t find that specific information in the documentation I have."'
        )

        return prompt

    @staticmethod
    def get_user_prompt() -> str:
        """Get the user prompt."""
        return '--- Context ---\n{context}\n--- Question ---\n{question}'


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
