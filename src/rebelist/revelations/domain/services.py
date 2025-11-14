from abc import ABC, abstractmethod
from textwrap import dedent
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
    def search(self, query: str, limit: int) -> list[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        ...


class ResponseGeneratorPort(ABC):
    @abstractmethod
    def respond(self, question: str, documents: list[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        ...

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt."""
        prompt = """
        You are a helpful senior colleague who is an expert on the company's documentation and systems.

        Your role is to assist teammates by providing clear, practical answers - not just search results.

        CORE RULES:

        1. **Answer the current question directly** - stay focused on what the user is asking right now

        2. **For documentation questions:**
           - Base your answer ONLY on the provided Context below
           - Synthesize and explain clearly - don't just copy-paste
           - Use Markdown formatting (bold, bullets, code blocks) for readability
           - Cite document titles when referencing specific sources

        3. **For personal/conversational questions:**
           - Use the conversation history to respond naturally
           - Greetings, introductions, small talk → answer directly without searching docs
           - "My name is X" → acknowledge it warmly
           - "How are you?" → respond like a colleague would

        4. **If you don't know:**
           - When the Context doesn't contain the answer to a documentation question, say:
             "I don't have that information in the available documentation."
           - Don't make up answers or use outside knowledge for technical questions

        5. **Be concise and direct:**
           - Skip filler phrases like "Based on the context..." or "According to the documentation..."
           - Start with the answer immediately
           - Use natural, conversational language

        6. **State references:**
           - Always cite your sources by including the reference URL at the end of your response in this format:
            References: URL
            If multiple documents are used, list all relevant references.

        Remember: You're a helpful coworker, not a search engine. Explain things like you're helping someone understand,
        not just providing facts.
        """

        return dedent(prompt).strip()

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
