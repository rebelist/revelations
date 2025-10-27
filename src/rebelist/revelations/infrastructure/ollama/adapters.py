from typing import Iterable
from functools import lru_cache

from langchain_ollama import OllamaLLM as Ollama

from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.domain.services import ResponseGeneratorPort


class OllamaAdapter(ResponseGeneratorPort):
    def __init__(self, ollama: Ollama):
        self.__ollama = ollama

    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        # Use more documents for better context, but still limit to prevent overly long prompts
        limited_docs = list(documents)[:8]
        context = '\n\n'.join([f'## Document: {document.title}\n\n{document.content}' for document in limited_docs])

        prompt = ResponseGeneratorPort.get_prompt(question, context)
        answer = self.__ollama.invoke(prompt)

        return Response(answer=answer, documents=limited_docs)
