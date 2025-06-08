from typing import Iterable

from langchain_ollama import OllamaLLM as Ollama

from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.domain.services import ResponseGeneratorPort


class OllamaAdapter(ResponseGeneratorPort):
    def __init__(self, ollama: Ollama):
        self.__ollama = ollama

    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        context_str = '\n\n'.join([f'## Document: {document.title}\n\n{document.content}' for document in documents])

        prompt = (
            f"You are an expert Q&A system for the internal documentation of a system called 'evelin'.\n"
            f'Your task is to provide accurate and concise answers based solely on the provided context from '
            f'Confluence pages.\n'
            f'If the answer is not found in the context, state that you cannot answer the question '
            f'with the given information.\n'
            f'Do not make up information.\n\n'
            f'Context:\n{context_str}\n\n'
            f'Question: {question}\n\n'
            f'Answer:'
        )
        answer = self.__ollama.invoke(prompt)
        return Response(answer=answer, documents=documents)
