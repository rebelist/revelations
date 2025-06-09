from typing import Iterable

from langchain_ollama import OllamaLLM as Ollama

from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.domain.services import ResponseGeneratorPort


class OllamaAdapter(ResponseGeneratorPort):
    def __init__(self, ollama: Ollama):
        self.__ollama = ollama

    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        context = '\n\n'.join([f'## Document: {document.title}\n\n{document.content}' for document in documents])

        prompt = ResponseGeneratorPort.get_prompt(question, context)
        answer = self.__ollama.invoke(prompt)

        return Response(answer=answer, documents=documents)
