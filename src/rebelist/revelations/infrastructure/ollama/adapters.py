from typing import Iterable, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM as Ollama

from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.domain.services import ResponseGeneratorPort


class OllamaAdapter(ResponseGeneratorPort):
    def __init__(self, ollama: Ollama):
        self.__ollama = ollama

    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generates an answer to the given query using the provided context documents."""
        context = '\n\n'.join([f'## Document: {document.title}\n\n{document.content}' for document in documents])

        prompt = super().get_prompt()
        prompt_template = PromptTemplate.from_template(prompt)
        rag_chain = cast(Runnable[dict[str, str], str], prompt_template | self.__ollama | StrOutputParser())

        answer = rag_chain.invoke({'question': question, 'context': context})

        return Response(answer=answer, documents=documents)
