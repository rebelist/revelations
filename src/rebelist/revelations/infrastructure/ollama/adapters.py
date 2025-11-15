from typing import Any, Final, Iterable, cast

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable, RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from rebelist.revelations.domain import ContextDocument, Response, ResponseGeneratorPort


class OllamaAdapter(ResponseGeneratorPort):
    HISTORY_KEY: Final[str] = 'chat_history'

    def __init__(self, ollama: ChatOllama):
        self.__ollama = ollama
        self.__chat_history = InMemoryChatMessageHistory()
        self.__prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.get_system_template()),
                MessagesPlaceholder(variable_name=OllamaAdapter.HISTORY_KEY),
                HumanMessagePromptTemplate.from_template(self.get_human_template()),
            ]
        )

        base_chain = cast(
            RunnableSerializable[dict[str, Any], str],
            self.__prompt_template | self.__ollama | StrOutputParser(),
        )

        self.__chain: Runnable[dict[str, Any], str] = RunnableWithMessageHistory(
            runnable=base_chain,  # type: ignore
            get_session_history=self.__get_session_history,
            input_messages_key='question',
            history_messages_key=self.HISTORY_KEY,
        )

    def __get_session_history(self) -> InMemoryChatMessageHistory:
        return self.__chat_history

    def respond(self, question: str, documents: Iterable[ContextDocument]) -> Response:
        """Generate an answer given a question and iterable of ContextDocument."""
        context: list[str] = []

        for doc in documents:
            context.append(f'## Document: {doc.title}\nURL: {doc.url}\n\n{doc.content}')

        config: RunnableConfig = {'configurable': {'session_id': 'default'}}

        answer = self.__chain.invoke(
            {'question': question, 'context': '\n\n'.join(context)},
            config=config,
        )

        return Response(answer=answer, documents=documents)
