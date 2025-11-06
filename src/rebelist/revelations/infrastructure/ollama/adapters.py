from typing import Any, Final, Iterable, cast

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
                ('system', self.get_system_prompt()),
                MessagesPlaceholder(variable_name=self.HISTORY_KEY),
                ('user', self.get_user_prompt()),
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
        context = ''
        if documents:
            context = '\n\n'.join([f'## Document: {doc.title}\n\n{doc.content}' for doc in documents])

        config: RunnableConfig = {'configurable': {'session_id': 'default'}}

        answer = self.__chain.invoke(
            {'question': question, 'context': context},
            config=config,
        )
        return Response(answer=answer, documents=documents)
