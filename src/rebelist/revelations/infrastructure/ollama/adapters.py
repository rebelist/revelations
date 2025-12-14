from typing import Any, Final, Iterable, Iterator, cast

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSequence,
    RunnableWithMessageHistory,
)
from langchain_ollama import ChatOllama

from rebelist.revelations.domain import ChatAdapterPort, ContextDocument, Response
from rebelist.revelations.domain.models import BenchmarkCase, FidelityScore, PromptConfig
from rebelist.revelations.domain.services import AnswerEvaluatorPort


class OllamaMemoryChatAdapter(ChatAdapterPort[Iterator[str]]):
    HISTORY_KEY: Final[str] = 'chat_history'

    def __init__(self, ollama: ChatOllama, prompt_config: PromptConfig):
        self.__chat_history = InMemoryChatMessageHistory()
        self.__prompt_config = prompt_config
        self.__chain = self.__build_chain(ollama)

    def __build_chain(self, ollama: ChatOllama) -> Runnable[dict[str, Any], str]:
        """Builds a chain wrapped with history management."""
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.__prompt_config.system_template),
                MessagesPlaceholder(variable_name=OllamaMemoryChatAdapter.HISTORY_KEY),
                HumanMessagePromptTemplate.from_template(self.__prompt_config.human_template),
            ]
        )

        base_chain = cast(
            RunnableSequence[dict[str, Any], str],
            prompt_template | ollama | StrOutputParser(),
        )

        return RunnableWithMessageHistory(
            runnable=base_chain,  # type: ignore
            get_session_history=self.__get_session_history,
            input_messages_key=ChatAdapterPort.HUMAN_TEMPLATE_INPUT_KEY,
            history_messages_key=self.HISTORY_KEY,
        )

    def __get_session_history(self) -> InMemoryChatMessageHistory:
        """Get memory session history."""
        return self.__chat_history

    def answer(self, question: str, documents: Iterable[ContextDocument]) -> Response[Iterator[str]]:
        """Generate an answer given a question and iterable of ContextDocument."""
        context: list[str] = []

        for doc in documents:
            context.append(f'## Document: {doc.title}\nURL: {doc.url}\n\n{doc.content}')

        config: RunnableConfig = {'configurable': {'session_id': 'default'}}

        answer: Iterator[str] = self.__chain.stream(
            {
                ChatAdapterPort.HUMAN_TEMPLATE_INPUT_KEY: question,
                ChatAdapterPort.HUMAN_TEMPLATE_CONTEXT_KEY: '\n\n'.join(context),
            },
            config=config,
        )

        return Response[Iterator[str]](answer=answer, documents=documents)


class OllamaStatelessChatAdapter(ChatAdapterPort[str]):
    def __init__(self, ollama: ChatOllama, prompt_config: PromptConfig):
        self.__prompt_config = prompt_config
        self.__chain = self.__build_chain(ollama)

    def __build_chain(self, ollama: ChatOllama) -> Runnable[dict[str, Any], str]:
        """Builds a runnable chain."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.__prompt_config.system_template),
                HumanMessagePromptTemplate.from_template(self.__prompt_config.human_template),
            ]
        )

        return cast(
            RunnableSequence[dict[str, Any], str],
            prompt_template | ollama | StrOutputParser(),
        )

    def answer(self, question: str, documents: Iterable[ContextDocument]) -> Response[str]:
        """Generate an answer given a question and iterable of ContextDocument."""
        context: list[str] = []

        for doc in documents:
            context.append(f'## Document: {doc.title}\nURL: {doc.url}\n\n{doc.content}')

        answer: str = self.__chain.invoke(
            {
                ChatAdapterPort.HUMAN_TEMPLATE_INPUT_KEY: question,
                ChatAdapterPort.HUMAN_TEMPLATE_CONTEXT_KEY: '\n\n'.join(context),
            },
        )

        return Response[str](answer=answer, documents=documents)


class OllamaAnswerEvaluator(AnswerEvaluatorPort):
    def __init__(self, ollama: ChatOllama, prompt_config: PromptConfig):
        self.__prompt_config = prompt_config
        self.__chain = self.__build_chain(ollama)

    def __build_chain(self, ollama: ChatOllama) -> Runnable[dict[str, Any], FidelityScore]:
        """Builds a runnable chain."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.__prompt_config.system_template),
                HumanMessagePromptTemplate.from_template(self.__prompt_config.human_template),
            ]
        )

        llm = cast(
            Runnable[dict[str, Any], FidelityScore],
            ollama.with_structured_output(FidelityScore),
        )

        return cast(
            RunnableSequence[dict[str, Any], FidelityScore],
            prompt_template | llm,
        )

    def evaluate(self, benchmark_case: BenchmarkCase, answer: str) -> FidelityScore:
        """Evaluate answer quality using LLM as a judge."""
        evaluation: FidelityScore = self.__chain.invoke(
            {
                AnswerEvaluatorPort.HUMAN_TEMPLATE_QUESTION_KEY: benchmark_case.question,
                AnswerEvaluatorPort.HUMAN_TEMPLATE_ANSWER_KEY: answer,
                AnswerEvaluatorPort.HUMAN_TEMPLATE_REFERENCE_KEY: benchmark_case.answer,
            },
        )

        return evaluation
