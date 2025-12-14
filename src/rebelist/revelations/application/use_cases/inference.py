from typing import Final, Iterator

from rebelist.revelations.domain import ChatAdapterPort, ContextReaderPort, Response
from rebelist.revelations.domain.services import LoggerPort


class InferenceUseCase:
    CONTEXT_LIMIT: Final[int] = 10

    def __init__(
        self, context_reader: ContextReaderPort, chat_adapter: ChatAdapterPort[Iterator[str]], logger: LoggerPort
    ):
        self.__context_reader = context_reader
        self.__chat_adapter = chat_adapter
        self.__logger = logger

    def __call__(self, query: str) -> Response[Iterator[str]]:
        """Executes the use case."""
        try:
            documents = self.__context_reader.search(query, 30)
            response = self.__chat_adapter.answer(query, documents[: InferenceUseCase.CONTEXT_LIMIT])
            return response
        except Exception as error:
            self.__logger.error(f'Semantic search has failed: {error}')
            raise
