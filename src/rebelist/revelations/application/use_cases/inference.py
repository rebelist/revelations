from typing import Iterator

from rebelist.revelations.config.settings import RagSettings
from rebelist.revelations.domain import ChatAdapterPort, ContextReaderPort, Response
from rebelist.revelations.domain.services import LoggerPort


class InferenceUseCase:
    def __init__(
        self,
        context_reader: ContextReaderPort,
        chat_adapter: ChatAdapterPort[Iterator[str]],
        settings: RagSettings,
        logger: LoggerPort,
    ):
        self.__context_reader = context_reader
        self.__chat_adapter = chat_adapter
        self.__settings = settings
        self.__logger = logger

    def __call__(self, query: str) -> Response[Iterator[str]]:
        """Executes the use case."""
        try:
            documents = self.__context_reader.search(query, self.__settings.retrieval_limit)
            response = self.__chat_adapter.answer(query, documents[: self.__settings.context_cutoff])
            return response
        except Exception as error:
            self.__logger.error(f'Semantic search has failed: {error}')
            raise
