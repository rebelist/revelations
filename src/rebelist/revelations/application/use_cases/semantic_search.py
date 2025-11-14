from rebelist.revelations.domain import ContextReaderPort, Response, ResponseGeneratorPort
from rebelist.revelations.domain.services import LoggerPort


class SemanticSearchUseCase:
    def __init__(
        self, context_reader: ContextReaderPort, response_generator: ResponseGeneratorPort, logger: LoggerPort
    ):
        self.__context_reader = context_reader
        self.__response_generator = response_generator
        self.__logger = logger

    def __call__(self, query: str) -> Response:
        """Executes the command."""
        try:
            documents = self.__context_reader.search(query, 10)
            response = self.__response_generator.respond(query, documents[:5])
            return response
        except Exception as e:
            self.__logger.error(f'Error in SemanticSearchUseCase: {e}')
            raise
