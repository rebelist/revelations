from rebelist.revelations.domain import ContextReaderPort, Response, ResponseGeneratorPort


class SemanticSearchUseCase:
    def __init__(self, context_reader: ContextReaderPort, response_generator: ResponseGeneratorPort):
        self.__context_reader = context_reader
        self.__response_generator = response_generator

    def __call__(self, query: str) -> Response:
        """Executes the command."""
        documents = self.__context_reader.search(query, 3)
        response = self.__response_generator.respond(query, documents)

        return response
