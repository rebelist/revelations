from rebelist.revelations.domain import ContextReaderPort, Response, ResponseGeneratorPort
from rebelist.revelations.infrastructure.monitoring import PerformanceTracker


class SemanticSearchUseCase:
    def __init__(self, context_reader: ContextReaderPort, response_generator: ResponseGeneratorPort):
        self.__context_reader = context_reader
        self.__response_generator = response_generator
        self.__tracker = PerformanceTracker()

    def __call__(self, query: str) -> Response:
        """Executes the command with performance monitoring."""
        print(f"🔍 Processing query: {query[:50]}...")
        
        # Track document retrieval
        self.__tracker.start_timer("document_search")
        # Balanced approach: fetch more documents for better recall, but still optimized
        documents = self.__context_reader.search(query, 25)
        self.__tracker.end_timer("document_search")
        
        # Track response generation
        self.__tracker.start_timer("response_generation")
        response = self.__response_generator.respond(query, documents)
        self.__tracker.end_timer("response_generation")
        
        # Print performance summary
        total_time = sum(self.__tracker.get_summary().values())
        print(f"✅ Total processing time: {total_time:.2f}s")
        
        return response
