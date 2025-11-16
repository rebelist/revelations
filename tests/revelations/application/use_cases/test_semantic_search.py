from datetime import datetime
from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest

from rebelist.revelations.application.use_cases.semantic_search import SemanticSearchUseCase
from rebelist.revelations.domain import ContextDocument, ContextReaderPort, Response, ResponseGeneratorPort
from rebelist.revelations.domain.services import LoggerPort


class TestSemanticSearchUseCase:
    """Test suite for the SemanticSearchUseCase class."""

    @pytest.fixture
    def document_fixtures(self) -> list[ContextDocument]:
        """Create document fixtures."""
        modified_at = datetime(2024, 2, 15, 10, 30, 0)
        return [
            ContextDocument(
                title='First Doc',
                content='Some processed content',
                modified_at=modified_at,
            ),
            ContextDocument(
                title='Second Doc',
                content='Another one',
                modified_at=modified_at,
            ),
        ]

    @pytest.fixture
    def response_fixture(self) -> Response:
        """Create a response fixture."""
        return create_autospec(Response, instance=True)

    @pytest.fixture
    def mock_context_reader(self, document_fixtures: list[ContextDocument]) -> ContextReaderPort:
        """Provides a mocked context reader port."""
        mock = create_autospec(ContextReaderPort, instance=True)
        mock.search.return_value = document_fixtures
        return mock

    @pytest.fixture
    def mock_response_generator(self, response_fixture: Response) -> ResponseGeneratorPort:
        """Provides a mocked response generator port."""
        mock = create_autospec(ResponseGeneratorPort, instance=True)
        mock.respond.return_value = response_fixture
        return mock

    def test_call_returns_expected_response(
        self,
        document_fixtures: list[ContextDocument],
        response_fixture: Response,
        mock_context_reader: ContextReaderPort,
        mock_response_generator: ResponseGeneratorPort,
    ) -> None:
        """Tests that the __call__ method returns the correct Response based on mocks."""
        mock_logger = create_autospec(LoggerPort)
        use_case = SemanticSearchUseCase(mock_context_reader, mock_response_generator, mock_logger)
        query = 'What is quantum entanglement?'

        result = use_case(query)

        cast(MagicMock, mock_context_reader.search).assert_called_once_with(query, 20)
        cast(MagicMock, mock_response_generator.respond).assert_called_once_with(query, document_fixtures)

        assert result is response_fixture

    def test_error_in_context_reader_is_handled(self, mock_response_generator: ResponseGeneratorPort) -> None:
        """Ensures that exceptions in context_reader.search are caught and re-raised."""
        mock_context_reader: MagicMock = create_autospec(ContextReaderPort, instance=True)
        mock_logger: MagicMock = create_autospec(LoggerPort)
        mock_context_reader.search.side_effect = Exception('ContextReader error')
        use_case = SemanticSearchUseCase(mock_context_reader, mock_response_generator, mock_logger)
        with pytest.raises(Exception, match='ContextReader error'):
            use_case('test query')

    def test_error_in_response_generator_is_handled(self, mock_context_reader: ContextReaderPort) -> None:
        """Ensures that exceptions in response_generator.respond are caught and re-raised."""
        mock_response_generator: MagicMock = create_autospec(ResponseGeneratorPort, instance=True)
        mock_logger = create_autospec(LoggerPort)
        mock_context_reader.search = MagicMock(return_value=[])
        mock_response_generator.respond.side_effect = Exception('ResponseGenerator error')
        use_case = SemanticSearchUseCase(mock_context_reader, mock_response_generator, mock_logger)
        with pytest.raises(Exception, match='ResponseGenerator error'):
            use_case('test query')
