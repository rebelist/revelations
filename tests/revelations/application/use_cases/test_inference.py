from datetime import datetime
from unittest.mock import MagicMock, create_autospec

import pytest

from rebelist.revelations.application.use_cases.inference import InferenceUseCase
from rebelist.revelations.config.settings import RagSettings
from rebelist.revelations.domain import ChatAdapterPort, ContextDocument, ContextReaderPort, Response
from rebelist.revelations.domain.services import LoggerPort


class TestInferenceUseCase:
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
    def response_fixture(self) -> MagicMock:
        """Create a response fixture."""
        return create_autospec(Response, instance=True)

    @pytest.fixture
    def rag_settings_fixture(self) -> RagSettings:
        """Create a response fixture."""
        return RagSettings(retrieval_limit=20)

    @pytest.fixture
    def mock_context_reader(self, document_fixtures: list[ContextDocument]) -> MagicMock:
        """Provides a mocked context reader port."""
        mock = create_autospec(ContextReaderPort, instance=True)
        mock.search.return_value = document_fixtures
        return mock

    @pytest.fixture
    def mock_chat_adapter(self, response_fixture: MagicMock) -> MagicMock:
        """Provides a mocked response generator port."""
        mock = create_autospec(ChatAdapterPort, instance=True)
        mock.answer.return_value = response_fixture
        return mock

    def test_call_returns_expected_response(
        self,
        document_fixtures: list[ContextDocument],
        response_fixture: MagicMock,
        rag_settings_fixture: RagSettings,
        mock_context_reader: MagicMock,
        mock_chat_adapter: MagicMock,
    ) -> None:
        """Tests that the __call__ method returns the correct Response based on mocks."""
        mock_logger = create_autospec(LoggerPort)
        use_case = InferenceUseCase(mock_context_reader, mock_chat_adapter, rag_settings_fixture, mock_logger)
        query = 'What is quantum entanglement?'

        result = use_case(query)

        mock_context_reader.search.assert_called_once_with(query, 20)
        mock_chat_adapter.answer.assert_called_once_with(query, document_fixtures)

        assert result is response_fixture

    def test_error_in_context_reader_is_handled(
        self,
        mock_chat_adapter: MagicMock,
        rag_settings_fixture: RagSettings,
    ) -> None:
        """Ensures that exceptions in context_reader.search are caught and re-raised."""
        mock_context_reader = create_autospec(ContextReaderPort, instance=True)
        mock_logger = create_autospec(LoggerPort)
        mock_context_reader.search.side_effect = Exception('ContextReader error')
        use_case = InferenceUseCase(mock_context_reader, mock_chat_adapter, rag_settings_fixture, mock_logger)
        with pytest.raises(Exception, match='ContextReader error'):
            use_case('test query')

    def test_error_in_response_generator_is_handled(
        self, mock_context_reader: MagicMock, rag_settings_fixture: RagSettings
    ) -> None:
        """Ensures that exceptions in response_generator.respond are caught and re-raised."""
        mock_chat_adapter = create_autospec(ChatAdapterPort, instance=True)
        mock_logger = create_autospec(LoggerPort)
        mock_context_reader.search.return_value = []
        mock_chat_adapter.answer.side_effect = Exception('ResponseGenerator error')
        use_case = InferenceUseCase(mock_context_reader, mock_chat_adapter, rag_settings_fixture, mock_logger)
        with pytest.raises(Exception, match='ResponseGenerator error'):
            use_case('test query')
