from datetime import datetime
from typing import List
from unittest.mock import MagicMock, Mock

import pytest
from langchain_ollama import ChatOllama
from pytest_mock import MockerFixture

from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.infrastructure.ollama.adapters import OllamaAdapter


@pytest.fixture
def mock_ollama() -> Mock:
    """Create a mocked ChatOllama instance."""
    return MagicMock(spec=ChatOllama)


@pytest.fixture
def mock_chain(mocker: MockerFixture) -> Mock:
    """Mock RunnableWithMessageHistory to return a controllable chain instance."""
    mock_chain_instance = MagicMock()
    mocker.patch(
        'rebelist.revelations.infrastructure.ollama.adapters.RunnableWithMessageHistory',
        return_value=mock_chain_instance,
    )
    return mock_chain_instance


@pytest.fixture
def sample_documents() -> List[ContextDocument]:
    """Provides a list of example context documents for testing."""
    return [
        ContextDocument(
            title='Doc 1',
            content='Content of document 1',
            modified_at=datetime(2024, 2, 15, 10, 30, 0),
        ),
        ContextDocument(
            title='Doc 2',
            content='Content of document 2',
            modified_at=datetime(2024, 2, 15, 10, 30, 0),
        ),
    ]


class TestOllamaAdapter:
    """Tests for OllamaAdapter behavior."""

    def test_respond_initializes_successfully(self, mock_ollama: Mock) -> None:
        """Should initialize without errors."""
        adapter = OllamaAdapter(ollama=mock_ollama)
        assert adapter is not None
        assert isinstance(adapter, OllamaAdapter)

    def test_respond_with_documents(
        self,
        mock_ollama: Mock,
        mock_chain: Mock,
        sample_documents: List[ContextDocument],
    ) -> None:
        """Should generate response with context from documents."""
        question = 'What is the answer?'
        expected_answer = 'This is the answer'
        mock_chain.invoke.return_value = expected_answer

        adapter = OllamaAdapter(ollama=mock_ollama)
        response = adapter.respond(question, sample_documents)

        assert isinstance(response, Response)
        assert response.answer == expected_answer
        assert list(response.documents) == sample_documents

        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args

        assert call_args[0][0]['question'] == question
        assert 'Doc 1' in call_args[0][0]['context']
        assert 'Content of document 1' in call_args[0][0]['context']
        assert 'Doc 2' in call_args[0][0]['context']
        assert 'Content of document 2' in call_args[0][0]['context']
        assert call_args[1]['config']['configurable']['session_id'] == 'default'

    def test_respond_without_documents(self, mock_ollama: Mock, mock_chain: Mock) -> None:
        """Should generate response with empty context when no documents provided."""
        question = 'Simple question?'
        expected_answer = 'Simple answer'
        mock_chain.invoke.return_value = expected_answer

        adapter = OllamaAdapter(ollama=mock_ollama)
        response = adapter.respond(question, [])

        assert response.answer == expected_answer
        assert list(response.documents) == []

        call_args = mock_chain.invoke.call_args
        assert call_args[0][0]['context'] == ''
        assert call_args[0][0]['question'] == question

    def test_respond_formats_context_correctly(self, mock_ollama: Mock, mock_chain: Mock) -> None:
        """Should format context with document title and content separated by newlines."""
        documents = [
            ContextDocument(
                title='First',
                content='First content',
                modified_at=datetime(2024, 2, 15, 10, 30, 0),
            ),
            ContextDocument(
                title='Second',
                content='Second content',
                modified_at=datetime(2024, 2, 15, 10, 30, 0),
            ),
        ]
        mock_chain.invoke.return_value = 'answer'

        adapter = OllamaAdapter(ollama=mock_ollama)
        adapter.respond('question', documents)

        call_args = mock_chain.invoke.call_args
        context = call_args[0][0]['context']

        assert '## Document: First' in context
        assert 'First content' in context
        assert '## Document: Second' in context
        assert 'Second content' in context
        assert context.count('\n\n') >= 2

    def test_respond_preserves_session_id(self, mock_ollama: Mock, mock_chain: Mock) -> None:
        """Should use consistent session_id across multiple invocations."""
        mock_chain.invoke.return_value = 'answer'

        adapter = OllamaAdapter(ollama=mock_ollama)
        adapter.respond('question 1', [])
        adapter.respond('question 2', [])

        assert mock_chain.invoke.call_count == 2
        for call_item in mock_chain.invoke.call_args_list:
            assert call_item[1]['config']['configurable']['session_id'] == 'default'

    def test_multiple_responses_maintain_history(self, mock_ollama: Mock, mock_chain: Mock) -> None:
        """Should maintain chat history across multiple response invocations."""
        mock_chain.invoke.side_effect = ['answer1', 'answer2', 'answer3']

        adapter = OllamaAdapter(ollama=mock_ollama)
        adapter.respond('q1', [])
        adapter.respond('q2', [])
        adapter.respond('q3', [])

        assert mock_chain.invoke.call_count == 3
        session_ids = [
            call_item[1]['config']['configurable']['session_id'] for call_item in mock_chain.invoke.call_args_list
        ]
        assert all(sid == 'default' for sid in session_ids)
