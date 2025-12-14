from datetime import datetime
from typing import Any, Iterable, List, cast
from unittest.mock import MagicMock, Mock, create_autospec

import pytest
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pytest_mock import MockerFixture

from rebelist.revelations.domain import (
    AnswerEvaluatorPort,
    BenchmarkCase,
    ChatAdapterPort,
    ContextDocument,
    FidelityScore,
    PromptConfig,
    Response,
)
from rebelist.revelations.infrastructure.ollama.adapters import (
    OllamaAnswerEvaluator,
    OllamaMemoryChatAdapter,
    OllamaStatelessChatAdapter,
)


@pytest.fixture
def mock_ollama() -> Mock:
    """Create a mocked ChatOllama instance."""
    return MagicMock(spec=ChatOllama)


@pytest.fixture
def prompt_config() -> PromptConfig:
    """Prompt configuration used by the adapter."""
    return PromptConfig(
        system_template='system template',
        human_template='human template',
    )


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


class TestOllamaMemoryChatAdapter:
    """Tests for OllamaMemoryChatAdapter behavior."""

    @pytest.fixture
    def mock_memory_chain(self, mocker: MockerFixture) -> Mock:
        """Mock RunnableWithMessageHistory to return a controllable chain instance."""
        mock_chain_instance = MagicMock()
        mocker.patch(
            'rebelist.revelations.infrastructure.ollama.adapters.RunnableWithMessageHistory',
            return_value=mock_chain_instance,
        )
        return mock_chain_instance

    def test_respond_initializes_successfully(self, mock_ollama: Mock) -> None:
        """Should initialize without errors."""
        prompt_config = PromptConfig(system_template='First content', human_template='Second content')
        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        assert adapter is not None
        assert isinstance(adapter, OllamaMemoryChatAdapter)

    def test_respond_with_documents(
        self,
        mock_ollama: Mock,
        mock_memory_chain: Mock,
        sample_documents: List[ContextDocument],
        prompt_config: PromptConfig,
    ) -> None:
        """Should generate response with context from documents."""
        question = 'What is the answer?'
        expected_answer = 'This is the answer'
        mock_memory_chain.stream.return_value = expected_answer

        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        response = adapter.answer(question, sample_documents)

        assert isinstance(response, Response)
        assert response.answer == expected_answer
        assert list(response.documents) == sample_documents

        mock_memory_chain.stream.assert_called_once()
        call_args = mock_memory_chain.stream.call_args

        assert call_args[0][0]['question'] == question
        assert 'Doc 1' in call_args[0][0]['context']
        assert 'Content of document 1' in call_args[0][0]['context']
        assert 'Doc 2' in call_args[0][0]['context']
        assert 'Content of document 2' in call_args[0][0]['context']
        assert call_args[1]['config']['configurable']['session_id'] == 'default'

    def test_respond_without_documents(
        self, mock_ollama: Mock, mock_memory_chain: Mock, prompt_config: PromptConfig
    ) -> None:
        """Should generate response with empty context when no documents provided."""
        question = 'Simple question?'
        expected_answer = 'Simple answer'
        mock_memory_chain.stream.return_value = expected_answer

        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        response = adapter.answer(question, [])

        assert response.answer == expected_answer
        assert list(response.documents) == []

        call_args = mock_memory_chain.stream.call_args
        assert call_args[0][0]['context'] == ''
        assert call_args[0][0]['question'] == question

    def test_respond_formats_context_correctly(
        self, mock_ollama: Mock, mock_memory_chain: Mock, prompt_config: PromptConfig
    ) -> None:
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
        mock_memory_chain.stream.return_value = 'answer'

        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        adapter.answer('question', documents)

        call_args = mock_memory_chain.stream.call_args
        context = call_args[0][0]['context']

        assert '## Document: First' in context
        assert 'First content' in context
        assert '## Document: Second' in context
        assert 'Second content' in context
        assert context.count('\n\n') >= 2

    def test_respond_preserves_session_id(
        self, mock_ollama: Mock, mock_memory_chain: Mock, prompt_config: PromptConfig
    ) -> None:
        """Should use consistent session_id across multiple invocations."""
        mock_memory_chain.stream.return_value = 'answer'

        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        adapter.answer('question 1', [])
        adapter.answer('question 2', [])

        assert mock_memory_chain.stream.call_count == 2
        for call_item in mock_memory_chain.stream.call_args_list:
            assert call_item[1]['config']['configurable']['session_id'] == 'default'

    def test_multiple_responses_maintain_history(
        self, mock_ollama: Mock, mock_memory_chain: Mock, prompt_config: PromptConfig
    ) -> None:
        """Should maintain chat history across multiple response invocations."""
        mock_memory_chain.stream.side_effect = ['answer1', 'answer2', 'answer3']

        adapter = OllamaMemoryChatAdapter(mock_ollama, prompt_config)
        adapter.answer('q1', [])
        adapter.answer('q2', [])
        adapter.answer('q3', [])

        assert mock_memory_chain.stream.call_count == 3
        session_ids = [
            call_item[1]['config']['configurable']['session_id']
            for call_item in mock_memory_chain.stream.call_args_list
        ]
        assert all(sid == 'default' for sid in session_ids)


class TestOllamaStatelessChatAdapter:
    """Tests for OllamaStatelessChatAdapter."""

    @pytest.fixture
    def runnable_chain(self) -> MagicMock:
        """Mocked runnable chain returning a fixed answer."""
        chain = create_autospec(Runnable, instance=True)
        chain.invoke.return_value = 'generated answer'
        return chain

    @pytest.fixture
    def adapter(
        self,
        prompt_config: PromptConfig,
        runnable_chain: Runnable[dict[str, object], str],
    ) -> OllamaStatelessChatAdapter:
        """Adapter with an injected runnable chain."""
        ollama = MagicMock()

        adapter = OllamaStatelessChatAdapter(
            ollama=ollama,
            prompt_config=prompt_config,
        )

        adapter._OllamaStatelessChatAdapter__chain = runnable_chain  # type: ignore[attr-defined]

        return adapter

    def test_answer_returns_response_with_expected_answer_and_documents(
        self,
        adapter: OllamaStatelessChatAdapter,
        runnable_chain: MagicMock,
        sample_documents: Iterable[ContextDocument],
    ) -> None:
        """Answer returns a Response containing the generated answer and input documents."""
        question = 'What is the answer?'

        response = adapter.answer(question, sample_documents)

        assert isinstance(response, Response)
        assert response.answer == 'generated answer'
        assert response.documents is sample_documents

        chain = runnable_chain  # explicit for type checkers
        chain.invoke.assert_called_once()

    def test_answer_builds_expected_context_and_invokes_chain(
        self,
        adapter: OllamaStatelessChatAdapter,
        runnable_chain: MagicMock,
        sample_documents: Iterable[ContextDocument],
    ) -> None:
        """Answer builds the expected context and passes it to the runnable chain."""
        question = 'Explain this'

        adapter.answer(question, sample_documents)

        expected_context = (
            '## Document: Doc 1\n'
            'URL: None\n\n'
            'Content of document 1\n\n'
            '## Document: Doc 2\n'
            'URL: None\n\n'
            'Content of document 2'
        )

        chain = runnable_chain
        chain.invoke.assert_called_once_with(
            {
                ChatAdapterPort.HUMAN_TEMPLATE_INPUT_KEY: question,
                ChatAdapterPort.HUMAN_TEMPLATE_CONTEXT_KEY: expected_context,
            }
        )


class TestOllamaAnswerEvaluator:
    """Test suite for the OllamaAnswerEvaluator class."""

    @pytest.fixture
    def benchmark_case(self) -> BenchmarkCase:
        """Provides a benchmark case."""
        return BenchmarkCase(
            question='What is polymorphism?',
            answer='Polymorphism allows objects of different types to be treated uniformly.',
            keywords={'polymorphism'},
        )

    @pytest.fixture
    def fidelity_score(self) -> FidelityScore:
        """Provides a FidelityScore returned by the LLM judge."""
        return FidelityScore(
            accuracy=4.0,
            completeness=3.5,
            relevance=4.5,
            feedback='The answer is mostly accurate but misses some nuances.',
        )

    @pytest.fixture
    def runnable_chain(self, fidelity_score: FidelityScore) -> Runnable[dict[str, Any], FidelityScore]:
        """Provides a mocked runnable chain."""
        chain = create_autospec(Runnable, instance=True)
        cast(MagicMock, chain).invoke.return_value = fidelity_score
        return chain

    def test_evaluate_invokes_chain_and_returns_fidelity_score(
        self,
        benchmark_case: BenchmarkCase,
        fidelity_score: FidelityScore,
        prompt_config: PromptConfig,
        mock_ollama: ChatOllama,
        runnable_chain: Runnable[dict[str, Any], FidelityScore],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensures evaluate invokes the chain with correct inputs and returns the FidelityScore."""
        evaluator = OllamaAnswerEvaluator(mock_ollama, prompt_config)

        # Replace the internally constructed chain
        monkeypatch.setattr(evaluator, '_OllamaAnswerEvaluator__chain', runnable_chain)

        answer = 'Polymorphism allows different objects to respond to the same message.'

        result = evaluator.evaluate(benchmark_case, answer)

        invoke = cast(MagicMock, runnable_chain.invoke)
        invoke.assert_called_once_with(
            {
                AnswerEvaluatorPort.HUMAN_TEMPLATE_QUESTION_KEY: benchmark_case.question,
                AnswerEvaluatorPort.HUMAN_TEMPLATE_ANSWER_KEY: answer,
                AnswerEvaluatorPort.HUMAN_TEMPLATE_REFERENCE_KEY: benchmark_case.answer,
            }
        )

        assert result is fidelity_score
