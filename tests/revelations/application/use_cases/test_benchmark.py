from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest

from rebelist.revelations.application.use_cases.benchmark import BenchmarkUseCase
from rebelist.revelations.domain import (
    AnswerEvaluatorPort,
    BenchmarkCase,
    BenchmarkScore,
    ChatAdapterPort,
    ContextReaderPort,
    LoggerPort,
    RetrievalEvaluator,
)
from rebelist.revelations.domain.models import FidelityScore, RetrievalScore


class TestBenchmarkUseCase:
    """Test suite for the BenchmarkUseCase class."""

    @pytest.fixture
    def benchmark_cases(self) -> list[BenchmarkCase]:
        """Create benchmark case fixtures."""
        return [
            BenchmarkCase(question='What is AI?', answer='Something magical.', keywords={'ai'}),
            BenchmarkCase(question='What is ML?', answer='Something old.', keywords={'ml'}),
        ]

    @pytest.fixture
    def retrieval_score_fixture(self) -> RetrievalScore:
        """Create a retrieval score fixture."""
        return RetrievalScore(
            mrr=0.8,
            ndcg=0.7,
            keyword_coverage=0.9,
            saturation_at_k=0.85,
        )

    @pytest.fixture
    def fidelity_score_fixture(self) -> FidelityScore:
        """Create a fidelity score fixture."""
        return FidelityScore(
            accuracy=0.75,
            completeness=0.8,
            relevance=0.7,
            feedback='Looks good.',
        )

    @pytest.fixture
    def mock_retrieval_evaluator(self, retrieval_score_fixture: RetrievalScore) -> RetrievalEvaluator:
        """Provides a mocked retrieval evaluator."""
        mock = create_autospec(RetrievalEvaluator, instance=True)
        mock.evaluate.return_value = retrieval_score_fixture
        return mock

    @pytest.fixture
    def mock_answer_evaluator(self, fidelity_score_fixture: FidelityScore) -> AnswerEvaluatorPort:
        """Provides a mocked answer evaluator."""
        mock = create_autospec(AnswerEvaluatorPort, instance=True)
        mock.evaluate.return_value = fidelity_score_fixture
        return mock

    @pytest.fixture
    def mock_context_reader(self) -> ContextReaderPort:
        """Provides a mocked context reader."""
        mock = create_autospec(ContextReaderPort, instance=True)
        mock.search.return_value = ['doc-1', 'doc-2', 'doc-3']
        return mock

    @pytest.fixture
    def mock_chat_adapter(self) -> ChatAdapterPort[str]:
        """Provides a mocked chat adapter."""
        mock = create_autospec(ChatAdapterPort, instance=True)
        mock.answer.return_value.answer = 'Some generated answer'
        return mock

    @pytest.fixture
    def mock_logger(self) -> LoggerPort:
        """Provides a mocked logger."""
        return create_autospec(LoggerPort, instance=True)

    def test_call_returns_aggregated_benchmark_score(
        self,
        benchmark_cases: list[BenchmarkCase],
        retrieval_score_fixture: RetrievalScore,
        fidelity_score_fixture: FidelityScore,
        mock_retrieval_evaluator: RetrievalEvaluator,
        mock_answer_evaluator: AnswerEvaluatorPort,
        mock_context_reader: ContextReaderPort,
        mock_chat_adapter: ChatAdapterPort[str],
        mock_logger: LoggerPort,
    ) -> None:
        """Tests that the use case aggregates retrieval and fidelity scores correctly."""
        use_case = BenchmarkUseCase(
            retrieval_evaluator=mock_retrieval_evaluator,
            answer_evaluator=mock_answer_evaluator,
            context_reader=mock_context_reader,
            chat_adapter=mock_chat_adapter,
            logger=mock_logger,
        )

        result = use_case(benchmark_cases, cutoff=10, limit=20)

        assert isinstance(result, BenchmarkScore)
        assert result.retrieval == retrieval_score_fixture
        assert result.fidelity.accuracy == fidelity_score_fixture.accuracy
        assert result.fidelity.completeness == fidelity_score_fixture.completeness
        assert result.fidelity.relevance == fidelity_score_fixture.relevance

        assert cast(MagicMock, mock_context_reader.search).call_count == len(benchmark_cases)
        assert cast(MagicMock, mock_chat_adapter.answer).call_count == len(benchmark_cases)
        assert cast(MagicMock, mock_retrieval_evaluator.evaluate).call_count == len(benchmark_cases)
        assert cast(MagicMock, mock_answer_evaluator.evaluate).call_count == len(benchmark_cases)

    def test_call_raises_when_cutoff_exceeds_maximum(
        self,
        benchmark_cases: list[BenchmarkCase],
        mock_retrieval_evaluator: RetrievalEvaluator,
        mock_answer_evaluator: AnswerEvaluatorPort,
        mock_context_reader: ContextReaderPort,
        mock_chat_adapter: ChatAdapterPort[str],
        mock_logger: LoggerPort,
    ) -> None:
        """Tests that a ValueError is raised when cutoff exceeds the maximum allowed."""
        use_case = BenchmarkUseCase(
            mock_retrieval_evaluator,
            mock_answer_evaluator,
            mock_context_reader,
            mock_chat_adapter,
            mock_logger,
        )

        with pytest.raises(ValueError, match='Cutoff value must be'):
            use_case(benchmark_cases, cutoff=BenchmarkUseCase.CUTOFF_MAX + 1, limit=10)

    def test_call_raises_when_limit_exceeds_maximum(
        self,
        benchmark_cases: list[BenchmarkCase],
        mock_retrieval_evaluator: RetrievalEvaluator,
        mock_answer_evaluator: AnswerEvaluatorPort,
        mock_context_reader: ContextReaderPort,
        mock_chat_adapter: ChatAdapterPort[str],
        mock_logger: LoggerPort,
    ) -> None:
        """Tests that a ValueError is raised when limit exceeds the maximum allowed."""
        use_case = BenchmarkUseCase(
            mock_retrieval_evaluator,
            mock_answer_evaluator,
            mock_context_reader,
            mock_chat_adapter,
            mock_logger,
        )

        with pytest.raises(ValueError, match='Limit value must be'):
            use_case(benchmark_cases, cutoff=10, limit=BenchmarkUseCase.LIMIT_MAX + 1)

    def test_call_raises_when_cutoff_is_greater_than_limit(
        self,
        benchmark_cases: list[BenchmarkCase],
        mock_retrieval_evaluator: RetrievalEvaluator,
        mock_answer_evaluator: AnswerEvaluatorPort,
        mock_context_reader: ContextReaderPort,
        mock_chat_adapter: ChatAdapterPort[str],
        mock_logger: LoggerPort,
    ) -> None:
        """Tests that a ValueError is raised when cutoff is greater than limit."""
        use_case = BenchmarkUseCase(
            mock_retrieval_evaluator,
            mock_answer_evaluator,
            mock_context_reader,
            mock_chat_adapter,
            mock_logger,
        )

        with pytest.raises(ValueError, match='cutoff should not be greater'):
            use_case(benchmark_cases, cutoff=20, limit=10)

    def test_call_logs_and_reraises_on_unexpected_exception(
        self,
        benchmark_cases: list[BenchmarkCase],
        mock_retrieval_evaluator: RetrievalEvaluator,
        mock_answer_evaluator: AnswerEvaluatorPort,
        mock_context_reader: ContextReaderPort,
        mock_chat_adapter: ChatAdapterPort[str],
        mock_logger: LoggerPort,
    ) -> None:
        """Tests that unexpected errors are logged and re-raised."""
        cast(MagicMock, mock_context_reader.search).side_effect = RuntimeError('Boom')

        use_case = BenchmarkUseCase(
            mock_retrieval_evaluator,
            mock_answer_evaluator,
            mock_context_reader,
            mock_chat_adapter,
            mock_logger,
        )

        with pytest.raises(ValueError, match='No retrieval scores were provided.'):
            use_case(benchmark_cases, cutoff=10, limit=20)

        assert cast(MagicMock, mock_logger.error).call_count == 3
