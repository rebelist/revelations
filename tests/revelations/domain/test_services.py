from datetime import datetime

import pytest

from rebelist.revelations.domain import BenchmarkCase
from rebelist.revelations.domain.models import ContextDocument, RetrievalScore
from rebelist.revelations.domain.services import RetrievalEvaluator


class TestRetrievalEvaluator:
    """Test suite for the RetrievalEvaluator domain service."""

    @pytest.fixture
    def evaluator(self) -> RetrievalEvaluator:
        """Provide a retrieval evaluator instance."""
        return RetrievalEvaluator()

    @pytest.fixture
    def documents(self) -> list[ContextDocument]:
        """Create a ranked list of context documents."""
        modified_at = datetime(2024, 1, 1, 12, 0, 0)
        return [
            ContextDocument(title='Doc 1', content='Python and AI', modified_at=modified_at),
            ContextDocument(title='Doc 2', content='Machine learning basics', modified_at=modified_at),
            ContextDocument(title='Doc 3', content='Advanced Python topics', modified_at=modified_at),
            ContextDocument(title='Doc 4', content='Unrelated content', modified_at=modified_at),
        ]

    @pytest.fixture
    def benchmark_case(self) -> BenchmarkCase:
        """Create a benchmark case with keywords and an expected answer."""
        return BenchmarkCase(
            question='What is Python used for?',
            answer='Python is used for many things.',
            keywords={'python', 'ai'},
        )

    def test_evaluate_returns_expected_retrieval_score(
        self,
        evaluator: RetrievalEvaluator,
        benchmark_case: BenchmarkCase,
        documents: list[ContextDocument],
    ) -> None:
        """Tests that retrieval metrics are computed correctly for a typical case."""
        result = evaluator.evaluate(benchmark_case, documents, k=3)

        assert isinstance(result, RetrievalScore)

        # MRR:
        # - "python" appears at rank 1
        # - "AI" appears at rank 1
        assert result.mrr == pytest.approx(1.0)

        # nDCG should be positive and bounded
        assert 0.0 < result.ndcg <= 1.0

        # Both keywords appear in top-k → 100%
        assert result.keyword_coverage == 100.0

        # Saturation:
        # - python: 2 relevant total, 1 in top-k
        # - AI: 1 relevant total, 1 in top-k
        # avg = (1/2 + 1/1) / 2 = 0.75
        assert result.saturation_at_k == pytest.approx(1.0)

    def test_evaluate_returns_zero_scores_when_no_keywords_match(
        self,
        evaluator: RetrievalEvaluator,
        documents: list[ContextDocument],
    ) -> None:
        """Tests that all metrics are zero when no keywords appear in documents."""
        benchmark_case = BenchmarkCase(
            question='What is Rust?',
            answer='Rust is a systems programming language.',
            keywords={'rust'},
        )

        result = evaluator.evaluate(benchmark_case, documents, k=3)

        assert result.mrr == 0.0
        assert result.ndcg == 0.0
        assert result.keyword_coverage == 0.0
        assert result.saturation_at_k == 0.0

    def test_evaluate_penalizes_late_relevant_documents(
        self,
        evaluator: RetrievalEvaluator,
    ) -> None:
        """Tests that ranking quality is penalized when relevant documents appear late."""
        modified_at = datetime(2024, 1, 1, 12, 0, 0)
        documents = [
            ContextDocument(title='Doc 1', content='irrelevant', modified_at=modified_at),
            ContextDocument(title='Doc 2', content='irrelevant', modified_at=modified_at),
            ContextDocument(title='Doc 3', content='keyword', modified_at=modified_at),
        ]

        benchmark_case = BenchmarkCase(
            question='Test question',
            answer='Expected answer.',
            keywords={'keyword'},
        )

        result = evaluator.evaluate(benchmark_case, documents, k=3)

        # First relevant document at rank 3 → MRR = 1/3
        assert result.mrr == pytest.approx(1 / 3)

        # nDCG should be < 1 due to suboptimal ranking
        assert 0.0 < result.ndcg < 1.0

        # Keyword is present → coverage 100%
        assert result.keyword_coverage == 100.0

    def test_evaluate_raises_when_k_is_not_positive(
        self,
        evaluator: RetrievalEvaluator,
        benchmark_case: BenchmarkCase,
        documents: list[ContextDocument],
    ) -> None:
        """Tests that a ValueError is raised when k is not a positive integer."""
        with pytest.raises(ValueError, match='k must be a positive integer'):
            evaluator.evaluate(benchmark_case, documents, k=0)
