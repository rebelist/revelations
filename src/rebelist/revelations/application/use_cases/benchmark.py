from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Final

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


class BenchmarkUseCase:
    CUTOFF_MAX: Final[int] = 100
    LIMIT_MAX: Final[int] = 200

    def __init__(
        self,
        retrieval_evaluator: RetrievalEvaluator,
        answer_evaluator: AnswerEvaluatorPort,
        context_reader: ContextReaderPort,
        chat_adapter: ChatAdapterPort[str],
        logger: LoggerPort,
    ):
        self.__retrieval_evaluator = retrieval_evaluator
        self.__answer_evaluator = answer_evaluator
        self.__context_reader = context_reader
        self.__chat_adapter = chat_adapter
        self.__logger = logger

    def __call__(self, benchmark_cases: list[BenchmarkCase], cutoff: int, limit: int) -> BenchmarkScore:
        """Executes the use case."""
        if cutoff > BenchmarkUseCase.CUTOFF_MAX:
            raise ValueError(f'Cutoff value must be ≤ {BenchmarkUseCase.CUTOFF_MAX}, got {cutoff}')

        if limit > BenchmarkUseCase.LIMIT_MAX:
            raise ValueError(f'Limit value must be ≤ {BenchmarkUseCase.LIMIT_MAX}, got {limit}')

        if cutoff > limit:
            raise ValueError('The cutoff should not be greater than the limit.')

        retrieval_scores: list[RetrievalScore] = []
        fidelity_scores: list[FidelityScore] = []
        total_cases = len(benchmark_cases)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._evaluate_case, case, cutoff, limit) for case in benchmark_cases]
                count = 1
                for future in as_completed(futures):
                    try:
                        retrieval_score, fidelity_score = future.result()
                        retrieval_scores.append(retrieval_score)
                        fidelity_scores.append(fidelity_score)
                        self.__logger.info(f'Benchmark case completed - {count}/{total_cases}')
                        count += 1
                    except Exception as error:
                        self.__logger.error(f'Failed to evaluate case: {error}')
                        continue

            avg_retrieval_score = self._aggregate_retrieval_scores(retrieval_scores)
            avg_fidelity_score = self._aggregate_fidelity_scores(fidelity_scores)

            return BenchmarkScore(retrieval=avg_retrieval_score, fidelity=avg_fidelity_score)

        except Exception as error:
            self.__logger.error(f'Benchmark evaluation has failed: {error}')
            raise

    def _evaluate_case(
        self, benchmark_case: BenchmarkCase, cutoff: int, limit: int
    ) -> tuple[RetrievalScore, FidelityScore]:
        documents = self.__context_reader.search(benchmark_case.question, limit)

        response = self.__chat_adapter.answer(benchmark_case.question, documents[:cutoff])

        retrieval_score = self.__retrieval_evaluator.evaluate(benchmark_case, documents, cutoff)
        fidelity_score = self.__answer_evaluator.evaluate(benchmark_case, response.answer)

        return retrieval_score, fidelity_score

    def _aggregate_retrieval_scores(self, retrieval_scores: list[RetrievalScore]) -> RetrievalScore:
        retrieval_scores_total = len(retrieval_scores)
        if not retrieval_scores_total:
            raise ValueError('No retrieval scores were provided.')

        total_mrr = 0.0
        total_ndcg = 0.0
        total_keywords_coverage = 0.0
        total_saturation_at_k = 0.0

        for retrieval_score in retrieval_scores:
            total_mrr += retrieval_score.mrr
            total_ndcg += retrieval_score.ndcg
            total_keywords_coverage += retrieval_score.keyword_coverage
            total_saturation_at_k += retrieval_score.saturation_at_k

        avg_mrr = total_mrr / retrieval_scores_total
        avg_ndcg = total_ndcg / retrieval_scores_total
        avg_coverage = total_keywords_coverage / retrieval_scores_total
        avg_saturation_at_k = total_saturation_at_k / retrieval_scores_total

        return RetrievalScore(
            mrr=avg_mrr, ndcg=avg_ndcg, keyword_coverage=avg_coverage, saturation_at_k=avg_saturation_at_k
        )

    def _aggregate_fidelity_scores(self, fidelity_scores: list[FidelityScore]) -> FidelityScore:
        fidelity_scores_total = len(fidelity_scores)
        if not fidelity_scores_total:
            raise ValueError('No fidelity scores were provided.')

        total_accuracy = 0.0
        total_completeness = 0.0
        total_relevance = 0.0

        for fidelity_score in fidelity_scores:
            total_accuracy += fidelity_score.accuracy
            total_completeness += fidelity_score.completeness
            total_relevance += fidelity_score.relevance

        avg_accuracy = total_accuracy / fidelity_scores_total
        avg_completeness = total_completeness / fidelity_scores_total
        avg_relevance = total_relevance / fidelity_scores_total

        return FidelityScore(
            accuracy=avg_accuracy,
            completeness=avg_completeness,
            relevance=avg_relevance,
            feedback='The combined average of multiple responses.',
        )
