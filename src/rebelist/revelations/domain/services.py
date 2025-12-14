import math
from abc import ABC, abstractmethod
from typing import Any, Final, Iterable

from rebelist.revelations.domain import ContextDocument, Document, Response
from rebelist.revelations.domain.models import BenchmarkCase, FidelityScore, RetrievalScore


class ContentProviderPort(ABC):
    @abstractmethod
    def fetch(self) -> Iterable[dict[str, Any]]:
        """Fetches raw content from the content source."""
        ...


class ContextWriterPort(ABC):
    @abstractmethod
    def add(self, document: Document) -> None:
        """Saves a context document."""
        ...


class ContextReaderPort(ABC):
    @abstractmethod
    def search(self, query: str, limit: int) -> list[ContextDocument]:
        """Searches for context documents based on a query embedding."""
        ...


class ChatAdapterPort[T](ABC):
    HUMAN_TEMPLATE_INPUT_KEY: Final[str] = 'question'
    HUMAN_TEMPLATE_CONTEXT_KEY: Final[str] = 'context'

    @abstractmethod
    def answer(self, question: str, documents: Iterable[ContextDocument]) -> Response[T]:
        """Generates an answer to the given query using the provided context documents."""
        ...


class LoggerPort(ABC):
    @abstractmethod
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log information messages."""

    @abstractmethod
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning messages."""

    @abstractmethod
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error messages."""


class PdfConverterPort(ABC):
    @abstractmethod
    def pdf_to_markdown(self, data: bytes) -> str:
        """Converts the raw binary content of a PDF document into a standardized Markdown formatted string."""


class AnswerEvaluatorPort(ABC):
    HUMAN_TEMPLATE_QUESTION_KEY: Final[str] = 'question'
    HUMAN_TEMPLATE_ANSWER_KEY: Final[str] = 'answer'
    HUMAN_TEMPLATE_REFERENCE_KEY: Final[str] = 'reference'

    @abstractmethod
    def evaluate(self, benchmark_case: BenchmarkCase, answer: str) -> FidelityScore:
        """Evaluate answer quality."""
        ...


class RetrievalEvaluator:
    """Domain service that evaluates retrieval quality at a fixed cutoff k."""

    def _calculate_mrr(self, keyword: str, documents: list[ContextDocument]) -> float:
        """Compute Reciprocal Rank (RR@k) for a single keyword.

        Answers:
            "At which rank does the first relevant document for this keyword appear?"

        Interpretation:
            - Returns 1.0 if the first document is relevant
            - Returns 1/rank for the first relevant document found
            - Returns 0.0 if no relevant document appears within k

        Notes:
            - Sensitive to early ranking quality
            - Ignores all relevant documents after the first match
            - Assumes documents are already ranked by the retriever
        """
        keyword_lower = keyword.lower()

        for rank, document in enumerate(documents, start=1):
            if keyword_lower in document.content.lower():
                return 1.0 / rank

        return 0.0

    def _calculate_dcg(self, relevances: list[int]) -> float:
        """Compute Discounted Cumulative Gain (DCG).

        Answers:
            "How much relevant signal is present in this ranked list,
             while discounting lower-ranked documents?"

        Interpretation:
            - Earlier relevant documents contribute more than later ones
            - Uses logarithmic discounting based on rank

        Notes:
            - Does not normalize for list length or ideal ordering
            - Intended to be used as a building block for nDCG
        """
        dcg = 0.0
        for i, relevance in enumerate(relevances):
            dcg += relevance / math.log2(i + 2)  # rank starts at 1
        return dcg

    def _calculate_ndcg(self, keyword: str, documents: list[ContextDocument]) -> float:
        """Compute binary Normalized Discounted Cumulative Gain (nDCG@k) for a single keyword.

        Answers:
            "How well are all relevant documents for this keyword ranked
             within the top k results?"

        Interpretation:
            - 1.0 indicates optimal ordering of relevant documents
            - Values closer to 0.0 indicate poor ranking
            - Late relevance is penalized but still contributes

        Notes:
            - Uses binary relevance (relevant / not relevant)
            - Normalized against the ideal ordering of the same result set
            - Sensitive to ranking quality across the full top-k list
        """
        keyword_lower = keyword.lower()

        relevances = [1 if keyword_lower in document.content.lower() else 0 for document in documents]

        dcg = self._calculate_dcg(relevances)

        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._calculate_dcg(ideal_relevances)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_saturation_at_k(
        self,
        keyword: str,
        all_documents: list[ContextDocument],
        top_k_documents: list[ContextDocument],
    ) -> float:
        """Compute saturation@k for a single keyword.

        Answers:
            "Of all documents that are relevant for this keyword,
             how many appear within the top k results?"

        Interpretation:
            - 1.0 means all relevant documents are surfaced within top k
            - 0.0 means none of the relevant documents appear within top k

        Notes:
            - Compares retrieval saturation, not ranking quality
            - Sensitive to the choice of k
            - Assumes `all_documents` represents the full retrieval output
              (i.e. no cutoff has already been applied)
            - Returns 0.0 if no relevant documents exist
        """
        keyword_lower = keyword.lower()

        total_relevant = sum(1 for doc in all_documents if keyword_lower in doc.content.lower())

        if total_relevant == 0:
            return 0.0

        relevant_in_k = sum(1 for doc in top_k_documents if keyword_lower in doc.content.lower())

        return relevant_in_k / total_relevant

    def evaluate(
        self,
        benchmark_case: BenchmarkCase,
        documents: list[ContextDocument],
        k: int,
    ) -> RetrievalScore:
        """Evaluate retrieval performance at cutoff k.

        Metrics:
            - MRR@k:
                Measures how early the first relevant document appears.
            - nDCG@k:
                Measures overall ranking quality of relevant documents.
            - Keyword Coverage@k:
                Percentage of benchmark keywords that appear at least once
                in the top k documents.
            - Saturation@k:
                Measures how much of the total relevant material is surfaced
                within the top k results.

        Assumptions:
            - `documents` is a ranked list produced by the retriever
            - `documents` represents the full retrieval output
            - Relevance is approximated via keyword presence
        """
        if k <= 0:
            raise ValueError('k must be a positive integer')

        top_k_documents = documents[:k]

        mrr_scores = [self._calculate_mrr(keyword, top_k_documents) for keyword in benchmark_case.keywords]
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        ndcg_scores = [self._calculate_ndcg(keyword, top_k_documents) for keyword in benchmark_case.keywords]
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

        saturation_scores = [
            self._calculate_saturation_at_k(keyword, documents, top_k_documents) for keyword in benchmark_case.keywords
        ]
        avg_saturation_at_k = sum(saturation_scores) / len(saturation_scores) if saturation_scores else 0.0

        keywords_found = sum(1 for score in mrr_scores if score > 0)
        total_keywords = len(benchmark_case.keywords)
        keyword_coverage = keywords_found / total_keywords * 100 if total_keywords > 0 else 0.0

        return RetrievalScore(
            mrr=avg_mrr,
            ndcg=avg_ndcg,
            keyword_coverage=keyword_coverage,
            saturation_at_k=avg_saturation_at_k,
        )
