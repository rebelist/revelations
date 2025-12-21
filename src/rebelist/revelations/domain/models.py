from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(frozen=True, slots=True)
class Document:
    id: int
    title: str
    content: str
    modified_at: datetime
    raw: str
    url: str | None

    def as_dict(self) -> dict[str, int | str | datetime]:
        """Converts the document to a dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContextDocument:
    title: str
    content: str
    modified_at: datetime
    url: str | None = None


@dataclass(frozen=True, slots=True)
class Response[T]:
    answer: T
    documents: Iterable[ContextDocument]


@dataclass(frozen=True, slots=True)
class PromptConfig:
    system_template: str
    human_template: str


class BenchmarkCase(BaseModel):
    question: str = Field(min_length=1, description='The test question.')
    answer: str = Field(min_length=1, description='The expected answer.')
    keywords: set[str] = Field(description='Associated keywords.')

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, value: set[str]) -> set[str]:
        """Validates keywords."""
        if not value:
            raise ValueError('Keywords list cannot be empty.')
        if any(not k for k in value):
            raise ValueError('Keywords cannot contain empty strings.')
        return value


class RetrievalScore(BaseModel):
    """Captures the retrieval performance metrics for the RAG system."""

    mrr: float = Field(
        description='Mean Reciprocal Rank - keyword-level MRR averaged across all keywords (not query-level MRR)'
    )
    ndcg: float = Field(
        description="""Normalized Discounted Cumulative Gain (binary relevance) - normalized against ideal ordering
                    within retrieved set"""
    )
    keyword_coverage: float = Field(description='Percentage of keywords found at least once in top k documents')
    saturation_at_k: float = Field(
        description="""Saturation@k within retrieved set - fraction of relevant documents (within retrieved set) that
        appear in top k. Relative to retrieved set, not full corpus."""
    )

    model_config = ConfigDict(frozen=True)


class FidelityScore(BaseModel):
    """Represents the LLM’s evaluation of the answer’s accuracy, completeness, and relevance."""

    accuracy: float = Field(
        description="""How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer
        must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."""
    )
    completeness: float = Field(
        description="""How complete is the answer in addressing all aspects of the question? 1
        (very poor - missing key information) to 5 (ideal - all the information from the reference answer is
        provided completely). Only answer 5 if ALL information from the reference answer is included."""
    )
    relevance: float = Field(
        description="""How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5
        (ideal - directly addresses question and gives no additional information).  Only answer 5 if the answer is
        completely relevant to the question and gives no additional information."""
    )
    feedback: str = Field(
        description="""Concise feedback on the answer quality, comparing it to the reference answer and evaluating based
        on the retrieved context. Explicitly mention what is missing or incorrect to justify the scores.""",
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkScore(BaseModel):
    retrieval: RetrievalScore = Field(description='Retrieval performance metrics')
    fidelity: FidelityScore = Field(description='Overall answer quality metrics.')

    model_config = ConfigDict(frozen=True)
