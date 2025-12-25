from abc import ABC, abstractmethod
from typing import Iterable

from rebelist.revelations.domain.models import BenchmarkCase, PromptConfig


class BenchmarkLoaderPort(ABC):
    """Interface responsible for loading BenchmarkCase instances from the file system."""

    @abstractmethod
    def load(self) -> Iterable[BenchmarkCase]:
        """Loads and yields BenchmarkCase objects."""
        ...


class PromptLoaderPort(ABC):
    """Interface for fetching prompt instructions."""

    @abstractmethod
    def load(self, key: str) -> PromptConfig:
        """Retrieves prompt templates by a unique key."""
