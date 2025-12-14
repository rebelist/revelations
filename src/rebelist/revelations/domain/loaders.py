from abc import ABC, abstractmethod
from typing import Iterable

from rebelist.revelations.domain.models import BenchmarkCase


class LoaderPort(ABC):
    """Adapter responsible for loading BenchmarkCase instances from the file system."""

    @abstractmethod
    def load(self) -> Iterable[BenchmarkCase]:
        """Loads and yields BenchmarkCase objects."""
        ...
