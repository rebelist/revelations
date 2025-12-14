import json
from pathlib import Path
from typing import Generator

from rebelist.revelations.domain.loaders import LoaderPort
from rebelist.revelations.domain.models import BenchmarkCase


class BenchmarkLoader(LoaderPort):
    """Adapter responsible for loading BenchmarkCase instances from the file system."""

    def __init__(self, dataset: Path):
        self.__dataset = dataset

    def load(self) -> Generator[BenchmarkCase, None, None]:
        """Loads and yields BenchmarkCase objects."""
        with open(self.__dataset, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                yield BenchmarkCase(**data)
