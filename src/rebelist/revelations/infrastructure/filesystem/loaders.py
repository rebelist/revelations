import json
from pathlib import Path
from typing import Generator

import yaml

from rebelist.revelations.domain.loaders import BenchmarkLoaderPort, PromptLoaderPort
from rebelist.revelations.domain.models import BenchmarkCase, PromptConfig


class JsonBenchmarkLoader(BenchmarkLoaderPort):
    """Adapter responsible for loading BenchmarkCase instances from the file system."""

    def __init__(self, dataset: Path):
        self.__dataset = dataset

    def load(self) -> Generator[BenchmarkCase, None, None]:
        """Loads and yields BenchmarkCase objects."""
        with open(self.__dataset, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                yield BenchmarkCase(**data)


class YamlPromptLoader(PromptLoaderPort):
    """Loads and hydrates prompt configurations from YAML files using constant injection."""

    def __init__(self, path: str, namespaces: dict[str, str]):
        self.path = path
        self.namespaces = namespaces

    def load(self, key: str) -> PromptConfig:
        """Finds a prompt by key and returns a validated PromptConfig instance."""
        with open(self.path, 'r') as prompt_file:
            raw_text = prompt_file.read()

        try:
            hydrated_text = raw_text.format(**self.namespaces)
            data = yaml.safe_load(hydrated_text)
        except (KeyError, AttributeError, yaml.YAMLError) as error:
            raise ValueError(f"Failed to load prompt '{key}': {error}") from error

        if key not in data:
            raise KeyError(f"Key '{key}' not found in {self.path}")

        return PromptConfig(**data[key])
