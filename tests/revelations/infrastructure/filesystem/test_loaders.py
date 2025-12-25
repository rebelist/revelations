import json
from pathlib import Path

import yaml

from rebelist.revelations.domain import BenchmarkCase, PromptConfig
from rebelist.revelations.infrastructure.filesystem import JsonBenchmarkLoader, YamlPromptLoader


class TestJsonBenchmarkLoader:
    """Test suite for the BenchmarkLoader."""

    def test_load_yields_benchmark_cases_from_jsonl_file(self, tmp_path: Path) -> None:
        """Ensures BenchmarkLoader loads and yields BenchmarkCase instances from disk."""
        dataset = tmp_path / 'benchmark.jsonl'

        raw_cases = [
            {
                'question': 'What is polymorphism?',
                'answer': 'Polymorphism allows treating different types uniformly.',
                'keywords': ['polymorphism', 'oop'],
            },
            {
                'question': 'What is encapsulation?',
                'answer': 'Encapsulation hides internal state.',
                'keywords': ['encapsulation'],
            },
        ]

        with open(dataset, 'w', encoding='utf-8') as f:
            for case in raw_cases:
                f.write(json.dumps(case))
                f.write('\n')

        loader = JsonBenchmarkLoader(dataset)

        results = list(loader.load())

        assert len(results) == 2
        assert all(isinstance(case, BenchmarkCase) for case in results)

        assert results[0].question == raw_cases[0]['question']
        assert results[0].answer == raw_cases[0]['answer']
        assert results[0].keywords == set(raw_cases[0]['keywords'])

        assert results[1].question == raw_cases[1]['question']
        assert results[1].answer == raw_cases[1]['answer']
        assert results[1].keywords == set(raw_cases[1]['keywords'])


class TestYamlPromptLoader:
    """Test suite for the YamlPromptLoader."""

    def test_load_returns_prompt_config_with_hydrated_namespaces(self, tmp_path: Path) -> None:
        """Ensures YAML prompt is loaded, hydrated, and converted into PromptConfig."""
        prompt_file = tmp_path / 'prompts.yaml'

        prompt_yaml = {
            'qa_prompt': {
                'system_template': 'You are a helpful assistant for {domain}.',
                'human_template': 'Answer the following question about {topic}.',
            }
        }

        with open(prompt_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(prompt_yaml, f)

        namespaces = {
            'domain': 'software engineering',
            'topic': 'polymorphism',
        }

        loader = YamlPromptLoader(
            path=str(prompt_file),
            namespaces=namespaces,
        )

        prompt = loader.load('qa_prompt')

        assert isinstance(prompt, PromptConfig)
        assert prompt.system_template == 'You are a helpful assistant for software engineering.'
        assert prompt.human_template == 'Answer the following question about polymorphism.'
