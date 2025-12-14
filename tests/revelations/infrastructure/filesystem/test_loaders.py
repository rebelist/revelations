import json
from pathlib import Path

from rebelist.revelations.domain import BenchmarkCase
from rebelist.revelations.infrastructure.filesystem import BenchmarkLoader


class TestBenchmarkLoader:
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

        loader = BenchmarkLoader(dataset)

        results = list(loader.load())

        assert len(results) == 2
        assert all(isinstance(case, BenchmarkCase) for case in results)

        assert results[0].question == raw_cases[0]['question']
        assert results[0].answer == raw_cases[0]['answer']
        assert results[0].keywords == set(raw_cases[0]['keywords'])

        assert results[1].question == raw_cases[1]['question']
        assert results[1].answer == raw_cases[1]['answer']
        assert results[1].keywords == set(raw_cases[1]['keywords'])
