from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from click import Command
from click.testing import CliRunner
from pytest_mock import MockerFixture

from rebelist.revelations.domain.models import BenchmarkScore, FidelityScore, RetrievalScore
from rebelist.revelations.handlers.commands import (
    benchmark,
    chat,
    dataset_download,
    dataset_index,
    dataset_initialize,
)


@pytest.fixture
def fake_container(mocker: MockerFixture):
    """Creates a fake dependency injection container for CLI tests."""
    settings = mocker.Mock()
    settings.mongo.source_collection = 'source_docs'
    settings.qdrant.context_collection = 'context_docs'
    settings.rag.embedding_dimension = 768
    settings.rag.ranker_model_path = '/tmp/model'
    settings.rag.ranker_model_name = 'mock/ranker'
    settings.confluence.spaces = 'DOCS'

    mongo = mocker.MagicMock()
    mongo.__getitem__.return_value = mocker.MagicMock()

    qdrant = mocker.MagicMock()
    qdrant.collection_exists.return_value = False

    retrieval = RetrievalScore(ndcg=0.1, mrr=0.23, keyword_coverage=20, saturation_at_k=0.4)
    fidelity = FidelityScore(accuracy=1, feedback='Nothing.', completeness=0.2, relevance=0.3)

    return SimpleNamespace(
        settings=lambda: settings,
        database=lambda: mongo,
        qdrant_client=lambda: qdrant,
        data_extraction_use_case=lambda: mocker.MagicMock(),
        data_embedding_use_case=lambda: mocker.MagicMock(),
        inference_use_case=lambda: mocker.MagicMock(return_value=mocker.Mock(answer='Answer', documents=[])),
        benchmark_use_case=lambda: mocker.MagicMock(
            return_value=BenchmarkScore(retrieval=retrieval, fidelity=fidelity)
        ),
    )


class TestCLICommands:
    def test_dataset_initialize_runs_successfully(self, mocker: MockerFixture, fake_container: SimpleNamespace):
        """Test dataset:initialize with no --drop flag."""
        mocker.patch('rebelist.revelations.handlers.commands.snapshot_download')
        runner = CliRunner()
        result = runner.invoke(cast(Command, dataset_initialize), obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully initialized' in result.output.lower()

    def test_dataset_download_runs_successfully(self, fake_container: SimpleNamespace):
        """Test dataset:download calls its use case and prints spaces."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, dataset_download), obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully pulled' in result.output.lower()

    def test_dataset_index_runs_successfully(self, fake_container: SimpleNamespace):
        """Test dataset:index calls its use case."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, dataset_index), obj=fake_container)
        assert result.exit_code == 0
        assert 'Documents have been successfully saved to qdrant' in result.output

    @patch('rebelist.revelations.handlers.commands.prompt', return_value='exit')
    def test_chat_quits_on_exit(self, fake_container: SimpleNamespace):
        """Test chat exits gracefully on 'exit'."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, chat), input='exit\n', obj=fake_container)
        assert result.exit_code == 0
        assert 'welcome to revelations' in result.output.lower()
        assert 'bye' in result.output.lower()

    def test_benchmark_runs_successfully(self, fake_container: SimpleNamespace):
        """Test benchmark calls its use case."""
        runner = CliRunner()
        result = runner.invoke(
            cast(Command, benchmark), ['--dataset', 'tests/data/benchmark.mini.dataset.jsonl'], obj=fake_container
        )

        assert result.exit_code == 0
        assert 'Mean Reciprocal Rank                   │  0.23' in result.output
        assert 'Normalized Discounted Cumulative Gain  │   0.1' in result.output
        assert 'Keyword Coverage                       │  20.0' in result.output
        assert 'Saturation@K                           │   0.4' in result.output
        assert 'Accuracy                       │           1.0' in result.output
        assert 'Completeness                   │           0.2' in result.output
        assert 'Relevance                      │           0.3' in result.output
