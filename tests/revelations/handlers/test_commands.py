from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from click import Command
from click.testing import CliRunner
from pytest_mock import MockerFixture

from rebelist.revelations.handlers.commands import data_fetcher, data_initialize, data_vectorizer, semantic_search


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

    return SimpleNamespace(
        settings=lambda: settings,
        database=lambda: mongo,
        qdrant_client=lambda: qdrant,
        data_fetch_use_case=lambda: mocker.MagicMock(),
        data_vectorize_use_case=lambda: mocker.MagicMock(),
        semantic_search_use_case=lambda: mocker.MagicMock(return_value=mocker.Mock(answer='Answer', documents=[])),
    )


class TestCLICommands:
    def test_data_initialize_runs_successfully(self, mocker: MockerFixture, fake_container: SimpleNamespace):
        """Test store:initialize with no --drop flag."""
        mocker.patch('rebelist.revelations.handlers.commands.snapshot_download')
        runner = CliRunner()
        result = runner.invoke(cast(Command, data_initialize), obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully initialized' in result.output.lower()

    def test_data_fetcher_runs_successfully(self, fake_container: SimpleNamespace):
        """Test data:fetch calls its use case and prints spaces."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, data_fetcher), obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully pulled' in result.output.lower()

    def test_data_vectorizer_runs_successfully(self, fake_container: SimpleNamespace):
        """Test data:vectorize calls its use case."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, data_vectorizer), obj=fake_container)
        assert result.exit_code == 0
        assert 'vectorized' in result.output.lower()

    @patch('rebelist.revelations.handlers.commands.prompt', return_value='exit')
    def test_semantic_search_quits_on_exit(self, fake_container: SimpleNamespace):
        """Test echo:run exits gracefully on 'exit'."""
        runner = CliRunner()
        result = runner.invoke(cast(Command, semantic_search), input='exit\n', obj=fake_container)
        assert result.exit_code == 0
        assert 'welcome to revelations' in result.output.lower()
        assert 'bye' in result.output.lower()
