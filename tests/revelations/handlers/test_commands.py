from types import SimpleNamespace
from typing import cast
from unittest.mock import create_autospec, patch

import pytest
from click import Command
from click.testing import CliRunner
from pymongo.synchronous.collection import Collection as MongoCollection
from pymongo.synchronous.database import Database as MongoDatabase
from qdrant_client import QdrantClient
from rich.live import Live

from rebelist.revelations.application.use_cases.benchmark import BenchmarkUseCase
from rebelist.revelations.application.use_cases.embedding import DataEmbeddingUseCase
from rebelist.revelations.application.use_cases.extraction import DataExtractionUseCase
from rebelist.revelations.application.use_cases.inference import InferenceUseCase
from rebelist.revelations.config.settings import (
    ConfluenceSettings,
    MongoSettings,
    QdrantSettings,
    RagSettings,
    Settings,
)
from rebelist.revelations.domain import ContextDocument, Response
from rebelist.revelations.domain.models import BenchmarkScore, FidelityScore, RetrievalScore
from rebelist.revelations.handlers.commands import (
    benchmark,
    chat,
    dataset_download,
    dataset_index,
    dataset_initialize,
)


@pytest.fixture
def fake_container() -> SimpleNamespace:
    """Creates a fake dependency injection container for CLI tests."""
    settings = create_autospec(Settings, instance=True)
    settings.mongo = create_autospec(MongoSettings, instance=True)
    settings.mongo.source_collection = 'source_docs'
    settings.qdrant = create_autospec(QdrantSettings, instance=True)
    settings.qdrant.context_collection = 'context_docs'
    settings.qdrant.vector_name = 'dense'
    settings.qdrant.sparse_vector_name = 'sparse'
    settings.rag = create_autospec(RagSettings, instance=True)
    settings.rag.embedding_dimension = 768
    settings.rag.ranker_model = ''
    settings.rag.ranker_model_path = ''
    settings.rag.tokenizer_model = ''
    settings.rag.tokenizer_model_path = ''
    settings.confluence = create_autospec(ConfluenceSettings, instance=True)
    settings.confluence.spaces = 'DOCS'

    mongo = create_autospec(MongoDatabase, instance=True)
    mongo.__getitem__.return_value = create_autospec(MongoCollection, instance=True)

    qdrant = create_autospec(QdrantClient, instance=True)
    qdrant.collection_exists.return_value = False

    retrieval = RetrievalScore(ndcg=0.1, mrr=0.23, keyword_coverage=20, saturation_at_k=0.4)
    fidelity = FidelityScore(accuracy=1, feedback='Nothing.', completeness=0.2, relevance=0.3)

    mock_extraction = create_autospec(DataExtractionUseCase, instance=True)
    mock_embedding = create_autospec(DataEmbeddingUseCase, instance=True)

    mock_response = create_autospec(Response, instance=True)
    mock_response.answer = 'Answer'
    mock_response.documents = []
    mock_inference = create_autospec(InferenceUseCase, instance=True)
    mock_inference.return_value = mock_response

    mock_benchmark = create_autospec(BenchmarkUseCase, instance=True)
    mock_benchmark.return_value = BenchmarkScore(retrieval=retrieval, fidelity=fidelity)

    return SimpleNamespace(
        settings=lambda: settings,
        database=lambda: mongo,
        qdrant_client=lambda: qdrant,
        data_extraction_use_case=lambda: mock_extraction,
        data_embedding_use_case=lambda: mock_embedding,
        inference_use_case=lambda: mock_inference,
        benchmark_use_case=lambda: mock_benchmark,
    )


class TestCLICommands:
    def test_dataset_initialize_runs_successfully(self, fake_container: SimpleNamespace):
        """Test dataset:initialize with no --drop flag."""
        with patch('rebelist.revelations.handlers.commands.snapshot_download'):
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

    def test_dataset_initialize_drops_databases_when_confirmed(self, fake_container: SimpleNamespace):
        """Test that --drop causes collections to be cleared when the user confirms."""
        with patch('rebelist.revelations.handlers.commands.snapshot_download'):
            runner = CliRunner()
            result = runner.invoke(cast(Command, dataset_initialize), ['--drop'], input='y\n', obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully initialized' in result.output.lower()
        fake_container.database().drop_collection.assert_called_once_with('source_docs')
        fake_container.qdrant_client().delete_collection.assert_called_once_with('context_docs')

    def test_dataset_initialize_skips_collection_creation_when_already_exists(self, fake_container: SimpleNamespace):
        """Test that collection creation is skipped when the collection already exists."""
        fake_container.qdrant_client().collection_exists.return_value = True
        with patch('rebelist.revelations.handlers.commands.snapshot_download'):
            runner = CliRunner()
            result = runner.invoke(cast(Command, dataset_initialize), obj=fake_container)
        assert result.exit_code == 0
        assert 'successfully initialized' in result.output.lower()
        fake_container.qdrant_client().create_collection.assert_not_called()

    def test_dataset_initialize_shows_error_on_failure(self, fake_container: SimpleNamespace):
        """Test that initialization errors are reported and the command exits cleanly."""
        with patch(
            'rebelist.revelations.handlers.commands.snapshot_download',
            side_effect=Exception('Download failed'),
        ):
            runner = CliRunner()
            result = runner.invoke(cast(Command, dataset_initialize), obj=fake_container)
        assert result.exit_code == 0
        assert 'Error initializing data' in result.output
        assert 'bye' in result.output.lower()

    def test_dataset_download_shows_error_on_failure(self, fake_container: SimpleNamespace):
        """Test that dataset:download reports errors and exits cleanly."""
        failing_use_case = create_autospec(DataExtractionUseCase, instance=True)
        failing_use_case.side_effect = Exception('fetch failed')
        fake_container.data_extraction_use_case = lambda: failing_use_case
        runner = CliRunner()
        result = runner.invoke(cast(Command, dataset_download), obj=fake_container)
        assert result.exit_code == 0
        assert 'Error fetching data' in result.output
        assert 'bye' in result.output.lower()

    def test_dataset_index_reraises_and_shows_error_on_failure(self, fake_container: SimpleNamespace):
        """Test that dataset:index logs the error, prints bye, then re-raises."""
        failing_use_case = create_autospec(DataEmbeddingUseCase, instance=True)
        failing_use_case.side_effect = Exception('embed failed')
        fake_container.data_embedding_use_case = lambda: failing_use_case
        runner = CliRunner()
        result = runner.invoke(cast(Command, dataset_index), obj=fake_container)
        assert result.exit_code != 0
        assert 'Error saving data to qdrant' in result.output
        assert 'bye' in result.output.lower()

    def test_chat_processes_question_and_streams_answer(self, fake_container: SimpleNamespace):
        """Test that a real question is processed and the streamed answer is displayed."""
        mock_live = create_autospec(Live, instance=True)
        mock_live.__enter__.return_value = mock_live
        mock_live.__exit__.return_value = False
        with patch('rebelist.revelations.handlers.commands.prompt', side_effect=['Hello there', 'exit']):
            with patch('rebelist.revelations.handlers.commands.Live', return_value=mock_live):
                runner = CliRunner()
                result = runner.invoke(cast(Command, chat), obj=fake_container)
        assert result.exit_code == 0
        assert 'bye' in result.output.lower()

    def test_chat_shows_evidence_when_flag_is_set(self, fake_container: SimpleNamespace):
        """Test that document evidence is printed when --evidence is passed."""
        mock_live = create_autospec(Live, instance=True)
        mock_live.__enter__.return_value = mock_live
        mock_live.__exit__.return_value = False

        doc = create_autospec(ContextDocument, instance=True)
        doc.title.strip.return_value = 'Evidence Title'
        doc.content.strip.return_value = 'Evidence content text'

        mock_response = create_autospec(Response, instance=True)
        mock_response.answer = 'Answer text'
        mock_response.documents = [doc]
        mock_inference = create_autospec(InferenceUseCase, instance=True)
        mock_inference.return_value = mock_response
        fake_container.inference_use_case = lambda: mock_inference

        with patch('rebelist.revelations.handlers.commands.prompt', side_effect=['Explain this', 'exit']):
            with patch('rebelist.revelations.handlers.commands.Live', return_value=mock_live):
                runner = CliRunner()
                result = runner.invoke(cast(Command, chat), ['--evidence'], obj=fake_container)
        assert result.exit_code == 0
        assert 'Evidence Title' in result.output

    def test_chat_skips_empty_question_and_exits(self, fake_container: SimpleNamespace):
        """Test that an empty question is skipped and the loop continues."""
        with patch('rebelist.revelations.handlers.commands.prompt', side_effect=['', 'exit']):
            runner = CliRunner()
            result = runner.invoke(cast(Command, chat), obj=fake_container)
        assert result.exit_code == 0
        assert 'bye' in result.output.lower()

    def test_chat_exits_gracefully_on_keyboard_interrupt(self, fake_container: SimpleNamespace):
        """Test that KeyboardInterrupt during prompt exits the chat loop cleanly."""
        with patch('rebelist.revelations.handlers.commands.prompt', side_effect=KeyboardInterrupt()):
            runner = CliRunner()
            result = runner.invoke(cast(Command, chat), obj=fake_container)
        assert result.exit_code == 0
        assert 'bye' in result.output.lower()

    def test_chat_shows_error_on_search_failure(self, fake_container: SimpleNamespace):
        """Test that exceptions during inference are caught and reported."""
        failing_use_case = create_autospec(InferenceUseCase, instance=True)
        failing_use_case.side_effect = Exception('search crashed')
        fake_container.inference_use_case = lambda: failing_use_case
        with patch('rebelist.revelations.handlers.commands.prompt', return_value='Failing question'):
            runner = CliRunner()
            result = runner.invoke(cast(Command, chat), obj=fake_container)
        assert result.exit_code == 0
        assert 'Error during semantic search' in result.output

    def test_benchmark_shows_error_on_failure(self, fake_container: SimpleNamespace):
        """Test that benchmark errors are reported and the command exits cleanly."""
        failing_use_case = create_autospec(BenchmarkUseCase, instance=True)
        failing_use_case.side_effect = Exception('benchmark crashed')
        fake_container.benchmark_use_case = lambda: failing_use_case
        runner = CliRunner()
        result = runner.invoke(
            cast(Command, benchmark), ['--dataset', 'tests/data/benchmark.mini.dataset.jsonl'], obj=fake_container
        )
        assert result.exit_code == 0
        assert 'Error running benchmark' in result.output
