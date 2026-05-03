import sys
from unittest.mock import create_autospec, patch

from langchain_text_splitters import MarkdownTextSplitter
from pymongo import MongoClient
from pymongo.synchronous.database import Database as MongoDatabase
from transformers import PreTrainedTokenizerFast

from rebelist.revelations.config.container import Container
from rebelist.revelations.config.settings import RagSettings


class TestContainer:
    def test_get_mongo_database_returns_default_database(self) -> None:
        """Verifies _get_mongo_database delegates to client.get_default_database()."""
        mock_client = create_autospec(MongoClient, instance=True)
        mock_db = create_autospec(MongoDatabase, instance=True)
        mock_client.get_default_database.return_value = mock_db

        result = Container._get_mongo_database(mock_client)  # type: ignore[reportPrivateUsage]

        assert result is mock_db
        mock_client.get_default_database.assert_called_once()

    def test_get_text_splitter_builds_markdown_splitter_from_tokenizer(self) -> None:
        """Verifies _get_text_splitter configures a MarkdownTextSplitter with the given settings."""
        settings = create_autospec(RagSettings, instance=True)
        settings.tokenizer_model_path = '/tmp/tokenizer'
        settings.chunk_size = 512
        settings.chunk_overlap = 50

        mock_tokenizer = create_autospec(PreTrainedTokenizerFast, instance=True)
        mock_tokenizer.__class__ = PreTrainedTokenizerFast  # type: ignore[reportAttributeAccessIssue]
        mock_splitter = create_autospec(MarkdownTextSplitter, instance=True)

        with patch(
            'rebelist.revelations.config.container.AutoTokenizer.from_pretrained',
            return_value=mock_tokenizer,
        ):
            with patch(
                'rebelist.revelations.config.container.MarkdownTextSplitter.from_huggingface_tokenizer',
                return_value=mock_splitter,
            ) as mock_from_tokenizer:
                result = Container._get_text_splitter(settings)  # type: ignore[reportPrivateUsage]

        assert result is mock_splitter
        assert mock_tokenizer.model_max_length == sys.maxsize
        mock_from_tokenizer.assert_called_once_with(
            tokenizer=mock_tokenizer,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
