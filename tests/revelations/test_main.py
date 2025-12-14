from datetime import datetime
from typing import List, cast

import pytest
import rich_click as click
from click.testing import CliRunner
from rich_click import Command

from rebelist.revelations.domain import ContextDocument, Document
from rebelist.revelations.main import console

container_holder = {}


@click.command(name='inspect-context')
@click.pass_context
def inspect_context(ctx: click.Context) -> None:
    """Test command to capture context.obj for inspection."""
    container_holder['value'] = ctx.obj


@pytest.fixture
def sample_document() -> Document:
    """Create document fixture."""
    return Document(
        id=123,
        title='Test Document',
        content='This is a test document about AI and ML.',
        modified_at=datetime(2024, 2, 15, 10, 30, 0),
        raw='',
        url='https://example.com',
    )


@pytest.fixture
def sample_context_documents() -> List[ContextDocument]:
    """Create context document fixture."""
    now = datetime(2024, 2, 15, 10, 30, 0)
    return [
        ContextDocument(title='Alpha', content='Alpha content', modified_at=now),
        ContextDocument(title='Beta', content='Beta content', modified_at=now),
    ]


class TestConsoleCLI:
    """Tests for the console CLI group and its commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a runner fixture."""
        return CliRunner()

    def test_cli_shows_help(self, runner: CliRunner) -> None:
        """Should display global help and list available commands."""
        result = runner.invoke(cast(Command, console), ['--help'])
        assert result.exit_code == 0
        output = result.output.lower()
        assert 'usage' in output
        assert any(cmd in output for cmd in ['initialize', 'fetch', 'vectorize', 'semantic'])

    def test_cli_shows_version(self, runner: CliRunner) -> None:
        """Should display the application version."""
        result = runner.invoke(cast(Command, console), ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower()

    def test_subcommands_are_registered(self) -> None:
        """Should verify command names are attached to the CLI group."""
        available = set(console.commands.keys())
        assert len(available) > 0
        expected = {'benchmark', 'chat', 'dataset:download', 'dataset:index', 'dataset:initialize'}
        assert expected.issubset(available)

    def test_context_obj_is_set(self, runner: CliRunner) -> None:
        """Should ensure context.obj is available inside a command."""
        # Register temporary test command
        console.add_command(inspect_context)

        try:
            result = runner.invoke(cast(Command, console), ['inspect-context'])
            assert result.exit_code == 0
            assert 'value' in container_holder
            assert container_holder['value'] is not None
        finally:
            console.commands.pop('inspect-context', None)
