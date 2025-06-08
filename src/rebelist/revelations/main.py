from typing import cast

import rich_click as click
from rich_click import Command, Context

from rebelist.revelations.config.container import Container
from rebelist.revelations.handlers.commands import data_collection, data_index, data_initialize, semantic_search

container = Container.create()
settings = container.settings()


@click.group(help=settings.app.description)
@click.version_option(settings.app.version, prog_name=settings.app.name)
@click.pass_context
def console(context: Context) -> None:
    """UseCase Menu."""
    context.obj = container


data_fetcher: Command = cast(Command, data_collection)
data_vectorizer: Command = cast(Command, data_index)
data_initialize: Command = cast(Command, data_initialize)
semantic_search: Command = cast(Command, semantic_search)

console.add_command(data_initialize)
console.add_command(data_fetcher)
console.add_command(data_vectorizer)
console.add_command(semantic_search)
