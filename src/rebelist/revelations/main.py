import logging
from typing import cast

import rich_click as click
from huggingface_hub.utils.tqdm import disable_progress_bars
from rich_click import Command, Context

from rebelist.revelations.config.container import Container
from rebelist.revelations.handlers.commands import (
    benchmark,
    chat,
    dataset_download,
    dataset_index,
    dataset_initialize,
)

# Disable Logging
disable_progress_bars()
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('docling').setLevel(logging.WARNING)
logging.getLogger('docling_parse').setLevel(logging.WARNING)


# Register Commands
container = Container.create()
settings = container.settings()


@click.group(help=settings.app.description)
@click.version_option(settings.app.version, prog_name=settings.app.name)
@click.pass_context
def console(context: Context) -> None:
    """UseCase Menu."""
    context.obj = container


console.add_command(cast(Command, dataset_initialize))
console.add_command(cast(Command, dataset_download))
console.add_command(cast(Command, dataset_index))
console.add_command(cast(Command, chat))
console.add_command(cast(Command, benchmark))
