import rich_click as click
from rich_click import Context


@click.group()
@click.pass_context
def console(context: Context) -> None:
    """Command group."""
    context.obj = None
