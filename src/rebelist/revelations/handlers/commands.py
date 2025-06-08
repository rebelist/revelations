from typing import Any, Mapping

import rich_click as click
from click import Context, style
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rich.console import Console


@click.command(name='store:initialize')
@click.option('--drop', is_flag=True, help='Drop databases if exists.')
@click.pass_context
def data_initialize(context: Context, drop: bool) -> None:
    """Initializes the application."""
    container = context.obj
    settings = container.settings()

    # Mongo
    mongo: Database[Mapping[str, Any]] = container.database()
    source_document_collection_name = settings.mongo.source_collection

    # Qdrant
    qdrant: QdrantClient = container.qdrant_client()
    context_document_collection_name = settings.qdrant.context_collection

    message = 'All databases will be ' + click.style('dropped', fg='bright_magenta') + ' Do you want to continue?'

    if drop and click.confirm(message):
        mongo.drop_collection(source_document_collection_name)
        qdrant.delete_collection(context_document_collection_name)

    if not qdrant.collection_exists(context_document_collection_name):
        params = VectorParams(size=settings.rag.embedding_dimension, distance=Distance.COSINE)
        qdrant.create_collection(context_document_collection_name, params)

    mongo_collection = mongo[source_document_collection_name]
    mongo_collection.create_index('id', unique=True)

    qdrant.close()

    click.secho('The application have been successfully initialized.', fg='white')
    click.secho('Bye!', fg='white')


@click.command(name='data:fetch')
@click.pass_context
def data_collection(context: Context) -> None:
    """Retrieves and stores documents into the port."""
    container = context.obj
    command = container.data_fetch_use_case()
    space = container.settings().confluence.space
    console = Console()

    with console.status('[bold yellow]Pulling data from the source...[/bold yellow]', spinner='dots'):
        command()

    click.secho(f'Documents from the space "{space}" have been successfully pulled from the source.', fg='white')
    click.secho('Bye!', fg='white')


@click.command(name='data:vectorize')
@click.pass_context
def data_index(context: Context) -> None:
    """Index and structure documents for RAG context retrieval."""
    container = context.obj
    command = container.data_vectorize_use_case()
    console = Console()

    with console.status('[bold yellow]Vectorizing documents...[/bold yellow]', spinner='dots'):
        command()

    click.secho('Documents have been successfully vectorized.', fg='white')
    click.secho('Bye!', fg='white')


@click.command(name='semantic:search')
@click.pass_context
def semantic_search(context: Context) -> None:
    """Index and structure documents for RAG context retrieval."""
    container = context.obj
    command = container.semantic_search_use_case()
    click.secho('Welcome to the Revelations! Type exit at any time to quit.', fg='white')

    while True:
        try:
            question = click.prompt(style('👤 You', bold=True, fg='green'))

            if question.strip().lower() == 'exit':
                break

            response = command(question)

            click.echo(style('🤖 Answer: ', bold=True, fg='yellow') + style(response.answer))

            for document in response.documents:
                click.secho(f'\nTitle: {document.title}\n{document.content}', fg='white', italic=True)

        except KeyboardInterrupt:
            break

    click.secho('Bye!', fg='white')
