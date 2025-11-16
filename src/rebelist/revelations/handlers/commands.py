from typing import Any, Mapping

import rich_click as click
from click import Context, style
from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
from rich.console import Console
from rich.markdown import Markdown


@click.command(name='store:initialize')
@click.option('--drop', is_flag=True, help='Drop databases if exists.')
@click.pass_context
def data_initialize(context: Context, drop: bool) -> None:
    """Initializes the application."""
    try:
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
            hnsw_config = HnswConfigDiff(
                # m = How many direct connections (or "shortcuts") each point gets on the map.
                m=16,
                # ef_construct = determines how thoroughly Qdrant searches for optimal connections when building the
                # HNSW index, directly influencing the final index quality and the time it takes to build.
                ef_construct=200,
            )

            # the minimum number of unindexed vectors a collection segment must accumulate before Qdrant's optimizer
            # will start the HNSW index building process.
            optimizers_config = OptimizersConfigDiff(indexing_threshold=1000)
            vector_params = VectorParams(
                size=settings.rag.embedding_dimension,
                distance=Distance.COSINE,
                hnsw_config=hnsw_config,
            )
            qdrant.create_collection(
                context_document_collection_name, vector_params, optimizers_config=optimizers_config
            )

        mongo_collection = mongo[source_document_collection_name]
        mongo_collection.create_index('id', unique=True)

        qdrant.close()

        snapshot_download(repo_id=settings.rag.ranker_model, local_dir=settings.rag.ranker_model_path)
        snapshot_download(
            repo_id=settings.rag.tokenizer_model,
            local_dir=settings.rag.tokenizer_model_path,
            allow_patterns=['*.json'],
        )

        click.secho('The application have been successfully initialized.', fg='white')
    except Exception as e:
        click.secho(f'Error initializing data: {e}', fg='red')
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='data:fetch')
@click.pass_context
def data_fetcher(context: Context) -> None:
    """Retrieves and stores documents into the port."""
    try:
        container = context.obj
        command = container.data_fetch_use_case()
        spaces = container.settings().confluence.spaces
        console = Console()

        with console.status('[bold yellow]Pulling data from the source...[/bold yellow]', spinner='dots'):
            command()

        click.secho(
            f'Documents from the spaces "{", ".join(spaces)}" have been successfully pulled from the source.',
            fg='white',
        )
    except Exception as e:
        click.secho(f'Error fetching data: {e}', fg='red')
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='data:vectorize')
@click.pass_context
def data_vectorizer(context: Context) -> None:
    """Index and structure documents for RAG context retrieval."""
    try:
        container = context.obj
        command = container.data_vectorize_use_case()
        console = Console()

        with console.status('[bold yellow]Vectorizing documents...[/bold yellow]', spinner='dots'):
            command()

        click.secho('Documents have been successfully vectorized.', fg='white')
    except Exception as e:
        click.secho(f'Error vectorizing data: {e}', fg='red')
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='chat:run')
@click.option('--evidence', is_flag=True, help='Shows evidence information from the documentation on every answer.')
@click.pass_context
def semantic_search(context: Context, evidence: bool) -> None:
    """Interactive Q&A RAG to answer questions based on documentation."""
    container = context.obj
    command = container.semantic_search_use_case()
    click.secho('Welcome to Revelations! Ask questions about the documentation or type "exit" to quit.', fg='white')
    console = Console(highlight=False)

    prompt_text = HTML('\n<ansigreen><b>👤 YOU:</b></ansigreen> ')

    while True:
        try:
            question = prompt(prompt_text).strip()

            if question.lower() == 'exit':
                break

            if not question:
                continue

            response = command(question)

            click.echo(style('\n🤖 ECHO: ', bold=True, fg='yellow'))
            console.print(Markdown(response.answer.strip(), justify='left'))

            if evidence:
                for index, document in enumerate(response.documents):
                    click.secho(
                        f'{index}Title: {document.title.strip()}\n{document.content.strip()}', fg='white', italic=True
                    )

        except KeyboardInterrupt:
            click.echo()
            break
        except Exception as e:
            click.secho(f'Error during semantic search: {e}', fg='red')

    click.secho('Bye!', fg='white')
