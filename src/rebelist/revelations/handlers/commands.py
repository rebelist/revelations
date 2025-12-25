from pathlib import Path
from typing import Any, Mapping, cast

import rich_click as click
from click import Context, style
from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pymongo.synchronous.database import Database
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff, SparseIndexParams, SparseVectorParams
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

from rebelist.revelations.domain import BenchmarkScore
from rebelist.revelations.handlers.console import Number
from rebelist.revelations.infrastructure.filesystem import JsonBenchmarkLoader


@click.command(name='dataset:initialize')
@click.option('--drop', is_flag=True, help='Drop databases if exists.')
@click.pass_context
def dataset_initialize(context: Context, drop: bool) -> None:
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
                m=32,
                # ef_construct = determines how thoroughly Qdrant searches for optimal connections when building the
                # HNSW index, directly influencing the final index quality and the time it takes to build.
                ef_construct=300,
            )

            # the minimum number of unindexed vectors a collection segment must accumulate before Qdrant's optimizer
            # will start the HNSW index building process.
            optimizers_config = OptimizersConfigDiff(indexing_threshold=200)

            vector_params = VectorParams(
                size=settings.rag.embedding_dimension,
                distance=Distance.COSINE,
                hnsw_config=hnsw_config,
            )

            sparse_params = SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=True,
                )
            )

            qdrant.create_collection(
                collection_name=context_document_collection_name,
                vectors_config={settings.qdrant.vector_name: vector_params},
                sparse_vectors_config={settings.qdrant.sparse_vector_name: sparse_params},
                optimizers_config=optimizers_config,
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
    except Exception as error:
        click.secho(f'Error initializing data: {error}', fg='red')
        return
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='dataset:download')
@click.pass_context
def dataset_download(context: Context) -> None:
    """Retrieves and stores documents into the database."""
    try:
        container = context.obj
        data_extraction_use_case = container.data_extraction_use_case()
        spaces = container.settings().confluence.spaces
        console = Console()

        with console.status('[bold yellow]Downloading data from the source...[/bold yellow]', spinner='dots'):
            data_extraction_use_case()

        click.secho(
            f'Documents from the spaces "{", ".join(spaces)}" have been successfully pulled from the source.',
            fg='white',
        )
    except Exception as error:
        click.secho(f'Error fetching data: {error}', fg='red')
        return
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='dataset:index')
@click.pass_context
def dataset_index(context: Context) -> None:
    """Index and structure documents for RAG context retrieval."""
    try:
        container = context.obj
        data_embedding_use_case = container.data_embedding_use_case()
        console = Console()

        with console.status('[bold yellow]Saving documents to Qdrant...[/bold yellow]', spinner='dots'):
            data_embedding_use_case()

        click.secho('Documents have been successfully saved to qdrant.', fg='white')
    except Exception as error:
        click.secho(f'Error saving data to qdrant: {error}', fg='red')
        raise
    finally:
        click.secho('Bye!', fg='white')


@click.command(name='chat')
@click.option('--evidence', is_flag=True, help='Shows evidence information from the documentation on every answer.')
@click.pass_context
def chat(context: Context, evidence: bool) -> None:
    """Interactive Q&A RAG to answer questions based on documentation."""
    container = context.obj
    inference_use_case = container.inference_use_case()
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

            response = inference_use_case(question)
            answer_buffer = '\n' + style('🤖 ECHO: ', bold=True, fg='yellow')

            with Live(console=console, screen=False, refresh_per_second=10) as live:
                for chunk in response.answer:
                    answer_buffer += chunk
                    live.update(Markdown(answer_buffer.strip()))

            if evidence:
                for index, document in enumerate(response.documents):
                    click.secho(
                        f'{index}Title: {document.title.strip()}\n{document.content.strip()}', fg='white', italic=True
                    )

        except KeyboardInterrupt:
            click.echo()
            break
        except Exception as error:
            click.secho(f'Error during semantic search: {error}', fg='red')
            return

    click.secho('Bye!', fg='white')


@click.command(name='benchmark')
@click.option(
    '--dataset',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help='Path to the dataset used for retrieval and evaluation.',
)
@click.option(
    '--cutoff',
    default=5,
    type=int,
    show_default=True,
    help='The number of top documents (K) to retrieve and use for metric calculation.',
)
@click.option(
    '--limit',
    default=15,
    type=int,
    show_default=True,
    help='Number of documents to retrieve from the database.',
)
@click.pass_context
def benchmark(context: Context, dataset: Path, cutoff: int, limit: int) -> None:
    """Benchmarks the full retrieval flow to measure how well the current RAG setup performs."""
    console = Console()
    container = context.obj

    try:
        with console.status('[bold yellow]Running benchmark...[/bold yellow]', spinner='dots'):
            benchmark_use_case = container.benchmark_use_case()
            loader = JsonBenchmarkLoader(dataset)
            benchmark_cases = list(loader.load())
            benchmark_score = cast(BenchmarkScore, benchmark_use_case(benchmark_cases, cutoff, limit))
    except Exception as error:
        click.secho(f'Error running benchmark: {error}', fg='red')
        return

    retrieval = benchmark_score.retrieval
    fidelity = benchmark_score.fidelity

    table_restrieval = Table(title='\nRetrieval performance metrics', width=50)
    table_restrieval.add_column('Metric', justify='left', style='grey70', no_wrap=True)
    table_restrieval.add_column('Score', justify='right')
    table_restrieval.add_row('Mean Reciprocal Rank', Number.prettify(retrieval.mrr, Number.Scale.ZERO_ONE))
    table_restrieval.add_row(
        'Normalized Discounted Cumulative Gain', Number.prettify(retrieval.ndcg, Number.Scale.ZERO_ONE)
    )
    table_restrieval.add_row('Keyword Coverage', Number.prettify(retrieval.keyword_coverage, Number.Scale.PERCENT))
    table_restrieval.add_row('Saturation@K', Number.prettify(retrieval.saturation_at_k, Number.Scale.ZERO_ONE))

    table_fidelity = Table(title='\nAnswer quality metrics', width=50)
    table_fidelity.add_column('Metric', justify='left', style='grey70', no_wrap=True)
    table_fidelity.add_column('Score', justify='right')
    table_fidelity.add_row('Accuracy', Number.prettify(fidelity.accuracy, Number.Scale.ONE_FIVE))
    table_fidelity.add_row('Completeness', Number.prettify(fidelity.completeness, Number.Scale.ONE_FIVE))
    table_fidelity.add_row('Relevance', Number.prettify(fidelity.relevance, Number.Scale.ONE_FIVE))

    console.print(table_restrieval)
    console.print(table_fidelity)
