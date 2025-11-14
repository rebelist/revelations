from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_NAME: Final[str] = 'rebelist-revelations'
PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[4])


class AppSettings(BaseSettings):
    """Configuration settings for the application metadata."""

    model_config = SettingsConfigDict(frozen=True)

    name: str
    description: str
    version: str


class RagSettings(BaseSettings):
    """Configuration settings for RAG integration."""

    model_config = SettingsConfigDict(frozen=True, env_prefix='RAG_')

    embedding_model: str = ''
    embedding_dimension: str = ''
    chunk_size: int = 0
    chunk_overlap: int = 0
    llm_model: str = ''
    tokenizer_model: str = ''
    tokenizer_model_path: str = ''
    ranker_model: str = ''
    ranker_model_path: str = ''


class ConfluenceSettings(BaseSettings):
    """Configuration settings for Confluence integration."""

    model_config = SettingsConfigDict(frozen=True, env_prefix='CONFLUENCE_')

    host: str = ''
    token: str = ''
    space: str = ''


class MongoSettings(BaseSettings):
    """Configuration settings for Mongo integration."""

    model_config = SettingsConfigDict(frozen=True, env_prefix='MONGO_')

    uri: str = ''
    source_collection: str = 'source_documents'


class OllamaSettings(BaseSettings):
    """Configuration settings for Ollama integration."""

    model_config = SettingsConfigDict(frozen=True, env_prefix='OLLAMA_')

    uri: str = ''


class QdrantSettings(BaseSettings):
    """Configuration settings for Qdrant integration."""

    model_config = SettingsConfigDict(frozen=True, env_prefix='QDRANT_')

    host: str = ''
    port: str = ''
    context_collection: str = 'context_documents'


class Settings(BaseSettings):
    """Main settings class aggregating all configuration sections."""

    model_config = SettingsConfigDict(frozen=True)

    app: AppSettings
    rag: RagSettings
    confluence: ConfluenceSettings
    mongo: MongoSettings
    ollama: OllamaSettings
    qdrant: QdrantSettings


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Loads the application settings."""
    path = Path(f'{PROJECT_ROOT}/.env')

    if path.is_file():
        load_dotenv(path)

    version = metadata.version(PACKAGE_NAME)
    description = metadata.metadata(PACKAGE_NAME).get('summary', '')
    name = PACKAGE_NAME.split('-')[0].capitalize()

    return Settings(
        app=AppSettings(name=name, description=description, version=version),
        rag=RagSettings(),
        confluence=ConfluenceSettings(),
        mongo=MongoSettings(),
        ollama=OllamaSettings(),
        qdrant=QdrantSettings(),
    )
