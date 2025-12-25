<h1 align="center"><img src="docs/images/revelations.png" alt="Revelations"/></h1>
<p align="center">
  <b>Instant Answers from Confluence Spaces</b><br>
  Lightweight prototype for querying Confluence documentation locally using natural language.
</p>

<p align="center">
   <a href="https://github.com/rebelist/revelations/releases"><img src="https://img.shields.io/badge/Release-0.11.0-e63946?logo=github&logoColor=white" alt="Release" /></a>
   <a href="https://www.gnu.org/licenses/gpl-3.0.html"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3" /></a>
   <a href="https://github.com/rebelist/revelations/actions/workflows/tests.yaml"><img src="https://github.com/rebelist/revelations/actions/workflows/tests.yaml/badge.svg" alt="Tests" /></a>
   <a href="https://codecov.io/gh/rebelist/revelations" ><img src="https://codecov.io/gh/rebelist/revelations/graph/badge.svg?token=0FWI5KLNLH" alt="Codecov"/></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://www.mongodb.com/"><img src="https://img.shields.io/badge/Database-MongoDB-4ea94b?logo=mongodb&logoColor=white" alt="MongoDB" /></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/VectorDB-Qdrant-e6462c?logo=qdrant&logoColor=white" alt="Qdrant" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

**Revelations** is an offline, open-source Retrieval-Augmented Generation (RAG) application that transforms your
Confluence documentation into an intelligent, queryable knowledge base. Simply point it to any Confluence space, index
the pages, and ask questions in natural language to get instant, contextually relevant answers using your preferred
language model (local or remote).

## ✨ Features

- **Hybrid Search**: Combines semantic (dense) and keyword-based (sparse) vector search for optimal retrieval accuracy
- **RAG Pipeline**: Complete retrieval-augmented generation workflow with document chunking, embedding, and
  context-aware generation
- **Cross-Encoder Reranking**: Advanced document reranking using transformer models to improve relevance
- **Benchmark Suite**: Comprehensive evaluation framework with retrieval and answer quality metrics
- **Clean Architecture**: Well-structured codebase following domain-driven design principles
- **Dependency Injection**: Modular design with dependency injection for testability and maintainability
- **Offline-First**: Works completely offline with local models and databases
- **Docker Support**: Containerized deployment for easy setup and portability
- **Type Safety**: Strict type checking with Pyright for robust code quality

## 🏗️ Architecture

This project follows **Clean Architecture** principles with clear separation of concerns:

- **Domain Layer**: Core business logic, models, and repository interfaces (ports)
- **Application Layer**: Use cases orchestrating business workflows
- **Infrastructure Layer**: External adapters (Confluence API, Qdrant, MongoDB, Ollama, etc.)
- **Handlers Layer**: CLI commands and user interface

The architecture uses the **Ports and Adapters** pattern, making the system highly testable and allowing easy swapping
of
infrastructure components. Dependency injection is handled via `dependency-injector`, enabling clean separation and
mockability for testing.

## 🔍 Hybrid Search

Revelations implements **hybrid search** via Qdrant, combining two complementary search strategies:

- **Dense Vector Search**: Uses semantic embeddings to find documents conceptually similar to your query, even without
  exact keyword matches. Enables understanding of meaning and context.
- **Sparse Vector Search**: Uses BM25-based keyword matching to find documents containing specific terms from your
  query. Ensures precise keyword matches are not missed.

By combining both approaches, hybrid search provides semantic understanding for natural language queries and precise
keyword matching for technical terms. Results are further refined using a cross-encoder reranker model to improve
relevance ranking.

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.13**: Modern Python with latest features
- **LangChain**: RAG framework and LLM orchestration
- **Qdrant**: Vector database with hybrid search support
- **MongoDB**: Document storage for source materials
- **Ollama**: Local LLM inference server

### Key Libraries

- **dependency-injector**: Dependency injection container
- **sentence-transformers**: Embedding and reranking models
- **pydantic**: Data validation and settings management
- **loguru**: Structured logging
- **pytest**: Testing framework with comprehensive coverage

### Development Tools

- **Pyright**: Strict type checking
- **Ruff**: Fast Python linter and formatter
- **pytest-cov**: Code coverage analysis
- **pre-commit**: Git hooks for code quality

## 📁 Project Structure

```
rebelist-revelations/
├── src/rebelist/revelations/
│   ├── domain/          # Business logic, models, and ports
│   ├── application/     # Use cases (business workflows)
│   ├── infrastructure/  # External adapters (Confluence, Qdrant, MongoDB, etc.)
│   ├── handlers/        # CLI commands and user interface
│   └── config/          # Configuration and dependency injection
├── tests/               # Comprehensive test suite
└── docker/              # Docker configurations
```

## ⚠️ Disclaimer

This project is a prototype built for learning and experimentation. It is not optimized for performance, and on a
typical personal machine the latency from the language model can be high.

---

## 🚀 Quick Start

### Prerequisites

* [Python 3.13](https://www.python.org/downloads/)
* [Docker](https://docs.docker.com/desktop/)
* [Ollama](https://ollama.com/download)
* [macOS*](https://www.apple.com/macos/) - Required for development mode; Docker mode works on any platform

### Installation

1. Clone the repository and initialize the project:
   ```bash
   make init
   ```

2. Configure environment variables:
    - Add `CONFLUENCE_HOST`, `CONFLUENCE_TOKEN`, and `CONFLUENCE_SPACE` to `.env` (development) or `.env.docker` (
      Docker)

3. Start the application:
   ```bash
   make start    # Docker mode
   # or
   make dev      # Development mode
   ```

### Usage

1. **Download documents from Confluence**:
   ```bash
   bin/console dataset:download
   ```

2. **Index documents for search**:
   ```bash
   bin/console dataset:index
   ```

3. **Query your documentation**:
   ```bash
   bin/console chat
   ```

   Ask questions like:
    - _"Who are the members of team A?"_
    - _"How does session handling work in Project B?"_

4. **View source evidence** (optional):
   ```bash
   bin/console chat --evidence
   ```

## 📊 Benchmarking

Evaluate your RAG system's performance using the built-in benchmark suite. The benchmark measures both retrieval
quality and answer fidelity across multiple metrics.

### Usage

```bash
bin/console benchmark --dataset <path-to-dataset.jsonl> [--cutoff K] [--limit N]
```

### Options

- `--dataset` (required): Path to a JSONL file containing benchmark test cases with questions and expected answers
- `--cutoff` (default: 5): Number of top documents (K) to retrieve and use for metric calculation
- `--limit` (default: 15): Total number of documents to retrieve from the database

### Metrics

The benchmark provides comprehensive evaluation across two categories:

#### Retrieval Performance Metrics

- **Mean Reciprocal Rank (MRR)**: Measures how well the system ranks relevant documents at the top
- **Normalized Discounted Cumulative Gain (NDCG)**: Evaluates ranking quality considering document position
- **Keyword Coverage**: Percentage of keywords from expected answers found in retrieved documents
- **Saturation@K**: Measures how many relevant documents are found within the top K results

#### Answer Quality Metrics

- **Accuracy**: Factual correctness of generated answers
- **Completeness**: How well answers cover all aspects of the expected answer
- **Relevance**: Relevance of answers to the questions asked

### Example

```bash
bin/console benchmark --dataset data/benchmark.dataset.jsonl --cutoff 5 --limit 15
```

This generates a comprehensive report showing retrieval and generation performance metrics.

## 🛑 Shutdown

Stop all containers and services:

```bash
make shutdown
```

## 🤝 Contributing

This is a learning and experimentation project. Contributions, suggestions, and feedback are welcome! Please ensure
that:

- Code follows the existing architecture patterns
- Tests are added for new features
- Type hints are included for all functions
- Code passes linting and type checking (`ruff` and `pyright`)
