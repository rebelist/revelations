<h1 align="center"><img src="docs/images/revelations.png" alt="Revelations"/></h1>
<p align="center">
  <b>Instant Answers from Confluence Spaces</b><br>
  Lightweight prototype for querying Confluence documentation locally using natural language.
</p>

<p align="center">
   <a href="https://github.com/rebelist/revelations/releases"><img src="https://img.shields.io/badge/Release-0.9.0-e63946?logo=github&logoColor=white" alt="Release" /></a>
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

**Revelations** is an offline, open-source Retrieval-Augmented Generation (RAG) app that scans a Confluence space and lets you query your internal documentation using natural language.
Point it to any space, index the pages, and ask questions, instantly get relevant answers using your preferred language model (local or remote).

## ⚠️ Disclaimer

This project is a prototype built for learning and experimentation. It is not optimized for performance, and on a typical personal machine the latency from the language model can be high.

---
## Development Requirements
* [Python 3.13](https://www.python.org/downloads/)
* [Docker](https://docs.docker.com/desktop/)
* [Ollama](https://ollama.com/download)
* [macOS*](https://www.apple.com/macos/)

\*Required for development mode; Docker mode works on any platform

## Initialization

1. Run `make init`
2. Add the variables `CONFLUENCE_HOST`, `CONFLUENCE_TOKEN` and `CONFLUENCE_SPACE` to the **.env** file if running in development mode, or to the **.env.docker** file if the chat container is being run via Docker.
3. Run `make start` to run within docker or `make dev` for development.

## Load data from a Confluence space

1. Run `bin/console dataset:download` to fetch documents from your Confluence space
2. After it finishes, run `bin/console dataset:index` to create vector embeddings and index documents for semantic search
3. All confluence space data is now in the local storage and ready for querying.

## Search

1. Run `bin/console chat`
2. Ask a question e.g. _"Who are the members of team A?"_ or _"How session handling works in Project B?"_
3. Wait ~10 seconds for a response (depending on your computer's resources).
4. The app returns an answer based on the scanned Confluence documentation.
5. Type `exit` to quit.

You can also use the `--evidence` flag to see the source documents used to generate each answer:
```bash
bin/console chat --evidence
```

## Benchmark

Evaluate the performance of your RAG setup using a test dataset. The benchmark command measures both retrieval quality and answer fidelity.

### Usage

```bash
bin/console benchmark --dataset <path-to-dataset.jsonl> [--cutoff K] [--limit N]
```

### Options

- `--dataset` (required): Path to a JSONL file containing benchmark test cases with questions and expected answers
- `--cutoff` (default: 5): The number of top documents (K) to retrieve and use for metric calculation
- `--limit` (default: 15): Number of documents to retrieve from the database

### Metrics

The benchmark provides two categories of metrics:

**Retrieval Performance Metrics:**
- **Mean Reciprocal Rank (MRR)**: Measures how well the system ranks relevant documents at the top
- **Normalized Discounted Cumulative Gain (NDCG)**: Evaluates the quality of ranking considering the position of relevant documents
- **Keyword Coverage**: Percentage of keywords from expected answers found in retrieved documents
- **Saturation@K**: Measures how many relevant documents are found within the top K results

**Answer Quality Metrics:**
- **Accuracy**: How factually correct the generated answers are
- **Completeness**: How well the answers cover all aspects of the expected answer
- **Relevance**: How relevant the answers are to the questions asked

### Example

```bash
bin/console benchmark --dataset data/benchmark.dataset.jsonl --cutoff 5 --limit 15
```

This will run the benchmark on your test dataset and display a comprehensive report showing how well your RAG system performs.

## Optional: Shutdown

Run `make shutdown` to stop the containers and Ollama locally.
