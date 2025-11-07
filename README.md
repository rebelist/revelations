<h1 align="center"><img src="docs/images/revelations.png" alt="Revelations"/></h1>
<p align="center">
  <b>Instant Answers from Confluence Spaces</b><br>
  Lightweight prototype for querying Confluence documentation locally using natural language.
</p>

<p align="center">
   <a href="https://github.com/rebelist/revelations/releases"><img src="https://img.shields.io/badge/Release-0.5.0--dev-e63946?logo=github&logoColor=white" alt="Release" /></a>
   <a href="https://www.gnu.org/licenses/gpl-3.0.html"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3" /></a>
   <a href="https://github.com/rebelist/revelations/actions/workflows/tests.yaml"><img src="https://github.com/rebelist/revelations/actions/workflows/tests.yaml/badge.svg" /></a>
   <a href="https://codecov.io/gh/rebelist/revelations" ><img src="https://codecov.io/gh/rebelist/revelations/graph/badge.svg?token=0FWI5KLNLH"/></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://www.mongodb.com/"><img src="https://img.shields.io/badge/Database-MongoDB-4ea94b?logo=mongodb&logoColor=white" alt="MongoDB" /></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/VectorDB-Qdrant-e6462c?logo=qdrant&logoColor=white" alt="Qdrant" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

**Revelations** is a an offline, open-source Retrieval-Augmented Generation (RAG) app that scans a Confluence space
and lets you query your internal documentation using natural language.
Point it to any space, index the pages, and ask questions, instantly get relevant answers using your preferred language model (local or remote).

## ⚠️ Disclaimer

This project is a prototype built for learning and experimentation. It is not optimized for performance, and on a typical personal machine the latency from the language model can be high.

---
## Development Requirements
* [Python 3.13](https://www.python.org/downloads/)
* [Docker](https://docs.docker.com/desktop/)
* [Ollama](https://ollama.com/download)

## Initialization

1. Run `make init`
2. Add the variables `CONFLUENCE_HOST`, `CONFLUENCE_TOKEN` and `CONFLUENCE_SPACE` to the **.env** file.
3. Run `make start`

## Load data from a Confluence space

1. Run `bin/console data:fetch`
2. After it finishes, run `bin/console data:vectorize`
3. All confluence space data is now in the local storage.

## Search

1. Run `bin/console chat:run`
2. Ask a question e.g. _"What is the problem with Pimcore?"_ or _"How session handling works in Evelin?"_
3. Wait ~35 seconds for a response (depending on your machine’s resources).
4. The app returns an answer based on the scanned Confluence documentation.
5. Type `exit` to quit.

## Optional: Shutdown

Run `make shutdown`
