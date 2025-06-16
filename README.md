# Revelations

###  Instant Answers from a Confluence documents

**Revelations** is a lightweight, offline, open-source Retrieval-Augmented Generation (RAG) app that scans a Confluence space
and lets you query your internal documentation using natural language.
Point it to any space, index the pages, and ask questions, instantly get relevant answers using your preferred language model (local or remote).

### Disclaimer
This application is experimental and was developed for learning purposes. While it can be further tuned and optimized, the latency from the generative model is currently too high for practical use on an average personal computer.

### 1. How to initialize

1. Run `make init`
2. Add the variables `CONFLUENCE_HOST`, `CONFLUENCE_TOKEN` and `CONFLUENCE_SPACE` to the **.env** file.
3. Run `make start`

### 2. How to load data from  a confluence space

1. Run `bin/console data:fetch`
2. After it finishes, run `bin/console data:vectorize`
3. All confluence space data is now in the local storage.

### 3. How to search

1. Run `bin/console echo:run`
2. Ask a question e.g. _"What is the problem with Pimcore?"_ or _"How session handling works in Evelin?"_
3. Wait ~35 seconds for a response (depending on your machine’s resources).
4. The app returns an answer based on the scanned Confluence documentation.
5. Type `exit` to quit.

### Optional: How to shutdown

Run `make shutdown`