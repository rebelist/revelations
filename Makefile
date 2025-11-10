.PHONY: init start dev shutdown check tests coverage

init:
	@echo "\nRevelations has been initialized."
	@if [ ! -f ".env" ]; then cp ".env.example" ".env"; fi

start:
	@echo "\nStarting Revelations..."
	@docker-compose --profile prod build
	@docker-compose --profile prod up -d
	@docker-compose --profile prod exec -t ollama sh -c 'ollama pull "$$RAG_LLM_MODEL"'
	@docker-compose --profile prod exec -t ollama sh -c 'ollama pull "$$RAG_EMBEDDING_MODEL"'
	@bin/console store:initialize

dev:
	@echo "\nStarting Revelations for development..."
	@docker-compose up -d
	@brew services start ollama
	@sleep 2
	@echo "Pulling Ollama models..."
	@set -a && . ./.env && set +a && \
    ollama pull $$RAG_LLM_MODEL && \
    ollama pull $$RAG_EMBEDDING_MODEL
	@revelations store:initialize

shutdown:
	@echo "\nShutting down Revelations..."
	@docker-compose --profile prod down
	@brew services stop ollama

check:
	@echo "\nRunning pre-commit all or a specific hook..."
	@pre-commit run --all-files

tests:
	@echo "\nRunning tests..."
	@uv run pytest -vv --cache-clear --color=yes --no-header --maxfail=1 --failed-first

coverage:
	@echo "\nGenerating test coverage..."
	@uv run coverage run -m pytest --no-summary --quiet
	@uv run coverage html