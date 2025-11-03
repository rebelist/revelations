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
	@docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_LLM_MODEL"'
	@docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_EMBEDDING_MODEL"'
	@revelations store:initialize

shutdown:
	@echo "\nShutting down Revelations..."
	@docker-compose --profile prod down

check:
	@echo "\nRunning pre-commit all or a specific hook..."
	@pre-commit run $(filter-out $@,$(MAKECMDGOALS))

tests:
	@echo "\nRunning tests..."
	@uv run pytest -vv  --cache-clear --color=yes --no-header --maxfail=1 --failed-first

coverage:
	@echo "\nGenerating test coverage..."
	@uv run coverage run -m pytest --no-summary --quiet
	@uv run coverage html

# Avoid treating the argument as a target
%:
	@: