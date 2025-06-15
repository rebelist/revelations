.PHONY: check tests coverage start shutdown

start:
	@echo "\nStarting Revelations..."
	@if [ ! -f ".env" ]; then cp ".env.example" ".env"; fi
	@docker-compose build
	@docker-compose up -d
	@docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_LLM_MODEL"'
	@docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_EMBEDDING_MODEL"'
	@bin/console store:initialize

shutdown:
	@echo "\nShutting down Revelations..."
	@docker-compose down

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