.PHONY: check tests coverage start shutdown

start:
	@echo "\nStarting Revelations..."
	@if [ ! -f ".env" ]; then cp ".env.example" ".env"; fi
	@docker-compose up -d
	# @docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_LLM_MODEL"'
	# @docker-compose exec -t ollama sh -c 'ollama pull "$$RAG_EMBEDDING_MODEL"'

shutdown:
	@echo "\nShutting down Revelations..."
	@docker-compose down

check:
	@echo "\nRunning pre-commit all or a specific hook..."
	@pre-commit run $(filter-out $@,$(MAKECMDGOALS))

tests:
	@echo "\nRunning tests..."
	@poetry run pytest -vv  --cache-clear --color=yes --no-header --maxfail=1 --failed-first

coverage:
	@echo "\nGenerating test coverage..."
	@poetry run coverage run -m pytest --no-summary --quiet
	@poetry run coverage html

# Avoid treating the argument as a target
%:
	@: