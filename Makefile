# ── Research Agent — Makefile ─────────────────────────────────────────────────
#
# Assumes you are running inside the activated conda environment:
#   conda activate research-agent
#
# Usage:
#   make setup        create the conda env and install all dependencies
#   make update       update the conda env when environment.yml or pyproject.toml changes
#   make run          run the agent (override with: make run QUERY="CRISPR")
#   make lint         ruff check + format check
#   make typecheck    mypy static analysis
#   make test         unit tests only (no network calls)
#   make test-all     unit + integration tests (requires real API keys in .env)
#   make ci           lint + typecheck + test (mirrors the CI pipeline)
#   make clean        remove conda env and cache files

QUERY          ?= "long-term potentiation"
CONDA_ENV_NAME := research-agent
ACTIVE_ENV     := $(CONDA_DEFAULT_ENV)

define CHECK_ENV
	@if [ "$(ACTIVE_ENV)" != "$(CONDA_ENV_NAME)" ]; then \
		echo "Warning: expected conda env '$(CONDA_ENV_NAME)', got '$(ACTIVE_ENV)'"; \
		echo "   Run: conda activate $(CONDA_ENV_NAME)"; \
	fi
endef

# ── Setup ─────────────────────────────────────────────────────────────────────
# Two-step: conda creates the env with Python + system deps,
# then we call pip explicitly inside that env to install from pyproject.toml.
# This avoids conda's pip-subprocess CWD bug with editable installs.
.PHONY: setup
setup:
	conda env create -f environment.yml
	conda run -n $(CONDA_ENV_NAME) pip install -e ".[dev]"
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  conda activate $(CONDA_ENV_NAME)"
	@echo "  cp .env.example .env   # then fill in your API keys"
	@echo "  make run"

# Update after changes to environment.yml or pyproject.toml
.PHONY: update
update:
	conda env update -f environment.yml --prune
	conda run -n $(CONDA_ENV_NAME) pip install -e ".[dev]"
	@echo "Environment updated."

# ── Run ───────────────────────────────────────────────────────────────────────
.PHONY: run
run:
	$(CHECK_ENV)
	python scripts/run_agent.py --query "$(QUERY)"

# ── Lint ──────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(CHECK_ENV)
	ruff check .
	ruff format --check .

.PHONY: format
format:
	$(CHECK_ENV)
	ruff format .
	ruff check --fix .

# ── Type checking ─────────────────────────────────────────────────────────────
.PHONY: typecheck
typecheck:
	$(CHECK_ENV)
	mypy agent/ config/ schemas/ scripts/

# ── Tests ─────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(CHECK_ENV)
	pytest -m "not integration" -v

.PHONY: test-all
test-all:
	$(CHECK_ENV)
	pytest -v

# ── CI ────────────────────────────────────────────────────────────────────────
.PHONY: ci
ci: lint typecheck test

# ── Clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	conda env remove -n $(CONDA_ENV_NAME)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	@echo "Conda env '$(CONDA_ENV_NAME)' removed and caches cleared."