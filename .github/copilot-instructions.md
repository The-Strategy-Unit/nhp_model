# NHP Model - Copilot Coding Agent Instructions

## Repository Overview

This is the **New Hospital Programme (NHP) Demand Model**, a Python package for healthcare activity prediction. The model provides modeling capabilities for inpatients, outpatients, and A&E (Accident & Emergency) services. It is built as a Python library using modern packaging tools and is deployed as both a Python package and a Docker container to Azure.

**Key Facts:**
- **Project Type:** Python package/library with Docker containerization
- **Python Version:** Requires Python 3.11 or higher (specified in pyproject.toml)
- **Package Manager:** `uv` (modern Python package manager from Astral)
- **Build System:** setuptools with setuptools-scm for versioning
- **Primary Language:** Python
- **Project Size:** Medium-sized Python project
- **Main Modules:** nhp.model (core model code), nhp.docker (Docker runtime)

## Environment Setup and Build Instructions

### Initial Setup

**ALWAYS start by installing uv and project dependencies:**

```bash
# Install uv using the recommended approach from Astral
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies (production only)
uv sync

# Install with dev dependencies for development/testing (RECOMMENDED for development)
uv sync --extra dev

# Install with docs dependencies for documentation
uv sync --extra docs

# Install multiple extras at once
uv sync --extra dev --extra docs
```

**Important:** The `uv sync` command only installs production dependencies. For development work (linting, testing), use `uv sync --extra dev` to install the dev dependencies.

**Python Version:** The project requires Python 3.11+. The CI uses Python 3.11 specifically via `uv python install` in workflows.

### Build Commands

**To build the package:**

```bash
# Standard build - creates wheel and source distribution
uv build

# Build for development (sets version to 0.dev0)
SETUPTOOLS_SCM_PRETEND_VERSION=0.dev0 uv build
```

The build creates:
- `dist/nhp_model-<version>-py3-none-any.whl`
- `dist/nhp_model-<version>.tar.gz`

**Note:** The Dockerfile includes a TODO comment about forcing version numbers during Docker builds. Currently it uses `ENV SETUPTOOLS_SCM_PRETEND_VERSION=v0.0.0` as a workaround.

### Testing

**Unit Tests (ALWAYS run these before committing):**

```bash
# Run all unit tests
uv run pytest tests/unit --verbose

# Run unit tests with coverage report
uv run pytest --cov=. tests/unit --ignore=tests --cov-branch --cov-report xml:coverage.xml
```

**Integration Tests:**

```bash
# Integration tests require test data in a specific format
# These are located in tests/integration/ but may require data setup
uv run pytest tests/integration --verbose
```

**All unit tests must pass. Test failures are NOT acceptable.**

### Linting and Formatting

**ALWAYS run linting before committing. All linting checks MUST pass:**

```bash
# Run ruff linting check
uvx ruff check .

# Run ruff format check (no auto-formatting)
uvx ruff format --check .

# Auto-format code (if needed)
uvx ruff format .

# Run type checking with ty
uvx ty check .
```

**Linting Configuration:**
- Ruff config is in `pyproject.toml` under `[tool.ruff]`
- Line length: 100 characters
- Target Python version: 3.11
- Excludes: `notebooks/` directory
- Key rules: pydocstyle (D), pycodestyle (E/W), isort (I), pylint (PL), pandas-vet (PD), numpy (NPY), ruff-specific (RUF)
- Docstring convention: Google style

**The notebooks directory is excluded from linting and should not be linted.**

### Documentation

```bash
# Build documentation (requires docs dependencies)
uv run mkdocs build --clean

# Serve documentation locally
uv run mkdocs serve
```

Documentation is deployed automatically to Connect via CI on main branch pushes.

### Running the Model

**Local execution:**

```bash
# Run with sample parameters (requires data in specified path)
uv run python -m nhp.model queue/params-sample.json -d data/synth --type all

# Run single model type
uv run python -m nhp.model queue/params-sample.json -d data --type ip   # inpatients
uv run python -m nhp.model queue/params-sample.json -d data --type op   # outpatients
uv run python -m nhp.model queue/params-sample.json -d data --type aae  # A&E

# Run specific model iteration for debugging
uv run python -m nhp.model queue/params-sample.json -d data --model-run 1 --type ip
```

**Command-line arguments:**
- `params_file`: Path to JSON parameters file (default: `queue/params-sample.json`)
- `-d, --data-path`: Path to data directory (default: `data`)
- `-r, --model-run`: Which model iteration to run (default: 1)
- `-t, --type`: Model type - `all`, `ip`, `op`, or `aae` (default: `all`)
- `--save-full-model-results`: Save complete model results

**Data Requirements:**
The model expects data in parquet format organized by fiscal year and dataset:
- Format: `{data_path}/{file}/fyear={year}/dataset={dataset}/`
- Required files: `ip`, `op`, `aae`, `demographic_factors`, `birth_factors`, `hsa_activity_tables`, `hsa_gams` (pickle)
- Sample data location: `data/synth/` (synthetic dataset for testing - see GitHub issue #347)

## Project Structure

### Directory Layout

**Core Directories:**
- `.github/workflows/` - CI/CD pipelines (linting, codecov, build, deploy)
- `src/nhp/model/` - Core model: `__main__.py`, `model.py`, `inpatients.py`, `outpatients.py`, `aae.py`, `run.py`, `results.py`, `data/`
- `src/nhp/docker/` - Docker runtime with Azure Storage integration
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests (require data)
- `docs/` - MkDocs documentation
- `notebooks/` - Databricks notebooks (excluded from linting)
- `queue/` - Parameter files (params-sample.json)

**Key Configuration Files:**
- `pyproject.toml` - Project metadata, dependencies, ruff/pytest/setuptools config
- `uv.lock` - Locked dependency versions (DO NOT modify manually)
- `params-schema.json` - JSON schema for model parameters (deployed to GitHub Pages)

### Architecture Overview

**Model Hierarchy:**
- `Model` (base class in model.py) - Common model functionality
  - `InpatientsModel` - Inpatient demand modeling
  - `OutpatientsModel` - Outpatient demand modeling  
  - `AaEModel` - A&E demand modeling

**Execution Flow:**
1. `__main__.py` parses CLI arguments and loads parameters
2. `run.py` orchestrates model execution (single or parallel runs)
3. `ModelIteration` runs a single model iteration
4. Results are aggregated and saved by `results.py`

**Data Loading:**
- Abstract `Data` interface allows multiple data sources
- `Local` loads from local parquet files
- `DatabricksNational` loads from Databricks (used in notebooks)

## CI/CD Validation Pipeline

### Pull Request Checks

**Every pull request triggers these workflows (ALL MUST PASS):**

1. **Linting** (`.github/workflows/linting.yaml`):
   - `ruff check` - Code quality checks
   - `ruff format --check` - Code formatting verification
   - `ty check .` - Type checking

2. **Code Coverage** (`.github/workflows/codecov.yaml`):
   - Runs unit tests with coverage
   - Uploads to Codecov
   - Requires passing tests

**IMPORTANT:** All linting and test checks must pass before merge. DO NOT skip or disable these checks.

### Main Branch / Release Workflows

On push to main or tags:

1. **build_app.yaml**: Builds Python wheel, uploads to Azure Storage and GitHub releases
2. **build_schema.yaml**: Deploys params-schema.json to GitHub Pages
3. **build_container.yaml**: Builds and pushes Docker image to GitHub Container Registry
4. **deploy_docs.yaml**: Builds and deploys MkDocs documentation to RStudio Connect

### Docker Deployment

The model is containerized using:
- Base image: `ghcr.io/astral-sh/uv:python3.11-alpine`
- Build args: `app_version`, `data_version`, `storage_account`
- Entry point: `python -m nhp.docker`
- Tags: `dev` (PRs), `v*.*.*` (releases), `latest` (latest release)

## Common Issues and Workarounds

**Known Issues:**
1. **Dockerfile Version**: Uses `ENV SETUPTOOLS_SCM_PRETEND_VERSION=v0.0.0` because setuptools-scm needs git metadata (TODO: build wheel and copy instead)
2. **Data Structure**: Model expects parquet files at `{data_path}/{file}/fyear={year}/dataset={dataset}/`. Missing files cause runtime errors.
3. **Notebooks**: `notebooks/` directory excluded from linting - don't lint these Databricks notebooks.

**Environment Variables (Docker):**
- `APP_VERSION`, `DATA_VERSION` (default: "dev")
- `STORAGE_ACCOUNT` (required for Azure), `BATCH_SIZE` (default: 16)
- `.env` file supported via python-dotenv for local development

## Testing Strategy

- **Unit Tests**: `tests/unit/` - Mock-based, parameterized. **ALWAYS run before committing.**
- **Integration Tests**: `tests/integration/` - Require properly formatted test data, test end-to-end runs
- **Test Organization**: pytest-mock for mocking, fixtures in `tests/conftest.py`
- **Coverage**: High coverage maintained via Codecov integration

## Best Practices for Coding Agents

1. **ALWAYS install dependencies first**: Run `uv sync --extra dev` before any development work.

2. **ALWAYS run linting before committing**: Run `uvx ruff check .` and `uvx ruff format --check .` - these MUST pass.

3. **ALWAYS run unit tests**: Run `uv run pytest tests/unit` before committing - all tests MUST pass.

4. **Follow Google docstring convention**: All public functions/classes must have Google-style docstrings (enforced by ruff).

5. **Respect line length**: Maximum 100 characters per line (ruff will enforce this).

6. **Don't modify notebooks**: The `notebooks/` directory is excluded from linting for a reason. These are Databricks notebooks with special formatting.

7. **Use uv for all Python commands**: Prefix commands with `uv run` to ensure correct virtual environment usage.

8. **Don't modify uv.lock manually**: Use `uv sync` to update dependencies.

9. **Test locally before pushing**: The CI checks are strict and will fail if linting/tests don't pass.

10. **Understand the data structure**: The model requires specific data formats. If testing model execution, ensure proper test data is available or use existing test fixtures.

## Quick Reference

```bash
# Setup (production + dev dependencies)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# Lint (MUST pass)
uvx ruff check .
uvx ruff format --check .

# Test (MUST pass)  
uv run pytest tests/unit --verbose

# Build
uv build

# Run model (requires data)
uv run python -m nhp.model queue/params-sample.json -d data --type all

# Build docs (requires docs extras)
uv sync --extra docs
uv run mkdocs build --clean
```

**When in doubt, check the CI workflows in `.github/workflows/` - they define the exact validation steps used in the pipeline.**
