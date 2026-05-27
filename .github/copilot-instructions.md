# NHP Model - Copilot Coding Agent Instructions

## Repository Overview

This is the **New Hospital Programme (NHP) Demand Model**, a Python package for healthcare
activity prediction. The model provides modeling capabilities for inpatients, outpatients,
and A&E (Accident & Emergency) services. It is built as a Python library using modern
packaging tools and is deployed as both a Python package and a Docker container to Azure.

**Key Facts:**
- **Project Type:** Python package/library with Docker containerization
- **Python Version:** Requires Python >=3.12,<3.14 (from pyproject.toml). Docker
   intentionally pins 3.13 for container builds within this range.
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

# Install with dev dependencies for development/testing (RECOMMENDED for development)
uv sync

# Install with docs dependencies for documentation
uv sync --group docs
```

For production-only or Docker setup, use `uv sync --no-dev` instead of `uv sync`.

**Python Version:** CI uses `astral-sh/setup-uv` and installs dependencies with `uv sync`.


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

**Note:** Docker installs dependencies with `uv sync --frozen --no-dev --no-install-project`,
then installs the local package with `uv pip install .`.

### Testing

The commands in this section are authoritative.

**Unit Tests (ALWAYS run these before committing):**
Run these exact unit test and coverage commands:

```bash
uv run pytest -m "unit" --verbose
uv run pytest --cov=src -m "unit" --cov-branch --cov-report xml:coverage.xml
```

**Integration Tests:**

These do not run in CI due to data requirements, but can be run locally with suitable test data.
If test data is not available locally, skip integration and E2E tests entirely. A `pytest.skip`
or missing-data error from these tests is not a code defect and should not be treated as a
blocking failure.

```bash
uv run pytest -m "integration" --verbose
```

**E2E Tests:**

These do not run in CI due to data requirements, but can be run locally with suitable test data.

```bash
uv run pytest -m "e2e" --verbose
```


**All unit tests must pass. Test failures are not acceptable. We require 100% branch
coverage from unit tests; integration and E2E tests are excluded from this coverage
requirement.**

**Unit Test Failure Decision Flow:**
1. Run unit tests.
2. If all unit tests pass, proceed.
3. If a unit test fails due to a code bug, fix the code and re-run until tests pass.
4. If a unit test fails due to a stale fixture or outdated assertion, update the fixture/assertion
   only. Never delete tests or weaken assertions to bypass failures.
5. If the test validity is unclear, halt work on the branch, report the uncertainty to the user with
   the exact test name and failure message, add a TODO comment in the code, and await human
   instruction before proceeding.

This applies exclusively to unit tests (marker: unit). Integration and E2E failures caused by
missing local data are not code defects and are not blocking failures.

If a branch is genuinely untestable (for example, an OS error handler), mark it with
`# pragma: no cover` and add a comment explaining why. Overuse of pragma is a code smell;
flag any new pragma usage in the PR description for human review.

When uncertain whether a failure is a code bug or stale fixture, default to treating it as a code
bug and fix the implementation first. Only update fixtures if the code change was intentional and
the fixture reflects the old, now-invalid behavior. Document the rationale in the commit message.

### Linting and Formatting

The commands in this section are authoritative.

**ALWAYS run linting before committing. All linting checks MUST pass:**
Run these exact linting, formatting, and type-check commands:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check .
```

If a type error originates from a third-party library stub and cannot be resolved in project code,
add a `# ty: ignore[<code>]` comment with an inline explanation. Do not use bare `# ty: ignore`
without a code. Document the suppression in the PR description.

**Linting Configuration:**
- Ruff config is in `pyproject.toml` under `[tool.ruff]`
- Line length: 100 characters
- Target Python version: 3.12
- Excludes: `docs/` directory
- Key rules: pydocstyle (D), pycodestyle (E/W), isort (I), pylint (PL), pandas-vet (PD),
  numpy (NPY), ruff-specific (RUF)
- Docstring convention: Google style

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
uv run python -m nhp.model -d data/synth --type all

# Run single model type
uv run python -m nhp.model -d data/synth --type ip   # inpatients
uv run python -m nhp.model -d data/synth --type op   # outpatients
uv run python -m nhp.model -d data/synth --type aae  # A&E

# Run specific model iteration for debugging
uv run python -m nhp.model -d data/synth --model-run 1 --type ip
```

**Command-line arguments:**
- `params_file`: Path to JSON parameters file (if left empty, uses the default params file
   in src/nhp/model/params/params-sample.json)
- `-d, --data-path`: Path to data directory (default: `data`)
- `-r, --model-run`: Which model iteration to run (default: 1)
- `-t, --type`: Model type - `all`, `ip`, `op`, or `aae` (default: `all`)
- `--save-full-model-results`: Save complete model results

**Data Requirements:**
The model expects data in parquet format organized by fiscal year and dataset:
- Format: `{data_path}/{file}/fyear={year}/dataset={dataset}/`
- Required files: `ip`, `op`, `aae`, `demographic_factors`, `birth_factors`,
  `hsa_activity_tables`, `inequalities`, `ip_activity_avoidance_strategies`,
  `ip_efficiencies_strategies`, and `hsa_gams` (pickle)
- Sample data location: `data/synth/` (synthetic dataset for testing - see GitHub issue #347)

## Project Structure

### Directory Layout

**Core Directories:**
- `.github/workflows/` - CI/CD pipelines (linting, codecov, build, deploy)
- `src/nhp/model/` - Core model: `__main__.py`, `model.py`, `inpatients.py`,
  `outpatients.py`, `aae.py`, `run.py`, `results.py`, `data/`
- `src/nhp/docker/` - Docker runtime with Azure Storage integration
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests (require data)
- `tests/e2e/` - End-to-end tests (require data)
- `docs/` - MkDocs documentation

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

**IMPORTANT:** All linting and test checks must pass before merge. DO NOT skip or disable
these checks.

### Main Branch / Release Workflows

On push to main or tags, reusable and deployment workflows are composed as follows:

1. **deploy_dev.yaml** (main): Builds and pushes `:dev` container image and deploys `dev` schema
2. **deploy_release.yaml** (tags): Builds and pushes release and `latest` container images
   and deploys versioned schema
3. **build_app.yaml** (main and tags): Builds wheel artifacts, uploads to Azure Storage,
   and attaches release artifacts on tags
4. **deploy_docs.yaml** (main): Builds MkDocs and deploys documentation to RStudio Connect
5. **deploy_pr.yaml** (pull requests): Builds and pushes PR-scoped container images for
   review/testing
6. **build_container.yaml** and **build_schema.yaml**: Reusable workflows called by deploy workflows

### Docker Deployment

The model is containerized using:
- Base image: `ghcr.io/astral-sh/uv:python3.13-trixie-slim`
- Build args: `app_version`, `data_version`, `storage_account`
- Entry point: `python -m nhp.docker`
- Tags: `dev` (main), `pr-<number>` (pull requests), `v*.*.*` and `latest` (releases)
- Python policy: Docker pins 3.13 intentionally; source compatibility remains `>=3.12,<3.14`.

## Common Issues and Workarounds

**Known Issues:**
1. **Dockerfile Version**: Uses `ENV SETUPTOOLS_SCM_PRETEND_VERSION=v0.0.0` because
   setuptools-scm needs git metadata
2. **Data Structure**: Model expects parquet files at
   `{data_path}/{file}/fyear={year}/dataset={dataset}/`. Missing files cause runtime errors.

**Environment Variables (Docker):**
- `APP_VERSION`, `DATA_VERSION` (default: "dev")
- `STORAGE_ACCOUNT` (required for Azure), `BATCH_SIZE` (default: 16)
- `.env` file supported via python-dotenv for local development

## Testing Strategy

- **Unit Tests**: Marker-based (`-m "unit"`) and mock-based, parameterized. **ALWAYS run
   before committing.**
- **Integration Tests**: Marker-based (`-m "integration"`) tests requiring formatted test
   data; cover single model runs
- **E2E Tests**: Marker-based (`-m "e2e"`) tests requiring formatted test data; cover full
   model execution
- **Test Organization**: pytest-mock for mocking, fixtures in `tests/conftest.py`
- **Coverage**: 100% branch coverage from unit tests is required and enforced via Codecov
   integration

## Best Practices for Coding Agents

1. **ALWAYS install dependencies first**: Run `uv sync` before any development work.

2. **ALWAYS run pre-commit checks using the inline commands**: Use the exact commands in the
Testing and Linting sections in this file.

3. **ALWAYS run unit tests**: All unit tests must pass.

4. **Follow Google docstring convention**: All public functions/classes must have
   Google-style docstrings (enforced by ruff).

5. **Respect line length**: Maximum 100 characters per line (ruff will enforce this).

6. **Don't lint docs as source code**: The `docs/` directory is excluded from Ruff/ty
   checks by configuration.

7. **Use uv for all Python commands**: Prefix commands with `uv run` to ensure correct virtual
   environment usage.

8. **Don't modify uv.lock manually**: Use `uv sync` to update dependencies. If `uv sync` fails
   due to dependency resolution conflicts after `uv add`, run `uv lock --upgrade-package <package>`
   to attempt re-resolution, and report unresolvable conflicts rather than modifying `uv.lock`
   manually. When reporting an unresolvable dependency conflict, open a comment in the relevant PR
   or issue with the full output of `uv lock --upgrade-package <package>` and the conflicting
   version constraints identified. If `uv lock --upgrade-package <package>` also fails, revert the
   `uv add` change with `uv remove <package>`, restore `uv.lock` from version control, and open an
   issue documenting the conflicting constraints before retrying.

9. **Add dependencies using uv commands**: Run `uv add <package>` for production dependencies and
   `uv add --dev <package>` for development dependencies. When adding a dependency, verify its
   Python version constraints are compatible with >=3.12,<3.14. If a candidate package requires a
   narrower range (e.g. >=3.13), do not add it; instead report the constraint conflict to the user
   and propose an alternative package.

10. **Use consistent branch and commit naming**: Use descriptive kebab-case branch names (for
    example, `fix-hsa-scaling`) and concise imperative commit messages.

11. **Test locally before pushing**: CI checks are strict and will fail if linting/tests do not
    pass.

12. **Understand the data structure**: The model requires specific data formats. If testing model
    execution, ensure proper test data is available or use existing test fixtures.

**When in doubt, check the CI workflows in `.github/workflows/` - they define the exact
validation steps used in the pipeline.**
