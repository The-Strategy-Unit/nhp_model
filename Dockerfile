FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Create user
RUN groupadd --gid 1000 nhp && useradd --uid 1000 --gid nhp --shell /bin/sh --home-dir /app --create-home nhp
WORKDIR /app

# Create directories with proper permissions (as root)
RUN mkdir -p data queue results && chown -R nhp:nhp /app

USER nhp

# Copy dependency files first (optimal caching)
COPY --chown=nhp:nhp pyproject.toml uv.lock ./

# Install dependencies only (skip local package)
RUN --mount=type=cache,target=/app/.cache/uv,uid=1000,gid=1000 \
  UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

# Ensure Python can find installed packages and local model
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code (changes most frequently)
COPY --chown=nhp:nhp src/nhp/ /app/src/nhp/
RUN uv pip install .

# define build arguments, these will set the environment variables in the container
ARG app_version
ARG data_version

ENV APP_VERSION=$app_version
ENV DATA_VERSION=$data_version

# Define static environment variables
ENV BATCH_SIZE=16

ENTRYPOINT ["python", "-m", "nhp.docker"]
