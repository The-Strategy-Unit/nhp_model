FROM ghcr.io/astral-sh/uv:python3.11-alpine

# Create user
RUN addgroup -g 1000 nhp && adduser -u 1000 -G nhp -s /bin/sh -h /app -D nhp
WORKDIR /app
USER nhp

# Create directories with proper permissions (as root)
RUN for DIR in data queue results; do mkdir -p $DIR; done

# Copy dependency files first (optimal caching)
COPY --chown=nhp:nhp pyproject.toml uv.lock ./

# Install dependencies only (skip local package)
RUN uv sync --frozen --no-dev --no-install-project

# Ensure Python can find installed packages and local model
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code (changes most frequently)
COPY --chown=nhp:nhp src/nhp/ /app/src/nhp/
RUN uv pip install .

# define build arguments, these will set the environment variables in the container
ARG app_version
ARG data_version
ARG storage_account

ENV APP_VERSION=$app_version
ENV DATA_VERSION=$data_version
ENV STORAGE_ACCOUNT=$storage_account

# Define static environment variables
ENV BATCH_SIZE=16

# temporary patch until we update the api
USER root
RUN printf '#!/bin/sh\n/app/.venv/bin/python -m nhp.docker "$@"\n' > /opt/docker_run.py && \
  chmod +x /opt/docker_run.py
USER nhp

ENTRYPOINT ["python", "-m", "nhp.docker"]
