FROM ghcr.io/astral-sh/uv:python3.11-alpine


# Create user
RUN addgroup -g 1000 nhp && adduser -u 1000 -G nhp -s /bin/sh -h /app -D nhp
# temporary fix, should change the api to run ./docker_run.py rather than /opt/docker_run.py
RUN rmdir /opt && ln -s /app /opt
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

# TODO: in order to build the docker container, we need to force the version number
# might be worth building the .whl and copying that into the container instead
ENV SETUPTOOLS_SCM_PRETEND_VERSION=v0.0.0

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

ENTRYPOINT ["python", "-m", "nhp.docker"]
