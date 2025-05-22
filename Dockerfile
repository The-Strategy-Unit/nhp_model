FROM python:3.11-slim

# Create user matching micromamba's approach
RUN groupadd --gid 1000 mambauser && \
    useradd --uid 1000 --gid mambauser --shell /bin/bash --create-home mambauser

WORKDIR /opt

# Create directories with proper permissions (as root)
RUN for DIR in data queue results; do \
    mkdir -p $DIR && \
    chown mambauser:mambauser $DIR && \
    chmod a+w $DIR; \
    done

# Change ownership of working directory to mambauser
RUN chown mambauser:mambauser /opt

# Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:python3.11-alpine /uv /bin/uv


# Switch to non-root user
USER mambauser

# Copy dependency files first (optimal caching)
COPY --chown=mambauser:mambauser pyproject.toml uv.lock ./

# Install dependencies only (skip local package)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code (changes most frequently)
COPY --chown=mambauser:mambauser model /opt/model
COPY --chown=mambauser:mambauser run_model.py /opt
COPY --chown=mambauser:mambauser docker_run.py /opt
COPY --chown=mambauser:mambauser config.py /opt

# Define build arguments and environment variables
ARG app_version
ARG data_version
ARG storage_account

ENV APP_VERSION=$app_version
ENV DATA_VERSION=$data_version
ENV STORAGE_ACCOUNT=$storage_account
ENV BATCH_SIZE=16

# Ensure Python can find installed packages and local model
ENV PATH="/opt/.venv/bin:$PATH"
ENV PYTHONPATH="/opt:$PYTHONPATH"

ENTRYPOINT ["python", "./docker_run.py"]
