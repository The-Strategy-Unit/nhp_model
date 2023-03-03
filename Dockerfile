FROM mambaorg/micromamba:1.3.1
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN awk 'NR==1,/# dev dependencies/' environment.yml | \
  sed -E 's/^(name: nhp)$/\1_prod/; s/\s*#.*//g' > environment_prod.yml && \
  micromamba install -y -n base -f /tmp/environment_prod.yml && \
  micromamba clean --all --yes

USER root
RUN mkdir -p /app

WORKDIR /app
COPY model /app/model
COPY docker_run.py /app
