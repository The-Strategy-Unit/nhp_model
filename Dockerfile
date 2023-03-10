FROM mambaorg/micromamba:1.3.1-alpine
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN awk 'NR==1,/# dev dependencies/' /tmp/environment.yml | \
  sed -E 's/^(name: nhp)$/\1_prod/; s/\s*#.*//g' > /tmp/environment_prod.yml && \
  micromamba install -y -n base -f /tmp/environment_prod.yml && \
  micromamba clean --all --yes

WORKDIR /opt
COPY model /opt/model
COPY docker_run.py /opt
COPY config.py /opt

ENTRYPOINT [ "./docker_run.py" ]
