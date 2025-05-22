FROM mambaorg/micromamba:1.3.1-alpine

WORKDIR /opt
# create data, queue, results folders, make sure the user has access to write into these folders
USER root
RUN for DIR in data queue results; do \
  mkdir -p $DIR && \
  chown $MAMBA_USER:$MAMBA_USER $DIR && \
  chmod a+w $DIR; \
  done;
USER $MAMBA_USER

# copy the conda environment file across, and strip out the "dev" dependencies
# before installing the environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN awk 'NR==1,/# dev dependencies/' /tmp/environment.yml | \
  sed -E 's/^(name: nhp)$/\1_prod/; s/\s*#.*//g' > /tmp/environment_prod.yml && \
  micromamba install -y -n base -f /tmp/environment_prod.yml && \
  micromamba clean --all --yes

# copy the app code
COPY --chown=$MAMBA_USER:$MAMBA_USER model /opt/model
COPY --chown=$MAMBA_USER:$MAMBA_USER run_model.py /opt
COPY --chown=$MAMBA_USER:$MAMBA_USER docker_run.py /opt
COPY --chown=$MAMBA_USER:$MAMBA_USER config.py /opt

# define build arguments, these will set the environment variables in the container
ARG app_version
ARG data_version
ARG storage_account

ENV APP_VERSION=$app_version
ENV DATA_VERSION=$data_version
ENV STORAGE_ACCOUNT=$storage_account

ENV BATCH_SIZE=16

# set the entry point of the container to be our script
ENTRYPOINT [ "./docker_run.py" ]
