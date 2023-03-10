FROM mambaorg/micromamba:1.3.1-alpine

# define build arguments, these will set the environment variables in the container
ARG app_version
ARG data_version
ARG storage_account
ARG keyvault_endpoint
ARG cosmos_endpoint
ARG cosmos_db

ENV APP_VERSION=$app_version
ENV DATA_VERSION=$data_version
ENV STORAGE_ACCOUNT=$storage_account
ENV KEYVAULT_ENDPOINT=$keyvault_endpoint
ENV COSMOS_ENDPOINT=$cosmos_endpoint
ENV COSMOS_DB=$cosmos_db

# create data / queue folders, make sure the user has access to write into these folders
USER root
RUN mkdir -p data && \
  chown $MAMBA_USER:$MAMBA_USER data && \
  chmod a+w data && \
  mkdir -p queue && \
  chown $MAMBA_USER:$MAMBA_USER queue && \
  chmod a+w data
USER $MAMBA_USER

# copy the conda environment file across, and strip out the "dev" dependencies
# before installing the environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN awk 'NR==1,/# dev dependencies/' /tmp/environment.yml | \
  sed -E 's/^(name: nhp)$/\1_prod/; s/\s*#.*//g' > /tmp/environment_prod.yml && \
  micromamba install -y -n base -f /tmp/environment_prod.yml && \
  micromamba clean --all --yes

# copy the app code
WORKDIR /opt
COPY model /opt/model
COPY docker_run.py /opt
COPY config.py /opt

# set the entry point of the container to be our script
ENTRYPOINT [ "./docker_run.py" ]
