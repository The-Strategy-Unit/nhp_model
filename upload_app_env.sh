#!/bin/bash

# stop executing this script if there are any errors
set -eo pipefail

# clean up code to always run, even if any step fails
function cleanup {
  rm -f nhp.tar.gz environment_prod.yml
}
trap cleanup EXIT

# create the conda environment file from the main environment.yml, stripping out anything after the "dev dependencies" comment
awk 'NR==1,/# dev dependencies/' environment.yml |
  sed -E 's/^(name: nhp)$/\1_prod/; s/\s*#.*//g' > environment_prod.yml
# update/create the conda environment from this file
conda env update -f environment_prod.yml --prune
# pack the conda environment into an archive
conda pack -n nhp_prod -o nhp.tar.gz -f
# upload the archive to blob storage
az storage blob upload --account-name $saname -c app -f nhp.tar.gz --auth-mode login