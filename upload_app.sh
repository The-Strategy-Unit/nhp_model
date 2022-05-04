#!/bin/bash

set -eo pipefail

export saname=nhpsa

az storage blob upload-batch --account-name $saname --source model --destination app --destination-path model
az storage blob upload --account-name $saname -c app -f run_model.py 

conda pack -n nhp -o test/nhp.tar.gz
tar -xzf nhp.tar.gz -C test/nhp_env
az storage blob upload-batch --account-name $saname -source test/nhp_env --destination app --destination-path nhp_env
rm -rf nhp.tar.gz test/nhp_env