#!/bin/bash

set -eo pipefail

export saname=nhpsa

az storage blob upload-batch --account-name $saname --source model --destination app --destination-path model --auth-mode login
az storage blob upload --account-name $saname -c app -f run_model.py --auth-mode login

conda pack -n nhp -o test/nhp.tar.gz -f
az storage blob upload --account-name $saname -c app -f test/nhp.tar.gz --auth-mode login