#!/bin/bash

set -eo pipefail

export saname=nhpsa

az storage blob upload-batch --account-name $saname --source model --destination app --destination-path model
az storage blob upload --account-name $saname -c app -f run_model.py 
az storage blob upload --account-name $saname -c app -f combine_results.py 

conda pack -n nhp -o test/nhp.tar.gz
az storage blob upload --account-name $saname -c app -f test/nhp.tar.gz