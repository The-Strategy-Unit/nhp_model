#!/bin/bash

set -eo pipefail

export saname=nhpsa

if [ $# -eq 0 ]
then
  echo "Missing argument: data version"
  exit
fi

az storage blob download-batch \
  -d data \
  --pattern "$1/synthetic/*" \
  -s data \
  --account-name nhpsa \
  --auth-mode login \
  --overwrite true

rm -rf data/synthetic
mv "data/$1/synthetic" data
rmdir "data/$1"
