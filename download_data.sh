#!/bin/bash

set -eo pipefail

export saname=nhpsa

if [ $# != 3 ]
then
  echo "Missing arguments: data_version, dataset, year"
  exit
fi

data_version=$1
dataset=$2
year=$3
data_path="$data_version/$year/$dataset"

mkdir -p "$data_version/$year"

az storage blob download-batch \
  -d data \
  --pattern "$data_path/*" \
  -s data \
  --account-name nhpsa \
  --auth-mode login \
  --overwrite true

rm -rf "data/$year/$dataset"
mv "data/$data_version/$year/$dataset" "data/$year/$dataset"
rm -rf "data/$data_version"
