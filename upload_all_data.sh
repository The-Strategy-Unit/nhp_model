#!/bin/bash

set -eo pipefail

export saname=nhpsa

# get the dataset name from the first command line argument and loop through all of the files in
# that folder
for file in data/*/*
do
  # if $file is not a directory
  if [ ! -d "$file" ]; then
    echo "uploading: $file"
    # create the blob name: strip data/ from the start of file
    blob_name=${file#"data/"}
    # upload the file to blob storage
    az storage blob upload --account-name $saname -c data -f $file -n $blob_name --only-show-errors --overwrite
  fi
done
