#/bin/sh

if [ $# -eq 0 ]
then
  echo "Missing argument: dataset name"
  exit
fi

dataset=$1
dir="data/$dataset"

for file in "$dir"/*
do
  if [ ! -d "$file" ]; then
    blob_name=${file#"data/"}
    az storage blob upload --account-name nhpdev -c data -f $file -n $blob_name
  fi
done
