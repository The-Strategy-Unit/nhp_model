#!/bin/env python3
import json
import os

from connections import connect_adls, connect_blob_storage

__RUNS_PER_TASK = 256
__QUEUE_PATH = os.path.join("test", "queue")


def prep_file(path, file):
  """
  Prepare the file

  Prepares a parameter file for processing. It creates the relevant folder structure in the results container, and
  uploads the json file to this folder. It then splits the json params up into individual tasks and places them in the
  queue container. 
  """
  with open(os.path.join(path, file)) as f:
    data = json.load(f)

  blob_storage_client = connect_blob_storage()
  adls_client = connect_adls()

  create_time = data["create_datetime"].replace(":", "").replace(" ", "_").replace("-", "")
  results_path = f"{data['name']}/{create_time}"

  blob_storage_client \
    .get_blob_client("results", f"{results_path}/params.json") \
    .upload_blob(json.dumps(data))

  d = adls_client.get_file_system_client("results")
  [d.create_directory(f"{results_path}/{x}") for x in ["results", "selected_strategy", "selected_variant"]]

  # split the json into tasks
  model_runs = data.pop("model_runs")
  # remove items from json not needed in the task json's
  data.pop("create_datetime")
  data.pop("name")
  data["path"] = results_path

  for i in range(1, model_runs, __RUNS_PER_TASK):
    data["run_start"] = i
    data["run_end"] = (j := i + __RUNS_PER_TASK - 1)

    blob_client = blob_storage_client.get_blob_client("queue", f"tasks/{file[0:-5]}_{i}_{j}.json")
    blob_client.upload_blob(json.dumps(data))

"""
Prepare the queue

Prepares all of the files currently in the queue by calling prep_file on each .json file.
"""
def prep_queue():
  [prep_file(__QUEUE_PATH, p) for p in os.listdir(__QUEUE_PATH) if p.endswith(".json")]

if __name__ == "__main__":
  prep_queue()