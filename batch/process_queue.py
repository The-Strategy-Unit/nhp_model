#!/bin/env python3
import json
import os

from batch.connections import connect_adls, connect_blob_storage

def prep_file(runs_per_task: int, path: str, file: str) -> None:
  """
  Prepares a parameters file into tasks

    * runs_per_task: the number of model runs to perform per task.
    * path: the path that contains the parameter file.
    * file: the name of the parameter file.

  This function will create the relevant folder structure in the results container, and then upload the parameter file
  to this location. It then splits the json params up into individual tasks and places them in the queue container.
  """
  
  with open(os.path.join(path, file)) as f:
    data = json.load(f)

  blob_storage_client = connect_blob_storage()
  adls_client = connect_adls()

  # create the location of where we are going to save in the results container
  create_time = data["create_datetime"].replace(":", "").replace(" ", "_").replace("-", "")
  results_path = f"results/{data['name']}/{create_time}"

  # upload the json to the results container
  blob_storage_client \
    .get_blob_client("data", f"{data['input_data']}/{results_path}/params.json") \
    .upload_blob(json.dumps(data), overwrite = True)

  # create the necessary folders in the results container
  d = adls_client.get_file_system_client("data")
  [d.create_directory(f"{data['input_data']}/{results_path}/{x}")
   for x in ["results", "selected_variant"]]

  # split the json into tasks
  model_runs = data.pop("model_runs")
  # remove items from json not needed in the task json's
  data.pop("create_datetime")
  data.pop("name")
  data["path"] = results_path

  # split the file up based on how many runs per task we are going to perform
  for i in range(1, model_runs, runs_per_task):
    data["run_start"] = i
    data["run_end"] = (j := i + runs_per_task - 1)

    blob_storage_client \
      .get_blob_client("queue", f"tasks/{file[0:-5]}_{i}_{j}.json") \
      .upload_blob(json.dumps(data), overwrite = True)

def prep_queue(runs_per_task: int, queue_path: str) -> None:
  """
  Prepare the queue

    * runs_per_task: the number of model runs to perform per task.
    * path: the path that contains parameter json files.

  Prepares all of the files currently in the queue by calling prep_file on each .json file.
  """

  [prep_file(runs_per_task, queue_path, p) for p in os.listdir(queue_path) if p.endswith(".json")]