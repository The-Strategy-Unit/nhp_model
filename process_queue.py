#!/bin/env python3
import json
import os
import shutil

__RUNS_PER_TASK = 100
__RESULTS_PATH = os.path.join("test", "results")
__QUEUE_PATH = os.path.join("test", "queue")

"""
Prepare the file

Prepares a parameter file for processing. It creates the folders in __RESULTS_PATH to store the model results,
copies the parameters file to __RESULTS_PATH, then breaks the parameters into individual tasks based on
__RUNS_PER_TASK (saving these tasks into __QUEUE_PATH).
"""
def prep_file(file):
  with open(params_file := os.path.join(__QUEUE_PATH, file)) as f:
    data = json.load(f)

  # create the folders for storing this model run
  create_time = data.pop("create_datetime").replace(":", "").replace(" ", "_").replace("-", "")
  data["path"] = f"{data.pop('name')}/{create_time}"
  results_path = os.path.join(__RESULTS_PATH, data["path"])

  if not os.path.exists(tasks_path := os.path.join(__QUEUE_PATH, "tasks")):
    os.mkdir(tasks_path)

  if os.path.exists(results_path):
    shutil.rmtree(results_path) # probably should remove this in production!

  [os.makedirs(os.path.join(results_path, x)) for x in ["results", "selected_strategy", "selected_variant"]]

  # move the params to the results_path
  shutil.copy(params_file, os.path.join(results_path, file)) # change to move in production

  # split the json into tasks
  model_runs = data.pop("model_runs")

  for i in range(1, model_runs, __RUNS_PER_TASK):
    data["run_start"] = i
    data["run_end"] = (j := i + __RUNS_PER_TASK - 1)

    with open(os.path.join(tasks_path, f"{file[0:-5]}_{i}_{j}.json"), "w") as f:
      json.dump(data, f)

"""
Prepare the queue

Prepares all of the files currently in the queue by calling prep_file on each .json file.
"""
def prep_queue():
  [prep_file(p) for p in os.listdir(__QUEUE_PATH) if p.endswith(".json")]

if __name__ == "__main__":
  prep_queue()