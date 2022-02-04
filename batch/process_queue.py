#!/bin/env python3
from datetime import date, datetime, timedelta
import json
import os

from batch.connections import connect_adls, connect_blob_storage, connect_batch_client
from batch.jobs import create_job, add_task, wait_for_tasks_to_complete
from azure.batch import BatchServiceClient
from azure.storage.blob import BlobServiceClient

import batch.config as config

def prep_file(runs_per_task: int, path: str, file: str) -> None:
  """
  Prepares a parameters file into tasks

    * runs_per_task: the number of model runs to perform per task.
    * path: the path that contains the parameter file.
    * file: the name of the parameter file.

  This function will create the relevant folder structure in the results container, and then upload the parameter file
  to this location. It then splits the json params up into individual tasks and places them in the queue container.
  """
  # open the parameters file
  with open(os.path.join(path, file)) as f:
    params = json.load(f)
  #
  adls_client = connect_adls()
  # extract the create_time from the parameters. We need to convert a string of the form:
  #   "yyyy-mm-dd HH:MM:SS" to "yyyy-mm-dd_HH:MM:SS"
  # create_time = params["create_datetime"].replace(":", "").replace(" ", "_").replace("-", "")
  create_time = f"{datetime.now():%Y%m%d_%H%M%S}"
  # the name of the job in the batch service
  job_path = f"{params['name']}/{create_time}"
  job_id = job_path.replace("/", "_")
  # create the path where we will store the results
  results_path = f"{params['input_data']}/results/{job_path}"
  # connect to adls
  d = adls_client.get_directory_client("data", results_path)
  # create the necessary folders in the results container
  [d.get_sub_directory_client(x).create_directory() for x in ["results", "selected_variant"]]
  # upload the json to the results container
  d.get_file_client("params.json").upload_data(json.dumps(params), overwrite = True)
  #
  # get how many runs of the model to perform
  model_runs = params["model_runs"]
  # create a job
  batch_client = connect_batch_client()
  create_job(batch_client, job_id, config._POOL_ID)
  # add the tasks for this job
  add_task(batch_client, job_id, results_path, model_runs, runs_per_task)
  wait_for_tasks_to_complete(batch_client, job_id, timedelta(minutes = 30))
  batch_client.job.terminate(job_id)

def run_queue(blob_service_client: BlobServiceClient, batch_client: BatchServiceClient) -> None:
  """
  Runs all the tasks in the queue

    * blob_service_client: A Blob service client.
    * batch_service_client: A Batch service client.
  """
  try:
    # Create the job that will run the tasks.
    job_id = f"{config._JOB_ID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    create_job(batch_client, job_id, config._POOL_ID)

    # Add the tasks to the job.
    bc = blob_service_client.get_container_client("queue")
    task_files = [f"/mnt/queue/{x.name}"
                  for x in bc.list_blobs()
                  if re.match("^tasks/.*\\.json$", x.name)]
    add_tasks(batch_client, job_id, task_files)

    # Pause execution until tasks reach Completed state.
    wait_for_tasks_to_complete(batch_client, job_id, timedelta(minutes = 30))

    print("  Success! All tasks reached the 'Completed' state within the specified timeout period.")

    # Print the stdout.txt and stderr.txt files for each task to the console
    print_task_output(batch_client, job_id)

  except batchmodels.BatchErrorException as err:
    print_batch_exception(err)
    raise

def prep_queue(runs_per_task: int, queue_path: str) -> None:
  """
  Prepare the queue

    * runs_per_task: the number of model runs to perform per task.
    * path: the path that contains parameter json files.

  Prepares all of the files currently in the queue by calling prep_file on each .json file.
  """

  [prep_file(runs_per_task, queue_path, p) for p in os.listdir(queue_path) if p.endswith(".json")]