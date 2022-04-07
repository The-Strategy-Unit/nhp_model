"""
Process queue of model runs
"""

import json
import os
from datetime import datetime, timedelta

import batch.config as config
from batch.connections import connect_adls, connect_batch_client
from batch.jobs import add_task, create_job, wait_for_tasks_to_complete


def prep_file(runs_per_task: int, path: str, file: str) -> None:
    """
    Prepares a parameters file into tasks

      * runs_per_task: the number of model runs to perform per task.
      * path: the path that contains the parameter file.
      * file: the name of the parameter file.

    This function will create the relevant folder structure in the results container, and then
    upload the parameter file to this location. It then splits the json params up into individual
    tasks and places them in the queue container.
    """
    # open the parameters file
    with open(os.path.join(path, file), encoding="UTF-8") as params_file:
        params = json.load(params_file)
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
    adls = adls_client.get_directory_client("data", results_path)
    # upload the json to the results container
    adls.get_file_client("params.json").upload_data(json.dumps(params), overwrite=True)
    #
    # get how many runs of the model to perform
    model_runs = params["model_runs"]
    # create a job
    batch_client = connect_batch_client()
    create_job(batch_client, job_id, config.POOL_ID)
    # add the tasks for this job
    add_task(batch_client, job_id, results_path, model_runs, runs_per_task)
    wait_for_tasks_to_complete(batch_client, job_id, timedelta(minutes=30))
    batch_client.job.terminate(job_id)


def prep_queue(runs_per_task: int, queue_path: str) -> None:
    """
    Prepare the queue

      * runs_per_task: the number of model runs to perform per task.
      * path: the path that contains parameter json files.

    Prepares all of the files currently in the queue by calling prep_file on each .json file.
    """

    for i in os.listdir(queue_path):
        if i.endswith(".json"):
            prep_file(runs_per_task, queue_path, i)
