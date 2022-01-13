from datetime import datetime, timedelta
import io
import re
import sys
import time

import batch.config as  config
from batch.batch_exception import print_batch_exception

from azure.batch import BatchServiceClient
from azure.storage.blob import BlobServiceClient
import azure.batch.models as batchmodels

def create_job(batch_service_client: BatchServiceClient, job_id: str, pool_id: str) -> None:
  """
  Creates a job with the specified ID, associated with the specified pool.

    * batch_service_client: A Batch service client.
    * job_id: The ID for the job.
    * pool_id: The ID for the pool.
  """
  print('Creating job [{}]...'.format(job_id))

  job = batchmodels.JobAddParameter(
    id = job_id,
    pool_info = batchmodels.PoolInformation(pool_id = pool_id)
  )

  batch_service_client.job.add(job)

def add_tasks(batch_service_client :BatchServiceClient, job_id: str, input_files: list) -> None:
  """
  Adds a task for each input file in the collection to the specified job.

    * batch_service_client: A Batch service client.
    * job_id: The ID of the job to which to add the tasks.
    * input_files: A collection of input files. One task will be created for each input file.
  """

  print('Adding {} tasks to job [{}]...'.format(len(input_files), job_id))

  task_container_settings = batchmodels.TaskContainerSettings(
    image_name = config._CR_CONTAINER_NAME,
    container_run_options = config._CR_RUN_OPTIONS
  )

  create_task = lambda idx, input_file: batchmodels.TaskAddParameter(
    id = 'Task{}'.format(idx), # TODO: this could/should? be the input file name itself
    command_line = f"/usr/bin/Rscript /opt/run_models.R {input_file}",
    container_settings = task_container_settings,
    user_identity = batchmodels.UserIdentity(
      auto_user = batchmodels.AutoUserSpecification(
        scope = batchmodels.AutoUserScope.pool,
        elevation_level = batchmodels.ElevationLevel.admin
      )
    )
  )

  tasks = [create_task(idx, input_file) for idx, input_file in enumerate(input_files)]
  batch_service_client.task.add_collection(job_id, tasks)

def wait_for_tasks_to_complete(batch_service_client: BatchServiceClient, job_id: str, timeout: timedelta) -> None:
  """
  Returns when all tasks in the specified job reach the Completed state.

    * batch_service_client: A Batch service client.
    * job_id: The id of the job whose tasks should be to monitored.
    * timeout: The duration to wait for task completion. If all tasks in the specified job do not reach Completed state
      within this time period, an exception will be raised.
  """
  timeout_expiration = datetime.now() + timeout

  print("Monitoring all tasks for 'Completed' state, timeout in {}..."
      .format(timeout), end = '')

  while datetime.now() < timeout_expiration:
    print('.', end = '')
    sys.stdout.flush()
    tasks = batch_service_client.task.list(job_id)

    incomplete_tasks = [task for task in tasks if task.state !=  batchmodels.TaskState.completed]
    if not incomplete_tasks:
      print()
      return True
    else:
      time.sleep(1)

  print()
  raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
             "timeout period of " + str(timeout))

def print_task_output(batch_service_client: BatchServiceClient, job_id: str, encoding: str = None) -> 1:
  """
  Prints the stdout.txt file for each task in the job.

    * batch_client: The batch client to use.
    * job_id: The id of the job with task output files to print.
  """

  def _read_stream_as_string(stream, encoding: str) -> str:
    """
    Read stream as string
    
      * stream: input stream generator
      * encoding: The encoding of the file. The default is utf-8.
    
    Returns the file content.
    """
    
    output = io.BytesIO()
    try:
      for data in stream:
        output.write(data)
      if encoding is None:
        encoding = 'utf-8'
      return output.getvalue().decode(encoding)
    finally:
      output.close()

  print('Printing task output...')

  tasks = batch_service_client.task.list(job_id)

  for task in tasks:
    node_id = batch_service_client.task.get(job_id, task.id).node_info.node_id
    print("Task: {}".format(task.id))
    print("Node: {}".format(node_id))

    stream = batch_service_client.file.get_from_task(
      job_id,
      task.id,
      config._STANDARD_OUT_FILE_NAME
    )

    file_text = _read_stream_as_string(stream, encoding)
    print("Standard output:")
    print(file_text)

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
