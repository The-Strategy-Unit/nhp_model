from __future__ import print_function
import datetime
import io
import re
import sys
import time
import config

from azure.storage.blob import BlobServiceClient
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import azure.batch.models as batchmodels


def print_batch_exception(batch_exception):
  """
  Prints the contents of the specified Batch exception.
  :param batch_exception:
  """
  print('-------------------------------------------')
  print('Exception encountered:')
  if batch_exception.error and \
      batch_exception.error.message and \
      batch_exception.error.message.value:
    print(batch_exception.error.message.value)
    if batch_exception.error.values:
      print()
      for mesg in batch_exception.error.values:
        print('{}:\t{}'.format(mesg.key, mesg.value))
  print('-------------------------------------------')

# Pool management
def create_pool(batch_service_client, pool_id):
  vm = batchmodels.VirtualMachineConfiguration(
    image_reference = batchmodels.ImageReference(
      publisher = "microsoft-azure-batch",
      offer = "ubuntu-server-container",
      sku = "20-04-lts",
      version = "latest"
    ),
    node_agent_sku_id = "batch.node.ubuntu 20.04",
    container_configuration = batchmodels.ContainerConfiguration(
      container_image_names = [config._CR_CONTAINER_NAME],
      container_registries = [
        batchmodels.ContainerRegistry(
          user_name = config._CR_REGISTRY_NAME,
          password = config._CR_PASSWORD,
          registry_server = config._CR_LOGIN_SERVER
        )
      ]
    )
  )

  mount = lambda container: batchmodels.MountConfiguration(
    azure_blob_file_system_configuration = batchmodels.AzureBlobFileSystemConfiguration(
      account_name = config._STORAGE_ACCOUNT_NAME,
      account_key = config._STORAGE_ACCOUNT_KEY,
      container_name = container,
      relative_mount_path = container
    )
  )

  pool = batchmodels.PoolAddParameter(
    id = pool_id,
    vm_size = config._POOL_VM_SIZE,
    virtual_machine_configuration = vm,
    mount_configuration =  [mount(x) for x in ["data", "queue", "results"]],
    target_dedicated_nodes = config._POOL_NODE_DEDICATED_COUNT,
    target_low_priority_nodes = config._POOL_NODE_LOW_PRIORITY_COUNT
  )
  batch_service_client.pool.add(pool)

  print ("pool created: waiting for nodes to be allocated")
  ps = lambda: batch_service_client.pool.get(pool_id)
  while ps().allocation_state != "steady":
    time.sleep(1)
  print ("pool ready")

def resize_pool(batch_service_client, pool_id, dedicated_nodes, low_priority_nodes):
  batch_service_client.pool.resize(
    pool_id = pool_id,
    pool_resize_parameter = batchmodels.PoolResizeParameter(
      target_dedicated_nodes = dedicated_nodes,
      target_low_priority_nodes = low_priority_nodes
    )
  )
  print ("pool created: waiting for nodes to be allocated")
  ps = lambda: batch_service_client.pool.get(pool_id)
  while ps().allocation_state != "steady":
    time.sleep(1)
  print ("pool ready")

def delete_pool(batch_service_client, pool_id):
  batch_service_client.pool.delete(pool_id)
  print ("pool deleting: waiting for nodes to be deallocated")
  ps = lambda: batch_service_client.pool.get(pool_id)
  while batch_service_client.pool.exists(pool_id):
    time.sleep(1)
  print ("pool deleted")

# Job/task management
def create_job(batch_service_client, job_id, pool_id):
  """
  Creates a job with the specified ID, associated with the specified pool.
  :param batch_service_client: A Batch service client.
  :type batch_service_client: `azure.batch.BatchServiceClient`
  :param str job_id: The ID for the job.
  :param str pool_id: The ID for the pool.
  """
  print('Creating job [{}]...'.format(job_id))

  job = batchmodels.JobAddParameter(
    id = job_id,
    pool_info = batchmodels.PoolInformation(pool_id = pool_id)
  )

  batch_service_client.job.add(job)

def add_tasks(batch_service_client, job_id, input_files):
  """
  Adds a task for each input file in the collection to the specified job.
  :param batch_service_client: A Batch service client.
  :type batch_service_client: `azure.batch.BatchServiceClient`
  :param str job_id: The ID of the job to which to add the tasks.
  :param list input_files: A collection of input files. One task will be
   created for each input file.
  :param output_container_sas_token: A SAS token granting write access to
  the specified Azure Blob storage container.
  """

  print('Adding {} tasks to job [{}]...'.format(len(input_files), job_id))

  tasks = list()

  task_container_settings = batchmodels.TaskContainerSettings(
    image_name = config._CR_CONTAINER_NAME,
    container_run_options = config._CR_RUN_OPTIONS
  )

  for idx, input_file in enumerate(input_files):
    tasks.append(
      batchmodels.TaskAddParameter(
        id = 'Task{}'.format(idx),
        command_line = f"/usr/bin/Rscript /opt/run_models.R {input_file}",
        container_settings = task_container_settings,
        user_identity = batchmodels.UserIdentity(
          auto_user = batchmodels.AutoUserSpecification(
            scope = batchmodels.AutoUserScope.pool,
            elevation_level = batchmodels.ElevationLevel.admin
          )
        )
      )
    )

  batch_service_client.task.add_collection(job_id, tasks)

def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
  """
  Returns when all tasks in the specified job reach the Completed state.
  :param batch_service_client: A Batch service client.
  :type batch_service_client: `azure.batch.BatchServiceClient`
  :param str job_id: The id of the job whose tasks should be to monitored.
  :param timedelta timeout: The duration to wait for task completion. If all
  tasks in the specified job do not reach Completed state within this time
  period, an exception will be raised.
  """
  timeout_expiration = datetime.datetime.now() + timeout

  print("Monitoring all tasks for 'Completed' state, timeout in {}..."
      .format(timeout), end = '')

  while datetime.datetime.now() < timeout_expiration:
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

def print_task_output(batch_service_client, job_id, encoding = None):
  """
  Prints the stdout.txt file for each task in the job.
  :param batch_client: The batch client to use.
  :type batch_client: `batchserviceclient.BatchServiceClient`
  :param str job_id: The id of the job with task output files to print.
  """
  def _read_stream_as_string(stream, encoding):
    """
    Read stream as string
    :param stream: input stream generator
    :param str encoding: The encoding of the file. The default is utf-8.
    :return: The file content.
    :rtype: str
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

    node_id = batch_service_client.task.get(
      job_id, task.id).node_info.node_id
    print("Task: {}".format(task.id))
    print("Node: {}".format(node_id))

    stream = batch_service_client.file.get_from_task(
      job_id, task.id, config._STANDARD_OUT_FILE_NAME)

    file_text = _read_stream_as_string(
      stream,
      encoding)
    print("Standard output:")
    print(file_text)

def run_queue(blob_service_client, batch_client):
  try:
    # Create the job that will run the tasks.
    job_id = f"{config._JOB_ID}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    create_job(batch_client, job_id, config._POOL_ID)

    # Add the tasks to the job.
    bc = blob_service_client.get_container_client("queue")
    task_files = [f"/mnt/queue/{x.name}"
                  for x in bc.list_blobs()
                  if re.match("^tasks/.*\\.json$", x.name)]
    add_tasks(batch_client, job_id, task_files)

    # Pause execution until tasks reach Completed state.
    wait_for_tasks_to_complete(batch_client, job_id, datetime.timedelta(minutes = 30))

    print("  Success! All tasks reached the 'Completed' state within the "
        "specified timeout period.")

    # Print the stdout.txt and stderr.txt files for each task to the console
    print_task_output(batch_client, job_id)

  except batchmodels.BatchErrorException as err:
    print_batch_exception(err)
    raise

# Connection helpers
def connect_blob_storage():
  return BlobServiceClient(
    account_url = "https://{}.{}/".format(
      config._STORAGE_ACCOUNT_NAME,
      config._STORAGE_ACCOUNT_DOMAIN
    ),
    credential = config._STORAGE_ACCOUNT_KEY
  )

def connect_batch_client():
  credentials = SharedKeyCredentials(config._BATCH_ACCOUNT_NAME, config._BATCH_ACCOUNT_KEY)
  batch_client = BatchServiceClient(credentials, batch_url = config._BATCH_ACCOUNT_URL)
  return batch_client

# Main
if __name__ ==  '__main__':

  start_time = datetime.datetime.now().replace(microsecond = 0)
  print('Start: {}'.format(start_time))
  print()

  blob_service_client = connect_blob_storage()
  batch_client = connect_batch_client()

  err_msg = "please provide argument: run_queue, create_pool, start_pool, stop_pool, delete_pool"
  if len(sys.argv) == 1:
    print (err_msg)
  elif sys.argv[1] == "run_queue":
    run_queue(blob_service_client, batch_client)
  elif sys.argv[1] == "create_pool":
    create_pool(batch_client, config._POOL_ID)
  elif sys.argv[1] == "start_pool":
    resize_pool(batch_client, config._POOL_ID, config._POOL_NODE_DEDICATED_COUNT, config._POOL_NODE_LOW_PRIORITY_COUNT)
  elif sys.argv[1] == "stop_pool":
    resize_pool(batch_client, config._POOL_ID, 0, 0)
  elif sys.argv[1] == "delete_pool":
    delete_pool(batch_client, config._POOL_ID)
  else:
    print (err_msg)

  # Print out some timing info
  end_time = datetime.datetime.now().replace(microsecond = 0)
  print()
  print('End: {}'.format(end_time))
  print('Elapsed time: {}'.format(end_time - start_time))
  print()