"""
Create and run batch jobs
"""

import io
import sys
import time
from datetime import datetime, timedelta

import azure.batch.models as batchmodels
from azure.batch import BatchServiceClient

from batch import config


def create_job(
    batch_service_client: BatchServiceClient, job_id: str, pool_id: str
) -> None:
    """
    Creates a job with the specified ID, associated with the specified pool.

      * batch_service_client: A Batch service client.
      * job_id: The ID for the job.
      * pool_id: The ID for the pool.
    """
    print(f"Creating job [{job_id}]...")

    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        on_all_tasks_complete="terminateJob",
        uses_task_dependencies=True
    )

    batch_service_client.job.add(job)


def add_task(
    batch_service_client: BatchServiceClient,
    job_id: str,
    params_file: str,
    runs_per_task: int,
    params: list,
) -> None:
    """
    Adds a task for each input file in the collection to the specified job.

      * batch_service_client: A Batch service client.
      * job_id: The ID of the job to which to add the tasks.
      * params_file: The path to the params file
      * model_runs: How many runs of the model to perform
    """
    # define the user to use in batch
    user=batchmodels.UserIdentity(
        auto_user=batchmodels.AutoUserSpecification(
            scope=batchmodels.AutoUserScope.pool,
            elevation_level=batchmodels.ElevationLevel.admin,
        )
    )
    # get how many runs of the model to perform
    model_runs = params["model_runs"]
    run_results_path=f"{config.BATCH_PATH}/{job_id}"
    def create_task (run_start):
        command_line = " ".join([
            "/opt/nhp/bin/python",
            f"{config.APP_PATH}/run_model.py",
            f"{config.QUEUE_PATH}/{params_file}",
            f"--data_path={config.DATA_PATH}",
            f"--results_path={run_results_path}",
            f"--run_start={run_start}",
            f"--model_runs={runs_per_task}"
        ])
        return batchmodels.TaskAddParameter(
            id=f"run{run_start}-{run_start+runs_per_task - 1}",
            display_name=f"Model Run [{run_start} to {run_start+runs_per_task -1}]",
            command_line=command_line,
            user_identity= user
        )
    #
    tasks = [create_task(rs) for rs in range(0, model_runs, runs_per_task)]

    combine_command = " ".join([
        "/opt/nhp/bin/python",
        f"{config.APP_PATH}/combine_results.py",
        run_results_path,
        config.RESULTS_PATH,
        params["input_data"],
        params["name"],
        params["create_datetime"]
    ])
    combine_task = batchmodels.TaskAddParameter(
        id="combine",
        display_name = "Combine Results",
        command_line=combine_command,
        user_identity=user,
        depends_on=batchmodels.TaskDependencies(task_ids = [t.id for t in tasks])
    )

    batch_service_client.task.add_collection(job_id, tasks)
    batch_service_client.task.add(job_id, combine_task)


def wait_for_tasks_to_complete(
    batch_service_client: BatchServiceClient, job_id: str, timeout: timedelta
) -> None:
    """
    Returns when all tasks in the specified job reach the Completed state.

      * batch_service_client: A Batch service client.
      * job_id: The id of the job whose tasks should be to monitored.
      * timeout: The duration to wait for task completion. If all tasks in the specified job do not
        reach Completed state within this time period, an exception will be raised.
    """
    timeout_expiration = datetime.now() + timeout

    print(
        f"Monitoring all tasks for 'Completed' state, timeout in {timeout}...",
        end="",
    )

    while datetime.now() < timeout_expiration:
        print(".", end="")
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [
            task for task in tasks if task.state != batchmodels.TaskState.completed
        ]
        if not incomplete_tasks:
            print()
            return True
        time.sleep(1)

    print()
    raise RuntimeError(
        "ERROR: Tasks did not reach 'Completed' state within "
        "timeout period of " + str(timeout)
    )


def print_task_output(
    batch_service_client: BatchServiceClient, job_id: str, encoding: str = None
) -> 1:
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
                encoding = "utf-8"
            return output.getvalue().decode(encoding)
        finally:
            output.close()

    print("Printing task output...")

    tasks = batch_service_client.task.list(job_id)

    for task in tasks:
        node_id = batch_service_client.task.get(job_id, task.id).node_info.node_id
        print(f"Task: {task.id}")
        print(f"Node: {node_id}")

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, config.STANDARD_OUT_FILE_NAME
        )

        file_text = _read_stream_as_string(stream, encoding)
        print("Standard output:")
        print(file_text)
        