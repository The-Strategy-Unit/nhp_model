#!/usr/bin/python3

"""
Run Model in Azure Batch

Helper tool to run the model in azure batch.

Usage:

python run_batch.py [ARG]

where [ARG] is:

  * create_pool: creates a pool in batch (configured in batch/config.py)
  * delete_pool: deletes the pool in batch
  * start_pool: starts the pool by creating the nodes (configured in batch/config.py)
  * stop_pool: stops the pool by deleting the nodes
  * run_tasks [RUNS_PER_TASK] [QUEUE_PATH]: runs the model param files in the folder [QUEUE_PATH]
      with [RUNS_PER_TASK] model iterations on each node in the pool

Generally, `create_pool` and `delete_pool` do not need to be used and are only there for setting
the batch environment up.

Before using `run_tasks`, the pool must be started with `start_pool`. Once `start_pool` is run, the
batch account will start incurring costs for the amount of nodes that are started. Once your tasks
have finished running you should run `stop_pool` as soon as possible to stop being billed for the
nodes.
"""

import sys
from datetime import datetime

from batch import (
    config,
    connect_batch_client,
    connect_blob_storage,
    prep_queue,
    resize_pool,
)

if __name__ == "__main__":

    start_time = datetime.now().replace(microsecond=0)
    print(f"Start: {start_time}")
    print()

    blob_service_client = connect_blob_storage()
    batch_client = connect_batch_client()

    ERR_MSG = "please provide argument: run_tasks, start_pool, stop_pool"
    if len(sys.argv) == 1:
        raise Exception(ERR_MSG)
    elif sys.argv[1] == "run_tasks":
        if len(sys.argv) != 4:
            raise Exception(
                "missing arguments, expecting: prep_queue RUNS_PER_TASK QUEUE_PATH"
            )
        prep_queue(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "start_pool":
        resize_pool(
            batch_client,
            config.POOL_ID,
            config.POOL_NODE_DEDICATED_COUNT,
            config.POOL_NODE_LOW_PRIORITY_COUNT,
        )
    elif sys.argv[1] == "stop_pool":
        resize_pool(batch_client, config.POOL_ID, 0, 0)
    else:
        raise Exception(ERR_MSG)

    # Print out some timing info
    end_time = datetime.now().replace(microsecond=0)
    print()
    print(f"End: {end_time}")
    print(f"Elapsed time: {end_time - start_time}")
    print()
