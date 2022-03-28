#!/usr/bin/python3

import sys
from datetime import datetime

from batch import *

if __name__ == "__main__":

    start_time = datetime.now().replace(microsecond=0)
    print("Start: {}".format(start_time))
    print()

    blob_service_client = connect_blob_storage()
    batch_client = connect_batch_client()

    err_msg = "please provide argument: prep_queue, run_queue, create_pool, start_pool, stop_pool, delete_pool"
    if len(sys.argv) == 1:
        raise Exception(err_msg)
    elif sys.argv[1] == "prep_queue":
        if len(sys.argv) != 4:
            raise Exception(
                "missing arguments, expecting: prep_queue RUNS_PER_TASK QUEUE_PATH"
            )
        prep_queue(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "run_queue":
        run_queue(blob_service_client, batch_client)
    elif sys.argv[1] == "create_pool":
        create_pool(batch_client, config._POOL_ID)
    elif sys.argv[1] == "start_pool":
        resize_pool(
            batch_client,
            config._POOL_ID,
            config._POOL_NODE_DEDICATED_COUNT,
            config._POOL_NODE_LOW_PRIORITY_COUNT,
        )
    elif sys.argv[1] == "stop_pool":
        resize_pool(batch_client, config._POOL_ID, 0, 0)
    elif sys.argv[1] == "delete_pool":
        delete_pool(batch_client, config._POOL_ID)
    else:
        raise Exception(err_msg)

    # Print out some timing info
    end_time = datetime.now().replace(microsecond=0)
    print()
    print("End: {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))
    print()
