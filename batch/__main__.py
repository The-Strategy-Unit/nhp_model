import sys
from datetime import datetime

from connections import connect_blob_storage, connect_batch_client
from process_queue import prep_queue
from pool import create_pool, resize_pool, delete_pool
from jobs import run_queue
import config

# Main
if __name__ ==  '__main__':

  start_time = datetime.now().replace(microsecond = 0)
  print('Start: {}'.format(start_time))
  print()

  blob_service_client = connect_blob_storage()
  batch_client = connect_batch_client()

  err_msg = "please provide argument: prep_queue, run_queue, create_pool, start_pool, stop_pool, delete_pool"
  if len(sys.argv) == 1:
    print (err_msg)
  elif sys.argv[1] == "prep_queue":
    prep_queue()
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
  end_time = datetime.now().replace(microsecond = 0)
  print()
  print('End: {}'.format(end_time))
  print('Elapsed time: {}'.format(end_time - start_time))
  print()