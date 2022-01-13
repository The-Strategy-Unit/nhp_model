from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

import config

def connect_blob_storage():
  return BlobServiceClient(
    account_url = "https://{}.{}/".format(
      config._STORAGE_ACCOUNT_NAME,
      config._STORAGE_ACCOUNT_DOMAIN
    ),
    credential = config._STORAGE_ACCOUNT_KEY
  )

def connect_adls():
  return DataLakeServiceClient(
    account_url = "https://{}.{}/".format(
      config._STORAGE_ACCOUNT_NAME,
      "dfs.core.windows.net"
    ),
    credential = config._STORAGE_ACCOUNT_KEY
  )

def connect_batch_client():
  credentials = SharedKeyCredentials(config._BATCH_ACCOUNT_NAME, config._BATCH_ACCOUNT_KEY)
  batch_client = BatchServiceClient(credentials, batch_url = config._BATCH_ACCOUNT_URL)
  return batch_client
