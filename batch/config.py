"""
Global constant variables (Azure Storage account/Batch details)

Set the values for these in a file named config_[ENVIRONMENT].py, e.g. config_production.py.
For an example file, see config_sample.py.

The various values should be taken from the Azure portal, or using the relevant Azure CLI commands.

In order to change the environment in use, change the import statement below to select the correct
environments config file.
"""

import batch.config_production as c

BATCH_ACCOUNT_NAME = c.BATCH_ACCOUNT_NAME
BATCH_ACCOUNT_KEY = c.BATCH_ACCOUNT_KEY
BATCH_ACCOUNT_URL = c.BATCH_ACCOUNT_URL

STORAGE_ACCOUNT_NAME = c.STORAGE_ACCOUNT_NAME
STORAGE_ACCOUNT_KEY = c.STORAGE_ACCOUNT_KEY

POOL_ID = c.POOL_ID
POOL_VM_SIZE = c.POOL_VM_SIZE
POOL_NODE_DEDICATED_COUNT = c.POOL_NODE_DEDICATED_COUNT
POOL_NODE_LOW_PRIORITY_COUNT = c.POOL_NODE_LOW_PRIORITY_COUNT
POOL_VNET_SUBNET = c.POOL_VNET_SUBNET

AAD_SECRET_VALUE = c.AAD_SECRET_VALUE
AAD_SECRET_ID = c.AAD_SECRET_ID
AAD_TENANT_ID = c.AAD_TENANT_ID
AAD_APPLICATION_ID = c.AAD_APPLICATION_ID

STANDARD_OUT_FILE_NAME = "stdout.txt"

DATA_PATH = "/mnt/batch/tasks/fsmounts/data"
APP_PATH = "/mnt/batch/tasks/fsmounts/app"
QUEUE_PATH = "/mnt/batch/tasks/fsmounts/queue"
