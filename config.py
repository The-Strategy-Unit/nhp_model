"""config values for docker container"""

import os

import dotenv

dotenv.load_dotenv()

APP_VERSION = os.environ.get("APP_VERSION", "dev")
DATA_VERSION = os.environ.get("DATA_VERSION", "dev")

STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT", None)

KEYVAULT_ENDPOINT = os.environ.get("KEYVAULT_ENDPOINT", None)

COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT", None)
COSMOS_DB = os.environ.get("COSMOS_DB", None)
