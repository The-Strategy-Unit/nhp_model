"""config values for docker container"""

import os

import dotenv

dotenv.load_dotenv()

APP_VERSION = os.environ["APP_VERSION"]
DATA_VERSION = os.environ["DATA_VERSION"]

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT"]

KEYVAULT_ENDPOINT = os.environ["KEYVAULT_ENDPOINT"]

COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
COSMOS_DB = os.environ["COSMOS_DB"]
