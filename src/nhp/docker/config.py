"""config values for docker container."""

import os

import dotenv

dotenv.load_dotenv()

APP_VERSION = os.environ.get("APP_VERSION", "dev")
DATA_VERSION = os.environ.get("DATA_VERSION", "dev")

__DEFAULT_CONTAINER_TIMEOUT_SECONDS = 60 * 60  # 1 hour
CONTAINER_TIMEOUT_SECONDS = int(
    os.environ.get("CONTAINER_TIMEOUT_SECONDS", __DEFAULT_CONTAINER_TIMEOUT_SECONDS)
)

QUEUE_STORAGE_ACCOUNT_URL = os.environ.get("QUEUE_STORAGE_ACCOUNT_URL")
DATA_STORAGE_ACCOUNT_URL = os.environ.get("DATA_STORAGE_ACCOUNT_URL")
RESULTS_STORAGE_ACCOUNT_URL = os.environ.get("RESULTS_STORAGE_ACCOUNT_URL")
