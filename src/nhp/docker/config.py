"""config values for docker container."""

import dotenv

__config_values = dotenv.dotenv_values()

APP_VERSION = __config_values.get("APP_VERSION", "dev")
DATA_VERSION = __config_values.get("DATA_VERSION", "dev")

STORAGE_ACCOUNT = __config_values.get("STORAGE_ACCOUNT", None)

__DEFAULT_CONTAINER_TIMEOUT_SECONDS = 60 * 60  # 1 hour
CONTAINER_TIMEOUT_SECONDS = int(
    __config_values.get("CONTAINER_TIMEOUT_SECONDS", __DEFAULT_CONTAINER_TIMEOUT_SECONDS)  # type: ignore
)
