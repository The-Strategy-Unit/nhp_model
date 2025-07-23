"""config values for docker container."""

import os

import dotenv


class Config:
    """Configuration class for Docker container."""

    __DEFAULT_CONTAINER_TIMEOUT_SECONDS = 60 * 60  # 1 hour

    def __init__(self):
        """Configuration settings for the Docker container."""
        dotenv.load_dotenv()

        self._app_version = os.environ.get("APP_VERSION", "dev")
        self._data_version = os.environ.get("DATA_VERSION", "dev")
        self._storage_account = os.environ.get("STORAGE_ACCOUNT")

        self._container_timeout_seconds = os.environ.get("CONTAINER_TIMEOUT_SECONDS")

    @property
    def APP_VERSION(self) -> str:
        """What is the version of the app?"""
        return self._app_version

    @property
    def DATA_VERSION(self) -> str:
        """What version of the data are we using?"""
        return self._data_version

    @property
    def STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account?"""
        if self._storage_account is None:
            raise ValueError("STORAGE_ACCOUNT environment variable must be set")
        return self._storage_account

    @property
    def CONTAINER_TIMEOUT_SECONDS(self) -> int:
        """How long should the container run before timing out?"""
        t = self._container_timeout_seconds
        return self.__DEFAULT_CONTAINER_TIMEOUT_SECONDS if t is None else int(t)
