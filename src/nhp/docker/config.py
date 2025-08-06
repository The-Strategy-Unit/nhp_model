"""config values for docker container."""

import os

import dotenv


class Config:
    """Configuration class for Docker container."""

    def __init__(self):
        """Configuration settings for the Docker container."""
        dotenv.load_dotenv()

        self._app_version = os.environ.get("APP_VERSION", "dev")
        self._data_version = os.environ.get("DATA_VERSION", "dev")
        self._storage_account = os.environ.get("STORAGE_ACCOUNT")

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
