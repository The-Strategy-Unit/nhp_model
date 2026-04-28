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

        self._queue_storage_account = os.environ.get("QUEUE_STORAGE_ACCOUNT")
        self._data_storage_account = os.environ.get("DATA_STORAGE_ACCOUNT")
        self._results_storage_account = os.environ.get("RESULTS_STORAGE_ACCOUNT")
        self._full_model_results_storage_account = os.environ.get(
            "FULL_MODEL_RESULTS_STORAGE_ACCOUNT"
        )
        self._model_runs_table_storage_account = os.environ.get("MODEL_RUNS_TABLE_STORAGE_ACCOUNT")

    @property
    def APP_VERSION(self) -> str:
        """What is the version of the app?"""
        return self._app_version

    @property
    def DATA_VERSION(self) -> str:
        """What version of the data are we using?"""
        return self._data_version

    @property
    def QUEUE_STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account for the queue container?"""
        if self._queue_storage_account is None:
            raise ValueError("QUEUE_STORAGE_ACCOUNT environment variable must be set")
        return self._queue_storage_account

    @property
    def DATA_STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account for the data container?"""
        if self._data_storage_account is None:
            raise ValueError("DATA_STORAGE_ACCOUNT environment variable must be set")
        return self._data_storage_account

    @property
    def RESULTS_STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account for the results container?"""
        if self._results_storage_account is None:
            raise ValueError("RESULTS_STORAGE_ACCOUNT environment variable must be set")
        return self._results_storage_account

    @property
    def FULL_MODEL_RESULTS_STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account for the full model results container?"""
        if self._full_model_results_storage_account is None:
            raise ValueError("FULL_MODEL_RESULTS_STORAGE_ACCOUNT environment variable must be set")
        return self._full_model_results_storage_account

    @property
    def MODEL_RUNS_TABLE_STORAGE_ACCOUNT(self) -> str:
        """What is the name of the storage account for the model runs table?"""
        if self._model_runs_table_storage_account is None:
            raise ValueError("MODEL_RUNS_TABLE_STORAGE_ACCOUNT environment variable must be set")
        return self._model_runs_table_storage_account
