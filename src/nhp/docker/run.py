"""Run the model inside of the docker container."""

import gzip
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from nhp.docker import config
from nhp.model.helpers import load_params


class RunWithLocalStorage:
    """Methods for running with local storage."""

    def __init__(self, filename: str):
        """Initialize the RunWithLocalStorage instance.

        :param filename: Name of the parameter file to load.
        :type filename: str
        """
        self.params = load_params(f"queue/{filename}")

    def finish(self, results_file: str, saved_files: list, save_full_model_results: bool) -> None:
        """Post model run steps.

        :param results_file: the path to the results file
        :type results_file: str
        :param saved_files: filepaths of results, saved in parquet format and params
        in json format
        :type saved_files: list
        :param save_full_model_results: whether to save the full model results or not
        :type save_full_model_results: bool
        """

    def progress_callback(self) -> Callable[[Any], Callable[[Any], None]]:
        """Progress callback method.

        for local storage do nothing
        """
        return lambda _: lambda _: None


class RunWithAzureStorage:
    """Methods for running with azure storage."""

    def __init__(self, filename: str, app_version: str = "dev"):
        """Initialise RunWithAzureStorage.

        :param filename:
        :type filename: str
        :param app_version: the version of the app, where we will load data from. defaults to "dev"
        :type app_version: str, optional
        """
        logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
            logging.WARNING
        )

        self._app_version = re.sub("(\\d+\\.\\d+)\\..*", "\\1", app_version)

        self.params = self._get_params(filename)
        self._get_data(self.params["start_year"], self.params["dataset"])

        if not config.QUEUE_STORAGE_ACCOUNT_URL:
            raise ValueError("QUEUE_STORAGE_ACCOUNT_URL environment variable not set")
        self._queue_storage_account_url = config.QUEUE_STORAGE_ACCOUNT_URL

        if not config.DATA_STORAGE_ACCOUNT_URL:
            raise ValueError("DATA_STORAGE_ACCOUNT_URL environment variable not set")
        self._data_storage_account_url = config.DATA_STORAGE_ACCOUNT_URL

        if not config.RESULTS_STORAGE_ACCOUNT_URL:
            raise ValueError("RESULTS_STORAGE_ACCOUNT_URL environment variable not set")
        self._results_storage_account_url = config.RESULTS_STORAGE_ACCOUNT_URL

    def _get_container(self, account_url: str, container_name: str):
        credential = DefaultAzureCredential()
        bsc = BlobServiceClient(account_url=account_url, credential=credential)
        return bsc.get_container_client(container_name)

    def _get_params(self, filename: str) -> dict:
        """Get the parameters for the model.

        :param filename: the name of the params file
        :type filename: str
        :return: the parameters for the model
        :rtype: dict
        """
        logging.info("downloading params: %s", filename)

        self._queue_blob = self._get_container(
            self._queue_storage_account_url, "queue"
        ).get_blob_client(filename)

        params_content = self._queue_blob.download_blob().readall()

        return json.loads(params_content)

    def _get_data(self, year: str, dataset: str) -> None:
        """Get data to run the model.

        for local storage, the data is already available, so do nothing.

        :param year: the year of data to load
        :type year: str
        :param year: the year of data to load
        :type year: str
        """
        logging.info("downloading data (%s / %s)", year, dataset)

        container = self._get_container(self._data_storage_account_url, "data")

        version = config.DATA_VERSION

        paths = container.list_blobs(name_starts_with=f"{version}/fyear={year}/dataset={dataset}")

        for p in paths:
            filename = p.name
            logging.info(" * %s", filename)
            local_name = "data" + filename.removeprefix(version)

            os.makedirs(os.path.dirname(local_name), exist_ok=True)

            with open(local_name, "wb") as local_file:
                file_client = container.download_blob(filename)
                local_file.write(file_client.readall())

    def _upload_results_json(self, results_file: str, metadata: dict) -> None:
        """Upload the results.

        once the model has run, upload the results to blob storage

        :param results_file: the saved results file
        :type results_file: str
        :param metadata: the metadata to attach to the blob
        :type metadata: dict
        """
        container = self._get_container(self._results_storage_account_url, "results")

        with open(f"results/{results_file}.json", "rb") as file:
            container.upload_blob(
                f"prod/{self._app_version}/{results_file}.json.gz",
                gzip.compress(file.read()),
                metadata=metadata,
                overwrite=True,
            )

    def _upload_results_files(self, files: list, metadata: dict) -> None:
        """Upload the results.

        once the model has run, upload the files (parquet for model results and json for
        model params) to blob storage

        :param files: list of files to be uploaded
        :type files: list
        :param metadata: the metadata to attach to the blob
        :type metadata: dict

        """
        container = self._get_container(self._results_storage_account_url, "results")
        for file in files:
            filename = file[8:]
            if file.endswith(".json"):
                metadata_to_use = metadata
            else:
                metadata_to_use = None
            with open(file, "rb") as f:
                container.upload_blob(
                    f"aggregated-model-results/{self._app_version}/{filename}",
                    f.read(),
                    overwrite=True,
                    metadata=metadata_to_use,
                )

    def _upload_full_model_results(self) -> None:
        container = self._get_container(self._results_storage_account_url, "results")

        dataset = self.params["dataset"]
        scenario = self.params["scenario"]
        create_datetime = self.params["create_datetime"]

        path = Path(f"results/{dataset}/{scenario}/{create_datetime}")

        for file in path.glob("**/*.parquet"):
            filename = file.as_posix()[8:]
            with open(file, "rb") as f:
                container.upload_blob(
                    f"full-model-results/{self._app_version}/{filename}",
                    f.read(),
                    overwrite=True,
                )

    def _cleanup(self) -> None:
        """Cleanup.

        once the model has run, remove the file from the queue
        """
        logging.info("cleaning up queue")

        self._queue_blob.delete_blob()

    def finish(self, results_file: str, saved_files: list, save_full_model_results: bool) -> None:
        """Post model run steps.

        :param results_file: the path to the results file
        :type results_file: str
        :param saved_files: filepaths of results, saved in parquet format and params
        in json format
        :type saved_files: list
        :param save_full_model_results: whether to save the full model results or not
        :type save_full_model_results: bool
        """
        metadata = {
            k: str(v)
            for k, v in self.params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }
        self._upload_results_json(results_file, metadata)
        self._upload_results_files(saved_files, metadata)
        if save_full_model_results:
            self._upload_full_model_results()
        self._cleanup()

    def progress_callback(self) -> Callable[[Any], Callable[[Any], None]]:
        """Progress callback method.

        updates the metadata for the blob in the queue to give progress
        """
        blob = self._queue_blob

        current_progress = {
            **blob.get_blob_properties()["metadata"],
            "Inpatients": 0,
            "Outpatients": 0,
            "AaE": 0,
        }

        blob.set_blob_metadata({k: str(v) for k, v in current_progress.items()})

        def callback(model_type):
            def update(n_completed):
                current_progress[model_type] = n_completed
                blob.set_blob_metadata({k: str(v) for k, v in current_progress.items()})

            return update

        return callback
