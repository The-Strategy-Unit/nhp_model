"""Run the model inside of the docker container."""

import gzip
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import pandas as pd
from azure.data.tables import TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from nhp.docker.config import Config
from nhp.model.params import load_params
from nhp.model.results import generate_results_json, save_results_files
from nhp.model.run import noop_progress_callback


class RunWithLocalStorage:
    """Methods for running with local storage."""

    def __init__(self, filename: str):
        """Initialize the RunWithLocalStorage instance.

        Args:
            filename: Name of the parameter file to load.
        """
        self.params = load_params(f"queue/{filename}")

    def finish(
        self,
        results: dict[str, pd.DataFrame],
        variants: list[str],
        _save_full_model_results: bool,
        _additional_metadata: dict[str, Any],
    ) -> None:
        """Post model run steps.

        Args:
            results: A dictionary containing the results dataframes.
            variants: A list of the variants that were run.
            save_full_model_results: Whether to save the full model results or not.
            additional_metadata: Additional metadata to log.
        """
        save_results_files(results, self.params, variants)

    def error(self, error_message: str) -> None:
        """Error handling.

        If there is an error during the model run, log the error message.

        Args:
            error_message: The error message to log.
        """
        pass

    def progress_callback(self) -> Callable[[Any], Callable[[Any], None]]:
        """Progress callback method.

        For local storage do nothing.

        Returns:
            A no-op progress callback function.
        """
        return noop_progress_callback


class RunWithAzureStorage:
    """Methods for running with azure storage."""

    def __init__(self, model_run_id: UUID, filename: str, config: Config | None = None):
        """Initialise RunWithAzureStorage.

        Args:
            model_run_id: Unique identifier for this model run.
            filename: Name of the parameter file to load.
            config: The configuration for the run. Defaults to Config().
        """
        logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
            logging.WARNING
        )
        self._model_run_id = model_run_id
        self._config = config or Config()

        self._app_version = re.sub("(\\d+\\.\\d+)\\..*", "\\1", self._config.APP_VERSION)

        self._blob_storage_account_url = (
            f"https://{self._config.STORAGE_ACCOUNT}.blob.core.windows.net"
        )
        self._adls_storage_account_url = (
            f"https://{self._config.STORAGE_ACCOUNT}.dfs.core.windows.net"
        )
        self._table_storage_account_url = (
            f"https://{self._config.STORAGE_ACCOUNT}.table.core.windows.net"
        )

        self.params = self._get_params(filename)
        self._get_data(self.params["start_year"], self.params["dataset"])

        self._table_client = TableServiceClient(
            endpoint=self._table_storage_account_url,
            credential=DefaultAzureCredential(),
        ).get_table_client("modelruns")

        self._update_table_storage(status="running")

    def _get_container(self, container_name: str):
        return BlobServiceClient(
            account_url=self._blob_storage_account_url,
            credential=DefaultAzureCredential(),
        ).get_container_client(container_name)

    def _get_params(self, filename: str) -> dict:
        """Get the parameters for the model.

        Args:
            filename: The name of the params file.

        Returns:
            The parameters for the model.
        """
        logging.info("downloading params: %s", filename)

        self._queue_blob = self._get_container("queue").get_blob_client(filename)

        params_content = self._queue_blob.download_blob().readall()

        return json.loads(params_content)

    def _get_data(self, year: str, dataset: str) -> None:
        """Get data to run the model.

        Downloads data from Azure storage for the specified year and dataset.

        Args:
            year: The year of data to load.
            dataset: The dataset to load.
        """
        logging.info("downloading data (%s / %s)", year, dataset)
        fs_client = DataLakeServiceClient(
            account_url=self._adls_storage_account_url,
            credential=DefaultAzureCredential(),
        ).get_file_system_client("data")

        version = self._config.DATA_VERSION

        paths = [p.name for p in fs_client.get_paths(version, recursive=False)]

        for p in paths:
            subpath = f"{p}/fyear={year}/dataset={dataset}"
            os.makedirs(f"data{subpath.removeprefix(version)}", exist_ok=True)

            for i in fs_client.get_paths(subpath):
                filename = i.name
                if not filename.endswith("parquet"):
                    continue

                logging.info(" * %s", filename)
                local_name = "data" + filename.removeprefix(version)
                with open(local_name, "wb") as local_file:
                    file_client = fs_client.get_file_client(filename)
                    local_file.write(file_client.download_file().readall())

    def _upload_results_json(
        self, results: dict[str, pd.DataFrame], metadata: dict[str, Any], variants: list[str]
    ) -> None:
        """Upload the results.

        Once the model has run, upload the results to blob storage.

        Args:
            results: Dictionary containing the results dataframes.
            metadata: The metadata to attach to the blob.
            variants: A list of the variants that were run.
        """
        container = self._get_container("results")

        results_file = generate_results_json(results, self.params, variants)

        with open(f"results/{results_file}.json", "rb") as file:
            container.upload_blob(
                f"prod/{self._app_version}/{results_file}.json.gz",
                gzip.compress(file.read()),
                metadata={k: str(v) for k, v in metadata.items()},
                overwrite=True,
            )

    def _upload_results_files(
        self,
        file_path: str,
        results: dict[str, pd.DataFrame],
        metadata: dict[str, str],
        variants: list[str],
    ) -> None:
        """Upload the results.

        Once the model has run, upload the files (parquet for model results and json for
        model params) to blob storage.

        Args:
            file_path: The path to save the results to.
            results: A dictionary containing the results dataframes.
            metadata: The metadata to attach to the blob.
            variants: A list of the variants that were run.
        """
        params = self.params
        container = self._get_container("results")
        for k, v in results.items():
            container.upload_blob(
                file_path + f"/{k}.parquet",
                v.to_parquet(index=False),
                overwrite=True,
                metadata=metadata,
            )
        container.upload_blob(
            f"{file_path}/params.json",
            json.dumps(params).encode("utf-8"),
            overwrite=True,
            metadata=metadata,
        )
        container.upload_blob(
            f"{file_path}/variants.json",
            json.dumps(variants).encode("utf-8"),
            overwrite=True,
            metadata=metadata,
        )

    def _upload_full_model_results(self) -> None:
        container = self._get_container("results")

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

    def _update_table_storage(self, **kwargs) -> None:
        """Update the table storage with the given data."""
        entity = {
            "PartitionKey": self.params["dataset"],
            "RowKey": self._model_run_id,
            **kwargs,
        }

        self._table_client.update_entity(entity, mode=UpdateMode.MERGE)

    def _cleanup(self) -> None:
        """Cleanup.

        Once the model has run, remove the file from the queue.
        """
        logging.info("cleaning up queue")

        self._queue_blob.delete_blob()

    def finish(
        self,
        results: dict[str, pd.DataFrame],
        variants: list[str],
        save_full_model_results: bool,
        additional_metadata: dict[str, Any],
    ) -> None:
        """Post model run steps.

        Args:
            results: A dictionary containing the results dataframes.
            variants: A list of the variants that were run.
            save_full_model_results: Whether to save the full model results or not.
            additional_metadata: Additional metadata to log.
        """
        metadata = {
            k: v
            for k, v in self.params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }
        metadata.update(additional_metadata)

        file_path = "/".join(
            [
                "aggregated-model-results",
                self._app_version,
                self.params["dataset"],
                self.params["scenario"],
                self.params["create_datetime"],
            ]
        )
        self._update_table_storage(
            status="complete",
            file_path=file_path,
            outputs_app_path=f"{self._app_version}/{self._model_run_id}",
        )

        self._upload_results_files(
            file_path, results, {"model_run_id": str(self._model_run_id)}, variants
        )
        # see issue #286, this should be removed once we no longer need the results json file
        self._upload_results_json(results, metadata, variants)
        if save_full_model_results:
            self._upload_full_model_results()

        self._cleanup()

    def error(self, error_message: str) -> None:
        """Error handling.

        If there is an error during the model run, update the table storage with the error
        message and clean up the queue.

        Args:
            error_message: The error message to log.
        """
        self._update_table_storage(status="error", error_message=error_message)

    def progress_callback(self) -> Callable[[Any], Callable[[Any], None]]:
        """Progress callback method.

        Updates the metadata for the blob in the queue to give progress.

        Returns:
            A callback function that updates progress for each model type.
        """
        blob = self._queue_blob

        current_progress = {
            **blob.get_blob_properties()["metadata"],
            "Inpatients": 0,
            "Outpatients": 0,
            "AaE": 0,
        }

        blob.set_blob_metadata({k: str(v) for k, v in current_progress.items()})

        def callback(model_type: Any) -> Callable[[Any], None]:
            def update(n_completed: Any) -> None:
                current_progress[model_type] = n_completed
                blob.set_blob_metadata({k: str(v) for k, v in current_progress.items()})

            return update

        return callback
