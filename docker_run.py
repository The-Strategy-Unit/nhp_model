#!/opt/conda/bin/python
"""Run the model inside of the docker container"""

import argparse
import gzip
import json
import logging
import os
import re

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

import config
from model.helpers import load_params
from run_model import run_all


class RunWithLocalStorage:
    """Methods for running with local storage"""

    def get_params(self, filename: str) -> dict:
        """Get the parameters for the model

        :param filename: the name of the params file
        :type filename: str
        :return: the parameters for the model
        :rtype: dict
        """
        return load_params(f"queue/{filename}")

    def get_data(self, path: str) -> None:
        """Get data to run the model

        for local storage, the data is already available, so do nothing.

        :param path: the path to load the files from
        :type path: str
        """

    def upload_results(self, results_file: str, metadata: dict) -> None:
        """Upload the results

        for local storage there is nothing else to do

        :param results_file: the saved results file
        :type results_file: str
        :param metadata: the metadata to attach to the blob
        :type metadata: dict
        """

    def cleanup(self) -> None:
        """Cleanup

        for local storage there is nothing else to do
        """

    def progress_callback(self) -> None:
        """Progress callback method

        for local storage do nothing
        """
        return lambda _: lambda _: None


class RunWithAzureStorage:
    """Methods for running with azure storage"""

    def __init__(self, app_version: str = "dev"):
        self._app_version = re.sub("(\\d+\\.\\d+)\\..*", "\\1", app_version)
        self._queue_blob = None

        logging.getLogger("azure.storage.common.storageclient").setLevel(
            logging.WARNING
        )
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
            logging.WARNING
        )

    def _get_container(self, container_name: str):
        return BlobServiceClient(
            account_url=f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        ).get_container_client(container_name)

    def get_params(self, filename: str) -> dict:
        """Get the parameters for the model

        :param filename: the name of the params file
        :type filename: str
        :return: the parameters for the model
        :rtype: dict
        """
        logging.info("downloading params: %s", filename)

        self._queue_blob = self._get_container("queue").get_blob_client(filename)

        params_content = self._queue_blob.download_blob().readall()

        return json.loads(params_content)

    def get_data(self, path: str) -> None:
        """Get data to run the model

        for local storage, the data is already available, so do nothing.

        :param path: the path to load the files from
        :type path: str
        """
        logging.info("downloading data (%s)", path)
        fs_client = DataLakeServiceClient(
            account_url=f"https://{config.STORAGE_ACCOUNT}.dfs.core.windows.net",
            credential=DefaultAzureCredential(),
        ).get_file_system_client("data")

        version = config.DATA_VERSION
        directory_path = f"{version}/{path}"

        paths = [p.name for p in fs_client.get_paths(directory_path)]

        os.makedirs(f"data/{path}", exist_ok=True)

        for filename in paths:
            logging.info(" * %s", filename)
            local_name = "data" + filename.removeprefix(version)
            with open(local_name, "wb") as local_file:
                file_client = fs_client.get_file_client(filename)
                local_file.write(file_client.download_file().readall())

    def upload_results(self, results_file: str, metadata: dict) -> None:
        """Upload the results

        once the model has run, upload the results to blob storage

        :param results_file: the saved results file
        :type results_file: str
        :param metadata: the metadata to attach to the blob
        :type metadata: dict
        """
        container = self._get_container("results")

        with open(f"results/{results_file}.json", "rb") as file:
            container.upload_blob(
                f"{self._app_version}/{results_file}.json.gz",
                gzip.compress(file.read()),
                metadata=metadata,
            )

    def cleanup(self) -> None:
        """Cleanup

        once the model has run, remove the file from the queue
        """
        logging.info("cleaning up queue")

        self._queue_blob.delete_blob()

    def progress_callback(self) -> None:
        """Progress callback method

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="sample_params.json",
        help="Name of the parameters file stored in Azure",
    )

    parser.add_argument(
        "--local-storage",
        "-l",
        action="store_true",
        help="Use local storage (instead of Azure)",
    )

    return parser.parse_args()


def main():
    """the main method"""

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("running model for: %s", args.params_file)

    if args.local_storage:
        runner = RunWithLocalStorage()
    else:
        runner = RunWithAzureStorage(config.APP_VERSION)

    params = runner.get_params(args.params_file)
    runner.get_data(f"{params['start_year']}/{params['dataset']}")
    runner.get_data("reference")

    results_file = run_all(params, "data", runner.progress_callback)

    metadata = {
        k: str(v)
        for k, v in params.items()
        if not isinstance(v, dict) and not isinstance(v, list)
    }
    runner.upload_results(results_file, metadata)
    runner.cleanup()

    logging.info("complete")


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
