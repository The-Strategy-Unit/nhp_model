#!/opt/conda/bin/python
"""Run the model inside of the docker container"""

import argparse
import gzip
import json
import logging
import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

import config
from run_model import run_all


def load_params(filename: str) -> dict:
    """Load model parameters

    if the file exists in the local folder `queue/`, then we will load the file from there.
    otherwise, the file will be downloaded from blob storage

    :param filename: the name of the parameter file to load
    :type filename: str
    :return: the parameters to use for a model run
    :rtype: dict
    """
    logging.info("downloading params: %s", filename)

    container = BlobServiceClient(
        account_url=f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_container_client("queue")

    params_content = container.download_blob(filename).readall()

    return json.loads(params_content)


def get_data(dataset: str) -> None:
    """Get the data to run the model

    if the data exists locally in the folder `data/`, then this function does nothing.
    otherwise, it will download the data from Azure Data Lake storage

    :param dataset: the name of the dataset that we are loading
    :type dataset: str
    """
    logging.info("downloading data")
    fs_client = DataLakeServiceClient(
        account_url=f"https://{config.STORAGE_ACCOUNT}.dfs.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_file_system_client("data")

    version = config.DATA_VERSION
    directory_path = f"{version}/{dataset}"

    paths = [p.name for p in fs_client.get_paths(directory_path)]

    os.makedirs(f"data/{dataset}", exist_ok=True)

    for filename in paths:
        logging.info(" * %s", filename)
        local_name = "data" + filename.removeprefix(version)
        with open(local_name, "wb") as local_file:
            file_client = fs_client.get_file_client(filename)
            local_file.write(file_client.download_file().readall())


def _upload_results(results_file: str) -> None:
    """"""
    container = BlobServiceClient(
        account_url=f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_container_client("results")

    with open(f"results/{config.APP_VERSION}/{results_file}", "rb") as file:
        container.upload_blob(f"{results_file}.gz", gzip.compress(file.read()))


def main():
    """the main method"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="sample_params.json",
        help="Name of the parameters file stored in Azure",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )

    logging.info("running model for: %s", args.params_file)

    params = load_params(args.params_file)
    get_data(params["dataset"])

    results_file = run_all(params, "data")

    _upload_results(results_file)

    logging.info("complete")


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
