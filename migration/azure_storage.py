"""Methods for working with Azure Storage"""

import gzip

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

import config


def _get_container(container_name: str):
    """Get storage container

    :param container_name: the name of the container
    :type container_name: str
    :return: Container Client
    """
    return BlobServiceClient(
        account_url=f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_container_client(container_name)


def upload_results(results_file: str, app_version: str, metadata: dict) -> None:
    """Upload results to azure storage"""
    container = _get_container("results")

    with open(f"results/{results_file}.json", "rb") as file:
        container.upload_blob(
            f"{app_version}/{results_file}.json.gz",
            gzip.compress(file.read()),
            metadata=metadata,
        )
