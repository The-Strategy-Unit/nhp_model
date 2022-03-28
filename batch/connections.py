from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

import batch.config as config


def connect_blob_storage() -> BlobServiceClient:
    """
    Create a connection to blob storage
    """

    return BlobServiceClient(
        account_url=f"https://{config._STORAGE_ACCOUNT_NAME}.blob.core.windows.net/",
        credential=config._STORAGE_ACCOUNT_KEY,
    )


def connect_adls() -> DataLakeServiceClient:
    """
    Create a connection to Azure Data Lake Storage
    """

    return DataLakeServiceClient(
        account_url=f"https://{config._STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/",
        credential=config._STORAGE_ACCOUNT_KEY,
    )


def connect_batch_client() -> BatchServiceClient:
    """
    Create a connection a Azure Batch
    """

    credentials = SharedKeyCredentials(
        config._BATCH_ACCOUNT_NAME, config._BATCH_ACCOUNT_KEY
    )
    return BatchServiceClient(credentials, batch_url=config._BATCH_ACCOUNT_URL)
