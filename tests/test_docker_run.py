"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import config
from docker_run import load_params

config.STORAGE_ACCOUNT = "sa"
config.APP_VERSION = "dev"


def test_load_params_skips_if_file_exists(mocker):
    # arrange
    expected = {
        "dataset": "synthetic",
        "id": "3d7a84337922277b8e6205011cbc168c1a008aa746c8c13e07031b01d42acfc7",
        "app_version": "dev",
    }

    path_m = mocker.patch("os.path.exists", return_value=True)

    # act
    with patch(
        "builtins.open", mock_open(read_data='{"dataset": "synthetic"}'.encode())
    ) as mock_file:
        actual = load_params("filename")

    # assert
    assert actual == expected
    path_m.assert_called_once_with("queue/filename")
    assert mock_file.call_args == call("queue/filename", "rb")


def test_load_params_from_blob_storage(mocker):
    # arrange
    expected = {
        "dataset": "synthetic",
        "id": "3d7a84337922277b8e6205011cbc168c1a008aa746c8c13e07031b01d42acfc7",
        "app_version": "dev",
    }

    path_m = mocker.patch("os.path.exists", return_value=False)

    mock = Mock()
    mock.get_container_client.return_value = mock
    mock.download_blob.return_value = mock
    mock.readall.return_value = '{"dataset": "synthetic"}'.encode()

    bsc_m = mocker.patch("docker_run.BlobServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    # act
    actual = load_params("filename")

    # assert
    assert actual == expected

    path_m.expect_called_once_with("queue/filename")
    bsc_m.assert_called_once_with(
        account_url="https://sa.blob.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()

    mock.get_container_client.assert_called_once_with("queue")
    mock.download_blob.assert_called_once_with("filename")
    mock.readall.assert_called_once()
