"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pytest

import config
from docker_run import _upload_results, get_data, load_params, main

config.STORAGE_ACCOUNT = "sa"
config.APP_VERSION = "dev"
config.DATA_VERSION = "dev"
config.KEYVAULT_ENDPOINT = "keyvault_endpoint"
config.COSMOS_ENDPOINT = "cosmos_endpoint"
config.COSMOS_DB = "cosmos_db"


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


def test_get_data_does_nothing_if_data_exists(mocker):
    # arrange
    path_m = mocker.patch("os.path.exists", return_value=True)

    # act
    get_data("synthetic")

    # assert
    path_m.assert_called_once_with("data/synthetic")


def test_get_data_gets_data_from_blob(mocker):
    # arrange
    files = [1, 2, 3, 4]

    def paths_helper(i):
        m = Mock()
        m.name = str(i)
        return m

    path_m = mocker.patch("os.path.exists", return_value=False)
    mkdir_m = mocker.patch("os.makedirs", return_value=False)

    mock = Mock()
    mock.get_file_system_client.return_value = mock
    mock.get_paths.return_value = [paths_helper(i) for i in files]
    mock.get_file_client.return_value = mock
    mock.download_file.return_value = mock
    mock.readall.side_effect = files

    dlsc_m = mocker.patch("docker_run.DataLakeServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        get_data("synthetic")

    # assert

    path_m.expect_called_once_with("data/synthetic")
    dlsc_m.assert_called_once_with(
        account_url="https://sa.dfs.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()

    mock.get_file_system_client.assert_called_once_with("data")
    mock.get_paths.assert_called_once_with("dev/synthetic")

    mkdir_m.assert_called_once_with("data/synthetic", exist_ok=True)

    assert mock.get_file_client.call_args_list == [call(str(i)) for i in files]
    assert mock.download_file.call_args_list == [call() for _ in files]
    assert mock.readall.call_args_list == [call() for _ in files]

    assert mock_file.call_args_list == [call(f"data{i}", "wb") for i in files]


def test_upload_results(mocker):
    # arrange
    bsc_m = mocker.patch("docker_run.BlobServiceClient")
    mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")
    mocker.patch("gzip.compress", return_value="gzdata")

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        _upload_results("filename")

    # assert
    mock_file.assert_called_once_with("results/filename", "rb")
    bsc_m.assert_called_once_with(
        account_url="https://sa.blob.core.windows.net", credential="cred"
    )
    bsc_m().get_container_client.assert_called_once_with("results")
    bsc_m().get_container_client().upload_blob.assert_called_once_with(
        "filename.gz", "gzdata"
    )


def test_main(mocker):
    # arrange
    args = Mock()
    args.params_file = "params.json"

    mocker.patch("argparse.ArgumentParser", return_value=args)
    args.parse_args.return_value = args

    params = {"dataset": "synthetic"}

    lp_m = mocker.patch("docker_run.load_params", return_value=params)
    gd_m = mocker.patch("docker_run.get_data")
    ru_m = mocker.patch("docker_run.run_all", return_value="results.json.gz")
    ur_m = mocker.patch("docker_run._upload_results")

    # act
    main()

    # assert
    assert args.add_argument.call_args_list == [
        call(
            "params_file",
            nargs="?",
            default="sample_params.json",
            help="Name of the parameters file stored in Azure",
        )
    ]
    args.parse_args.assert_called_once()

    lp_m.assert_called_once_with("params.json")
    gd_m.assert_called_once_with("synthetic")
    ru_m.assert_called_once_with(params, "data")

    ur_m.assert_called_once_with("results.json.gz")


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
