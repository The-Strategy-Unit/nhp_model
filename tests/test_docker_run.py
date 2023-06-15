"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import config
from docker_run import (
    _cleanup,
    _get_container,
    _upload_results,
    get_data,
    load_params,
    main,
    progress_callback,
)

config.STORAGE_ACCOUNT = "sa"
config.APP_VERSION = "dev"
config.DATA_VERSION = "dev"


def test_get_container(mocker):
    # arrange
    mock = Mock()
    mock.get_container_client.return_value = "container_client"

    bsc_m = mocker.patch("docker_run.BlobServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    # act
    actual = _get_container("container")

    # assert
    assert actual == "container_client"
    bsc_m.assert_called_once_with(
        account_url="https://sa.blob.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()


def test_progress_callback(mocker):
    # arrange
    m = mocker.patch("docker_run._get_container")
    m().get_blob_client().get_blob_properties.return_value = {"metadata": {"id": 1}}
    m.reset_mock()

    # (1) the initial set up
    # act (1)
    p = progress_callback(1)

    # assert (1)
    m.assert_called_once_with("queue")
    m().get_blob_client.assert_called_once_with("1.json")
    sbm = m().get_blob_client().set_blob_metadata

    sbm.assert_called_once_with(
        {
            "id": "1",
            "Inpatients": "0",
            "Outpatients": "0",
            "AaE": "0",
        }
    )

    # (2) calling the callback
    # act (2)
    p("Inpatients")(5)

    # assert (2)
    sbm.assert_called_with(
        {
            "id": "1",
            "Inpatients": "5",
            "Outpatients": "0",
            "AaE": "0",
        }
    )


def test_load_params_from_blob_storage(mocker):
    # arrange
    expected = {"dataset": "synthetic"}

    path_m = mocker.patch("os.path.exists", return_value=False)

    m = mocker.patch("docker_run._get_container")
    m().download_blob().readall.return_value = """{"dataset": "synthetic"}"""
    m.reset_mock()

    # act
    actual = load_params("filename")

    # assert
    assert actual == expected

    path_m.expect_called_once_with("queue/filename")

    m.assert_called_once_with("queue")
    m().download_blob.assert_called_once_with("filename")
    m().download_blob().readall.assert_called_once()


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
        get_data("2020/synthetic")

    # assert

    path_m.expect_called_once_with("data/2020/synthetic")
    dlsc_m.assert_called_once_with(
        account_url="https://sa.dfs.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()

    mock.get_file_system_client.assert_called_once_with("data")
    mock.get_paths.assert_called_once_with("dev/2020/synthetic")

    mkdir_m.assert_called_once_with("data/2020/synthetic", exist_ok=True)

    assert mock.get_file_client.call_args_list == [call(str(i)) for i in files]
    assert mock.download_file.call_args_list == [call() for _ in files]
    assert mock.readall.call_args_list == [call() for _ in files]

    assert mock_file.call_args_list == [call(f"data{i}", "wb") for i in files]


def test_upload_results(mocker):
    # arrange
    m = mocker.patch("docker_run._get_container")
    mocker.patch("gzip.compress", return_value="gzdata")

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        _upload_results("filename", "metadata")

    # assert
    mock_file.assert_called_once_with("results/filename.json", "rb")
    m.assert_called_once_with("results")
    m().upload_blob.assert_called_once_with(
        "dev/filename.json.gz", "gzdata", metadata="metadata"
    )


def test_cleanup(mocker):
    # arrange
    m = mocker.patch("docker_run._get_container")

    # act
    _cleanup(1)

    # assert
    m.assert_called_once_with("queue")
    m().delete_blob.assert_called_once_with("1.json")


def test_main(mocker):
    # arrange
    args = Mock()
    args.params_file = "params.json"

    mocker.patch("argparse.ArgumentParser", return_value=args)
    args.parse_args.return_value = args

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    lp_m = mocker.patch("docker_run.load_params", return_value=params)
    gd_m = mocker.patch("docker_run.get_data")
    ru_m = mocker.patch("docker_run.run_all", return_value="results.json")
    ur_m = mocker.patch("docker_run._upload_results")
    cu_m = mocker.patch("docker_run._cleanup")

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
    assert gd_m.call_args_list == [call("2020/synthetic"), call("reference")]
    ru_m.assert_called_once_with(params, "data", progress_callback)

    ur_m.assert_called_once_with("results.json", metadata)
    cu_m.assert_called_once_with("1")


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
