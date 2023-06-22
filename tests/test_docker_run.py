"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pytest

import config
from docker_run import RunWithAzureStorage, RunWithLocalStorage, main, parse_args

config.STORAGE_ACCOUNT = "sa"
config.APP_VERSION = "dev"
config.DATA_VERSION = "dev"

# RunWithLocalStorage


def test_RunWithLocalStorage_get_params(mocker):
    # arrange
    m = mocker.patch("docker_run.load_params", return_value="params")

    s = RunWithLocalStorage()

    # act
    actual = s.get_params("filename")

    # assert
    assert actual == "params"

    m.assert_called_once_with("queue/filename")


def test_RunWithLocalStorage_get_data():
    # arrange
    s = RunWithLocalStorage()

    # act
    s.get_data("data")

    # assert (nothing to assert)


def test_RunWithLocalStorage_upload_results():
    # arrange
    s = RunWithLocalStorage()

    # act
    s.upload_results("filename", "metadata")

    # assert (nothing to assert)


def test_RunWithLocalStorage_cleanup():
    # arrange
    s = RunWithLocalStorage()

    # act
    s.cleanup()

    # assert (nothing to assert)


def test_RunWithLocalStorage_progress_callback():
    # arrange
    s = RunWithLocalStorage()

    # act
    p = s.progress_callback()
    p("Inpatients")(5)

    # assert (nothing to assert)


# RunWithAzureStorage
def test_RunWithAzureStorage_init():
    # act
    a1 = RunWithAzureStorage()
    a2 = RunWithAzureStorage("v0.3.5")

    # assert
    assert a1._app_version == "dev"
    assert a2._app_version == "v0.3"

    assert not a1._queue_blob
    assert not a2._queue_blob


def test_RunWithAzureStorage_get_container(mocker):
    # arrange
    mock = Mock()
    mock.get_container_client.return_value = "container_client"

    bsc_m = mocker.patch("docker_run.BlobServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    s = RunWithAzureStorage()

    # act
    actual = s._get_container("container")

    # assert
    assert actual == "container_client"
    bsc_m.assert_called_once_with(
        account_url="https://sa.blob.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()


def test_RunWithAzureStorage_get_params(mocker):
    # arrange
    expected = {"dataset": "synthetic"}

    m1 = mocker.patch("docker_run.RunWithAzureStorage._get_container")
    m2 = Mock()

    m1().get_blob_client.return_value = m2

    m2.download_blob().readall.return_value = """{"dataset": "synthetic"}"""

    m1.reset_mock()
    m2.reset_mock()

    s = RunWithAzureStorage()

    # act
    actual = s.get_params("filename")

    # assert
    assert actual == expected

    m1.assert_called_once_with("queue")
    assert s._queue_blob == m2

    m2.download_blob.assert_called_once_with()
    m2.download_blob().readall.assert_called_once()


def test_RunWithAzureStorage_get_data(mocker):
    # arrange
    files = [1, 2, 3, 4]

    def paths_helper(i):
        m = Mock()
        m.name = str(i)
        return m

    mkdir_m = mocker.patch("os.makedirs", return_value=False)

    mock = Mock()
    mock.get_file_system_client.return_value = mock
    mock.get_paths.return_value = [paths_helper(i) for i in files]
    mock.get_file_client.return_value = mock
    mock.download_file.return_value = mock
    mock.readall.side_effect = files

    dlsc_m = mocker.patch("docker_run.DataLakeServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    s = RunWithAzureStorage()

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        s.get_data("2020/synthetic")

    # assert

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


def test_RunWithAzureStorage_upload_results(mocker):
    # arrange
    m = mocker.patch("docker_run.RunWithAzureStorage._get_container")
    mocker.patch("gzip.compress", return_value="gzdata")
    s = RunWithAzureStorage()

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s.upload_results("filename", "metadata")

    # assert
    mock_file.assert_called_once_with("results/filename.json", "rb")
    m.assert_called_once_with("results")
    m().upload_blob.assert_called_once_with(
        "dev/filename.json.gz", "gzdata", metadata="metadata"
    )


def test_RunWithAzureStorage_cleanup():
    # arrange
    s = RunWithAzureStorage()
    m = s._queue_blob = Mock()

    # act
    s.cleanup()

    # assert
    m.delete_blob.assert_called_once_with()


def test_RunWithAzureStorage_progress_callback():
    # arrange
    s = RunWithAzureStorage()
    m = s._queue_blob = Mock()
    m.get_blob_properties.return_value = {"metadata": {"id": 1}}

    # (1) the initial set up
    # act (1)
    p = s.progress_callback()

    # assert (1)
    m.get_blob_properties.assert_called_once_with()
    sbm = m.set_blob_metadata

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


@pytest.mark.parametrize(
    "args, expected_file, expected_local_storage",
    [
        ([], "sample_params.json", False),
        (["-l"], "sample_params.json", True),
        (["test.json"], "test.json", False),
        (["test.json", "-l"], "test.json", True),
    ],
)
def test_parse_args(mocker, args, expected_file, expected_local_storage):
    # arrange
    mocker.patch("sys.argv", ["docker_run.py"] + args)

    # act
    actual = parse_args()

    # assert
    assert actual.params_file == expected_file
    assert actual.local_storage == expected_local_storage


def test_main_local(mocker):
    # arrange
    m = mocker.patch("docker_run.parse_args")
    m().params_file = "params.json"
    m().local_storage = True

    rwls = mocker.patch("docker_run.RunWithLocalStorage")
    rwas = mocker.patch("docker_run.RunWithAzureStorage")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    rwls().get_params.return_value = params
    rwls.reset_mock()

    ru_m = mocker.patch("docker_run.run_all", return_value="results.json")

    # act
    main()

    # assert
    rwls.assert_called_once()
    rwas.assert_not_called()

    s = rwls()

    s.get_params.assert_called_once_with("params.json")
    assert s.get_data.call_count == 2
    assert s.get_data.call_args_list == [call("2020/synthetic"), call("reference")]

    ru_m.assert_called_once_with(params, "data", s.progress_callback)

    s.upload_results.assert_called_once_with("results.json", metadata)
    s.cleanup.assert_called_once_with()


def test_main_azure(mocker):
    # arrange
    m = mocker.patch("docker_run.parse_args")
    m().params_file = "params.json"
    m().local_storage = False

    rwls = mocker.patch("docker_run.RunWithLocalStorage")
    rwas = mocker.patch("docker_run.RunWithAzureStorage")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    rwas().get_params.return_value = params
    rwas.reset_mock()

    ru_m = mocker.patch("docker_run.run_all", return_value="results.json")

    # act
    main()

    # assert
    rwls.assert_not_called()
    rwas.assert_called_once_with("dev")

    s = rwas()

    s.get_params.assert_called_once_with("params.json")
    assert s.get_data.call_count == 2
    assert s.get_data.call_args_list == [call("2020/synthetic"), call("reference")]

    ru_m.assert_called_once_with(params, "data", s.progress_callback)

    s.upload_results.assert_called_once_with("results.json", metadata)
    s.cleanup.assert_called_once_with()


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
