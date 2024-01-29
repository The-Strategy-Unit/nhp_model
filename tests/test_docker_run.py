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


def test_RunWithLocalStorage(mocker):
    # arrange
    m = mocker.patch("docker_run.load_params", return_value="params")

    # act
    s = RunWithLocalStorage("filename")

    # assert
    assert s.params == "params"

    m.assert_called_once_with("queue/filename")


def test_RunWithLocalStorage_finish(mocker):
    # arrange
    mocker.patch("docker_run.load_params", return_value="params")
    s = RunWithLocalStorage("filename")

    # act
    s.finish("results", False)

    # assert (nothing to assert)


def test_RunWithLocalStorage_progress_callback(mocker):
    # arrange
    mocker.patch("docker_run.load_params", return_value="params")
    s = RunWithLocalStorage("filename")

    # act
    p = s.progress_callback()
    p("Inpatients")(5)

    # assert (nothing to assert)


# RunWithAzureStorage


@pytest.fixture
def mock_run_with_azure_storage():
    """create a mock Model instance"""
    with patch.object(RunWithAzureStorage, "__init__", lambda *args: None):
        rwas = RunWithAzureStorage(None)

    rwas.params = {}
    rwas._app_version = "dev"

    return rwas


@pytest.mark.parametrize(
    "args, expected_version", [(["filename"], "dev"), (["filename", "v0.3.5"], "v0.3")]
)
def test_RunWithAzureStorage_init(mocker, args, expected_version):
    # arrange
    expected_params = {"start_year": 2020, "dataset": "synthetic"}
    gpm = mocker.patch(
        "docker_run.RunWithAzureStorage._get_params",
        return_value=expected_params,
    )
    gdm = mocker.patch("docker_run.RunWithAzureStorage._get_data")

    # act
    s = RunWithAzureStorage(*args)

    # assert
    assert s._app_version == expected_version
    assert s.params == expected_params

    gpm.assert_called_once_with("filename")

    gdm.assert_called_once_with("2020/synthetic")


def test_RunWithAzureStorage_get_container(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    mock = Mock()
    mock.get_container_client.return_value = "container_client"

    bsc_m = mocker.patch("docker_run.BlobServiceClient", return_value=mock)

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    # act
    actual = s._get_container("container")

    # assert
    assert actual == "container_client"
    bsc_m.assert_called_once_with(
        account_url="https://sa.blob.core.windows.net", credential="cred"
    )
    dac_m.assert_called_once_with()


def test_RunWithAzureStorage_get_params(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    expected = {"dataset": "synthetic"}

    m1 = mocker.patch("docker_run.RunWithAzureStorage._get_container")
    m2 = Mock()

    m1().get_blob_client.return_value = m2

    m2.download_blob().readall.return_value = """{"dataset": "synthetic"}"""

    m1.reset_mock()
    m2.reset_mock()

    # act
    actual = s._get_params("filename")

    # assert
    assert actual == expected

    m1.assert_called_once_with("queue")
    assert s._queue_blob == m2

    m2.download_blob.assert_called_once_with()
    m2.download_blob().readall.assert_called_once()


def test_RunWithAzureStorage_get_data(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

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

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        s._get_data("2020/synthetic")

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


def test_RunWithAzureStorage_upload_results(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    m = mocker.patch("docker_run.RunWithAzureStorage._get_container")
    mocker.patch("gzip.compress", return_value="gzdata")

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s._upload_results("filename", "metadata")

    # assert
    mock_file.assert_called_once_with("results/filename.json", "rb")
    m.assert_called_once_with("results")
    m().upload_blob.assert_called_once_with(
        "dev/filename.json.gz", "gzdata", metadata="metadata", overwrite=True
    )


def test_RunWithAzureStorage_upload_full_model_results(
    mock_run_with_azure_storage, mocker
):
    # arrange
    s = mock_run_with_azure_storage
    s.params["dataset"] = "synthetic"
    s.params["id"] = "id"

    def create_file_mock(name):
        fm = Mock()
        fm.as_posix.return_value = f"results/{name}"
        return fm

    file_mocks = list(map(create_file_mock, ["1", "2", "3"]))

    m = mocker.patch("docker_run.RunWithAzureStorage._get_container")
    path_mock = mocker.patch("docker_run.Path")
    path_mock().glob.return_value = file_mocks

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s._upload_full_model_results()

    # assert
    assert mock_file.call_count == 3
    assert mock_file.call_args_list == [call(i, "rb") for i in file_mocks]
    m.assert_called_once_with("results")

    assert m().upload_blob.call_count == 3
    assert m().upload_blob.call_args_list == [
        call(f"dev/{i}", "data", overwrite=True) for i in ["1", "2", "3"]
    ]


def test_RunWithAzureStorage_cleanup(mock_run_with_azure_storage):
    # arrange
    s = mock_run_with_azure_storage
    m = s._queue_blob = Mock()

    # act
    s._cleanup()

    # assert
    m.delete_blob.assert_called_once_with()


def test_RunWithAzureStorage_finish_save_full_model_results_false(
    mock_run_with_azure_storage, mocker
):
    # arrange
    s = mock_run_with_azure_storage
    m1 = mocker.patch("docker_run.RunWithAzureStorage._upload_results")
    m2 = mocker.patch("docker_run.RunWithAzureStorage._upload_full_model_results")
    m3 = mocker.patch("docker_run.RunWithAzureStorage._cleanup")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    s.params = params

    # act
    s.finish("results_file", False)

    # assert
    m1.assert_called_once_with("results_file", metadata)
    m2.assert_not_called()
    m3.assert_called_once()


def test_RunWithAzureStorage_finish_save_full_model_results_true(
    mock_run_with_azure_storage, mocker
):
    # arrange
    s = mock_run_with_azure_storage
    m1 = mocker.patch("docker_run.RunWithAzureStorage._upload_results")
    m2 = mocker.patch("docker_run.RunWithAzureStorage._upload_full_model_results")
    m3 = mocker.patch("docker_run.RunWithAzureStorage._cleanup")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]
    params["dict"] = {"a": 1}

    s.params = params

    # act
    s.finish("results_file", True)

    # assert
    m1.assert_called_once_with("results_file", metadata)
    m2.assert_called_once()
    m3.assert_called_once()


def test_RunWithAzureStorage_progress_callback(mock_run_with_azure_storage):
    # arrange
    s = mock_run_with_azure_storage
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
    "args, expected_file, expected_local_storage, expected_save_full_model_results",
    [
        ([], "sample_params.json", False, False),
        (["-l"], "sample_params.json", True, False),
        (["test.json"], "test.json", False, False),
        (["test.json", "-l"], "test.json", True, False),
        (["--save-full-model-results"], "sample_params.json", False, True),
    ],
)
def test_parse_args(
    mocker,
    args,
    expected_file,
    expected_local_storage,
    expected_save_full_model_results,
):
    # arrange
    mocker.patch("sys.argv", ["docker_run.py"] + args)

    # act
    actual = parse_args()

    # assert
    assert actual.params_file == expected_file
    assert actual.local_storage == expected_local_storage
    assert actual.save_full_model_results == expected_save_full_model_results


def test_main_local(mocker):
    # arrange
    m = mocker.patch("docker_run.parse_args")
    m().params_file = "params.json"
    m().local_storage = True
    m().save_full_model_results = False

    rwls = mocker.patch("docker_run.RunWithLocalStorage")
    rwas = mocker.patch("docker_run.RunWithAzureStorage")

    rwls().params = "params"
    rwls.reset_mock()

    ru_m = mocker.patch("docker_run.run_all", return_value="results.json")

    # act
    main()

    # assert
    rwls.assert_called_once_with("params.json")
    rwas.assert_not_called()

    s = rwls()
    ru_m.assert_called_once_with("params", "data", s.progress_callback, False)
    s.finish.assert_called_once_with("results.json", False)


def test_main_azure(mocker):
    # arrange
    m = mocker.patch("docker_run.parse_args")
    m().params_file = "params.json"
    m().local_storage = False
    m().save_full_model_results = False

    rwls = mocker.patch("docker_run.RunWithLocalStorage")
    rwas = mocker.patch("docker_run.RunWithAzureStorage")

    rwas().params = "params"
    rwas.reset_mock()

    ru_m = mocker.patch("docker_run.run_all", return_value="results.json")

    # act
    main()

    # assert
    rwls.assert_not_called()
    rwas.assert_called_once_with("params.json", "dev")

    s = rwas()
    ru_m.assert_called_once_with("params", "data", s.progress_callback, False)
    s.finish.assert_called_once_with("results.json", False)


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
