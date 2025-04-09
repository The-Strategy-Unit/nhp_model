"""test docker run."""

import time
from unittest.mock import Mock, call, mock_open, patch

import pytest

from nhp.docker import config
from nhp.docker.run import RunWithAzureStorage, RunWithLocalStorage

# RunWithLocalStorage


def test_RunWithLocalStorage(mocker):
    # arrange
    m = mocker.patch("nhp.docker.run.load_params", return_value="params")

    # act
    s = RunWithLocalStorage("filename")

    # assert
    assert s.params == "params"

    m.assert_called_once_with("queue/filename")


def test_RunWithLocalStorage_finish(mocker):
    # arrange
    mocker.patch("nhp.docker.run.load_params", return_value="params")
    s = RunWithLocalStorage("filename")

    # act
    s.finish("results", ["saved_files"], False)

    # assert (nothing to assert)


def test_RunWithLocalStorage_progress_callback(mocker):
    # arrange
    mocker.patch("nhp.docker.run.load_params", return_value="params")
    s = RunWithLocalStorage("filename")

    # act
    p = s.progress_callback()
    p("Inpatients")(5)

    # assert (nothing to assert)


# RunWithAzureStorage


@pytest.fixture
def mock_run_with_azure_storage():
    """Create a mock Model instance."""
    with patch.object(RunWithAzureStorage, "__init__", lambda *args: None):
        rwas = RunWithAzureStorage(None)  # type: ignore

    rwas.params = {}
    rwas._app_version = "dev"

    rwas._queue_storage_account_url = "queue_url"
    rwas._data_storage_account_url = "data_url"
    rwas._results_storage_account_url = "results_url"

    return rwas


@pytest.mark.parametrize(
    "args, expected_version", [(["filename"], "dev"), (["filename", "v0.3.5"], "v0.3")]
)
def test_RunWithAzureStorage_init(mocker, args, expected_version):
    # arrange
    expected_params = {"start_year": 2020, "dataset": "synthetic"}
    gpm = mocker.patch(
        "nhp.docker.run.RunWithAzureStorage._get_params",
        return_value=expected_params,
    )
    gdm = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_data")

    config.QUEUE_STORAGE_ACCOUNT_URL = "queue_url"
    config.DATA_STORAGE_ACCOUNT_URL = "data_url"
    config.RESULTS_STORAGE_ACCOUNT_URL = "results_url"

    # act
    s = RunWithAzureStorage(*args)  # type: ignore

    # assert
    assert s._app_version == expected_version
    assert s.params == expected_params
    assert s._queue_storage_account_url == "queue_url"
    assert s._data_storage_account_url == "data_url"
    assert s._results_storage_account_url == "results_url"

    gpm.assert_called_once_with("filename")

    gdm.assert_called_once_with(2020, "synthetic")


@pytest.mark.parametrize(
    "queue_url, results_url, data_url, error_str",
    [
        (None, "results_url", "data_url", "QUEUE"),
        ("queue_url", None, "data_url", "RESULTS"),
        ("queue_url", "results_url", None, "DATA"),
    ],
)
def test_RunWithAzureStorage_init_checks_urls(mocker, queue_url, results_url, data_url, error_str):
    # arrange
    expected_params = {"start_year": 2020, "dataset": "synthetic"}
    mocker.patch(
        "nhp.docker.run.RunWithAzureStorage._get_params",
        return_value=expected_params,
    )
    mocker.patch("nhp.docker.run.RunWithAzureStorage._get_data")

    config.QUEUE_STORAGE_ACCOUNT_URL = queue_url
    config.DATA_STORAGE_ACCOUNT_URL = data_url
    config.RESULTS_STORAGE_ACCOUNT_URL = results_url

    # act & assert
    with pytest.raises(
        ValueError, match=f"{error_str}_STORAGE_ACCOUNT_URL environment variable not set"
    ):
        RunWithAzureStorage(None)  # type: ignore


def test_RunWithAzureStorage_get_container(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    mock = Mock()
    mock.get_container_client.return_value = "container_client"

    bsc_m = mocker.patch("nhp.docker.run.BlobServiceClient", return_value=mock)

    dac_m = mocker.patch("nhp.docker.run.DefaultAzureCredential", return_value="cred")

    # act
    actual = s._get_container("url", "container")

    # assert
    assert actual == "container_client"
    bsc_m.assert_called_once_with(account_url="url", credential="cred")
    dac_m.assert_called_once_with()


def test_RunWithAzureStorage_get_params(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    expected = {"dataset": "synthetic"}

    m1 = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_container")
    m2 = Mock()

    m1().get_blob_client.return_value = m2

    m2.download_blob().readall.return_value = """{"dataset": "synthetic"}"""

    m1.reset_mock()
    m2.reset_mock()

    # act
    actual = s._get_params("filename")

    # assert
    assert actual == expected

    m1.assert_called_once_with("queue_url", "queue")
    assert s._queue_blob == m2

    m2.download_blob.assert_called_once_with()
    m2.download_blob().readall.assert_called_once()


def test_RunWithAzureStorage_get_data(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    def paths_helper(i):
        m = Mock()
        m.name = str(i)
        return m

    files = [
        paths_helper(f"dev/{i}/fyear=2020/dataset=synthetic/0.parquet") for i in ["ip", "op", "aae"]
    ]

    mkdir_m = mocker.patch("os.makedirs", return_value=False)

    m1 = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_container")
    m2 = m1().list_blobs = Mock(return_value=files)
    m3 = m1().download_blob = Mock()

    m1.reset_mock()

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        s._get_data(2020, "synthetic")

    # assert
    m1.assert_called_once_with("data_url", "data")
    m2.assert_called_once_with(name_starts_with="dev/fyear=2020/dataset=synthetic")
    assert m3.call_args_list == [
        call("dev/ip/fyear=2020/dataset=synthetic/0.parquet"),
        call("dev/op/fyear=2020/dataset=synthetic/0.parquet"),
        call("dev/aae/fyear=2020/dataset=synthetic/0.parquet"),
    ]

    assert mkdir_m.call_args_list == [
        call("data/ip/fyear=2020/dataset=synthetic", exist_ok=True),
        call("data/op/fyear=2020/dataset=synthetic", exist_ok=True),
        call("data/aae/fyear=2020/dataset=synthetic", exist_ok=True),
    ]

    assert mock_file.call_args_list == [
        call("data/ip/fyear=2020/dataset=synthetic/0.parquet", "wb"),
        call("data/op/fyear=2020/dataset=synthetic/0.parquet", "wb"),
        call("data/aae/fyear=2020/dataset=synthetic/0.parquet", "wb"),
    ]


def test_RunWithAzureStorage_upload_results_json(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    m = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_container")
    mocker.patch("gzip.compress", return_value="gzdata")

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s._upload_results_json("filename", "metadata")

    # assert
    mock_file.assert_called_once_with("results/filename.json", "rb")
    m.assert_called_once_with("results_url", "results")
    m().upload_blob.assert_called_once_with(
        "prod/dev/filename.json.gz", "gzdata", metadata="metadata", overwrite=True
    )


def test_RunWithAzureStorage_upload_results_files(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage

    m = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_container")
    metadata = {"k": "v"}

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s._upload_results_files(["results/filename", "results/filename.json"], metadata)

    # assert
    assert mock_file.call_args_list == [
        call("results/filename", "rb"),
        call("results/filename.json", "rb"),
    ]
    m.assert_called_once_with("results_url", "results")
    m().upload_blob.assert_has_calls(
        [
            call(
                "aggregated-model-results/dev/filename",
                "data",
                overwrite=True,
                metadata=None,
            ),
            call(
                "aggregated-model-results/dev/filename.json",
                "data",
                overwrite=True,
                metadata=metadata,
            ),
        ]
    )


def test_RunWithAzureStorage_upload_full_model_results(mock_run_with_azure_storage, mocker):
    # arrange
    s = mock_run_with_azure_storage
    s.params["dataset"] = "synthetic"
    s.params["scenario"] = "test"
    s.params["create_datetime"] = "20240101_012345"

    def create_file_mock(name):
        fm = Mock()
        fm.as_posix.return_value = f"results/{name}"
        return fm

    file_mocks = list(map(create_file_mock, ["1", "2", "3"]))

    m = mocker.patch("nhp.docker.run.RunWithAzureStorage._get_container")
    path_mock = mocker.patch("nhp.docker.run.Path")
    path_mock().glob.return_value = file_mocks

    # act
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        s._upload_full_model_results()

    # assert
    assert mock_file.call_count == 3
    assert mock_file.call_args_list == [call(i, "rb") for i in file_mocks]
    m.assert_called_once_with("results_url", "results")

    assert m().upload_blob.call_count == 3
    assert m().upload_blob.call_args_list == [
        call(f"full-model-results/dev/{i}", "data", overwrite=True) for i in ["1", "2", "3"]
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
    m1 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_results_json")
    m2 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_results_files")
    m3 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_full_model_results")
    m4 = mocker.patch("nhp.docker.run.RunWithAzureStorage._cleanup")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]  # type: ignore
    params["dict"] = {"a": 1}  # type: ignore

    s.params = params

    # act
    s.finish("results_file", ["saved_files"], False)

    # assert
    m1.assert_called_once_with("results_file", metadata)
    m2.assert_called_once_with(["saved_files"], metadata)
    m3.assert_not_called()
    m4.assert_called_once()


def test_RunWithAzureStorage_finish_save_full_model_results_true(
    mock_run_with_azure_storage, mocker
):
    # arrange
    s = mock_run_with_azure_storage
    m1 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_results_json")
    m2 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_results_files")
    m3 = mocker.patch("nhp.docker.run.RunWithAzureStorage._upload_full_model_results")
    m4 = mocker.patch("nhp.docker.run.RunWithAzureStorage._cleanup")

    metadata = {"id": "1", "dataset": "synthetic", "start_year": "2020"}
    params = metadata.copy()
    params["list"] = [1, 2]  # type: ignore
    params["dict"] = {"a": 1}  # type: ignore

    s.params = params

    # act
    s.finish("results_file", ["saved_files"], True)

    # assert
    m1.assert_called_once_with("results_file", metadata)
    m2.assert_called_once_with(["saved_files"], metadata)
    m3.assert_called_once()
    m4.assert_called_once()


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
