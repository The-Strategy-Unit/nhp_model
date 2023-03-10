"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import config
from docker_run import _run_model, _upload_to_cosmos, get_data, load_params, run
from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.outpatients import OutpatientsModel

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

    mkdir_m.assert_called_once_with("data/synthetic")

    assert mock.get_file_client.call_args_list == [call(str(i)) for i in files]
    assert mock.download_file.call_args_list == [call() for _ in files]
    assert mock.readall.call_args_list == [call() for _ in files]

    assert mock_file.call_args_list == [call(f"data{i}", "wb") for i in files]


def test_upload_to_cosmos(mocker):
    # arrange
    kyv_m = mocker.patch("docker_run.SecretClient")
    kyv_m().get_secret("cosmos-key").value = "cosmos_key"
    kyv_m.reset_mock()

    dac_m = mocker.patch("docker_run.DefaultAzureCredential", return_value="cred")

    mock = Mock()
    csc_m = mocker.patch("docker_run.CosmosClient", return_value=mock)
    mock.get_database_client.return_value = mock

    mock.get_container_client.return_value = mock

    mocker.patch("uuid.uuid4", side_effect=[1, 2])

    params = {"id": 1}
    results = {"a": 1, "b": 2}

    # act
    _upload_to_cosmos(params, results)

    # assert
    # these both should be called once, but the setup above forces an extra call
    kyv_m.assert_called_once_with("keyvault_endpoint", "cred")
    kyv_m().get_secret.assert_called_once_with("cosmos-key")

    csc_m.assert_called_once_with("cosmos_endpoint", "cosmos_key")
    dac_m.assert_called_once()

    mock.get_database_client.assert_called_once_with("cosmos_db")
    assert mock.get_container_client.call_args_list == [
        call("model_runs"),
        call("model_results"),
    ]

    assert mock.create_item.call_args_list == [
        call(
            {"id": "1", "model_run_id": 1, "aggregation": "a", "values": 1},
        ),
        call(
            {"id": "2", "model_run_id": 1, "aggregation": "b", "values": 2},
        ),
        call(
            {"id": 1},
        ),
    ]


def test_run_model(mocker):
    # arrange
    model_m = Mock()
    model_m.__name__ = "InpatientsModel"

    params = {"model_runs": 2}
    mocker.patch("os.cpu_count", return_value=2)

    pool_mock = mocker.patch("docker_run.Pool")
    pool_ctm = pool_mock.return_value.__enter__.return_value
    pool_ctm.name = "pool"
    pool_ctm.imap = Mock(wraps=lambda f, i, **kwargs: map(f, i))

    # act
    actual = _run_model(model_m, params, "data", "hsa", "run_params")

    # assert
    pool_ctm.imap.assert_called_once_with(model_m().go, [-1, 0, 1, 2], chunksize=16)
    assert actual == [model_m().go()] * 4


def test_run(mocker):
    # arrange
    grp_m = mocker.patch(
        "docker_run.Model.generate_run_params", return_value="run_params"
    )
    hsa_m = mocker.patch(
        "docker_run.HealthStatusAdjustmentInterpolated", return_value="hsa"
    )

    rm_m = mocker.patch("docker_run._run_model", side_effect=["ip", "op", "aae"])
    cr_m = mocker.patch("docker_run._combine_results", return_value="results")

    params = {"id": 1, "dataset": "synthetic", "life_expectancy": "le"}
    # act
    actual = run(params, "data")

    # assert
    assert actual == "results"

    grp_m.assert_called_once_with(params)
    hsa_m.assert_called_once_with("data/synthetic", "le")

    assert rm_m.call_args_list == [
        call(m, params, "data", "hsa", "run_params")
        for m in [InpatientsModel, OutpatientsModel, AaEModel]
    ]

    cr_m.assert_called_once_with(["ip", "op", "aae"])


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
