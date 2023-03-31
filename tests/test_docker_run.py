"""test docker run"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pytest

import config
from docker_run import (
    _combine_results,
    _run_model,
    _split_model_runs_out,
    _upload_to_cosmos,
    get_data,
    load_params,
    main,
    run,
)
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

    mkdir_m.assert_called_once_with("data/synthetic", exist_ok=True)

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


def test_combine_results(mocker):
    # arrange
    m = mocker.patch("docker_run._split_model_runs_out")
    results = [
        [
            {
                **{
                    k: {
                        frozenset({("measure", "a"), ("pod", "a")}): 1 + i * 4 + j * 20,
                        frozenset({("measure", "a"), ("pod", "b")}): 2 + i * 4 + j * 20,
                        frozenset({("measure", "b"), ("pod", "a")}): 3 + i * 4 + j * 20,
                        frozenset({("measure", "b"), ("pod", "b")}): 4 + i * 4 + j * 20,
                    }
                    for k, i in [
                        ("a", 0),
                        ("b", 1),
                        ("c", 2),
                        ("d", 3),
                    ]
                },
                "step_counts": {
                    frozenset(
                        {
                            ("change_factor", "baseline"),
                            ("strategy", "-"),
                        }
                    ): [0, 1],
                    frozenset(
                        {
                            ("change_factor", "a"),
                            ("strategy", "a"),
                        }
                    ): [2 + j, 3 + j],
                },
            }
            for j in range(4)
        ],
        [{"a": {frozenset({("measure", "a"), ("pod", "a")}): 100}}],
    ]
    expected = {
        "a": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 101,
                "principal": 21,
                "model_runs": [41, 61],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 2,
                "principal": 22,
                "model_runs": [42, 62],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 3,
                "principal": 23,
                "model_runs": [43, 63],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 4,
                "principal": 24,
                "model_runs": [44, 64],
            },
        ],
        "b": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 5,
                "principal": 25,
                "model_runs": [45, 65],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 6,
                "principal": 26,
                "model_runs": [46, 66],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 7,
                "principal": 27,
                "model_runs": [47, 67],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 8,
                "principal": 28,
                "model_runs": [48, 68],
            },
        ],
        "c": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 9,
                "principal": 29,
                "model_runs": [49, 69],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 10,
                "principal": 30,
                "model_runs": [50, 70],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 11,
                "principal": 31,
                "model_runs": [51, 71],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 12,
                "principal": 32,
                "model_runs": [52, 72],
            },
        ],
        "d": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 13,
                "principal": 33,
                "model_runs": [53, 73],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 14,
                "principal": 34,
                "model_runs": [54, 74],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 15,
                "principal": 35,
                "model_runs": [55, 75],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 16,
                "principal": 36,
                "model_runs": [56, 76],
            },
        ],
        "step_counts": [
            {
                "change_factor": "baseline",
                "strategy": "-",
                "baseline": [0, 1],
                "principal": [0, 1],
                "model_runs": [[0, 1], [0, 1]],
            },
            {
                "change_factor": "a",
                "strategy": "a",
                "baseline": [2, 3],
                "principal": [3, 4],
                "model_runs": [[4, 5], [5, 6]],
            },
        ],
    }

    # act
    actual = _combine_results(results)

    # assert
    assert actual == expected
    m.call_args_list == [call(k, v) for k, v in expected.items()]


@pytest.mark.parametrize(
    "agg_type",
    [
        "default",
        "bed_occupancy",
        "theatres_available",
    ],
)
def test_split_model_runs_out_default(agg_type):
    # arrange
    results = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
        }
    ]
    expected = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
            "lwr_ci": 5.0,
            "median": 50.0,
            "upr_ci": 95.0,
        }
    ]

    # act
    _split_model_runs_out(agg_type, results)

    # assert
    assert results == expected


def test_split_model_runs_out_other():
    # arrange
    results = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
        }
    ]
    expected = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "lwr_ci": 5.0,
            "median": 50.0,
            "upr_ci": 95.0,
        }
    ]

    # act
    _split_model_runs_out("other", results)

    # assert
    assert results == expected


def test_split_model_runs_out_step_counts():
    # arrange
    results = [
        {
            "change_factor": "baseline",
            "strategy": "-",
            "baseline": [0, 1],
            "principal": [0, 1],
            "model_runs": [[0, 1], [0, 1]],
        },
        {
            "change_factor": "a",
            "strategy": "a",
            "baseline": [2, 3],
            "principal": [3, 4],
            "model_runs": [[4, 5], [5, 6]],
        },
    ]
    expected = [
        {"change_factor": "baseline", "baseline": [0, 1]},
        {
            "change_factor": "a",
            "strategy": "a",
            "principal": [3, 4],
            "model_runs": [[4, 5], [5, 6]],
        },
    ]

    # act
    _split_model_runs_out("step_counts", results)

    # assert
    assert results == expected


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


@pytest.mark.parametrize("skip_upload", [True, False])
def test_main(mocker, skip_upload):
    # arrange
    args = Mock()
    args.params_file = "params.json"
    args.skip_upload_to_cosmos = skip_upload

    mocker.patch("argparse.ArgumentParser", return_value=args)
    args.parse_args.return_value = args

    params = {"dataset": "synthetic"}

    lp_m = mocker.patch("docker_run.load_params", return_value=params)
    gd_m = mocker.patch("docker_run.get_data")
    ru_m = mocker.patch("docker_run.run", return_value="results")
    uc_m = mocker.patch("docker_run._upload_to_cosmos")

    # act
    main()

    # assert
    assert args.add_argument.call_args_list == [
        call(
            "params_file",
            nargs="?",
            default="sample_params.json",
            help="Name of the parameters file stored in Azure",
        ),
        call("--skip-upload-to-cosmos", action="store_true"),
    ]
    args.parse_args.assert_called_once()

    lp_m.assert_called_once_with("params.json")
    gd_m.assert_called_once_with("synthetic")
    ru_m.assert_called_once_with(params, "data")

    if skip_upload:
        uc_m.assert_not_called()
    else:
        uc_m.assert_called_once_with(params, "results")


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import docker_run as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("docker_run.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
