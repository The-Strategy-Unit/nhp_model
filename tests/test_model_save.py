"""test hsa gams"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

import os
import sys
from collections import namedtuple
from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.model_save import CosmosDBSave, LocalSave, ModelSave


@pytest.fixture
def mock_params():
    """create a mock params dictionary"""
    return {
        "dataset": "dataset",
        "scenario": "scenario",
        "create_datetime": "20220101_012345",
        "model_runs": 1024,
        "submitted_by": "username",
        "start_year": 2018,
        "end_year": 2020,
        "app_version": "1.2",
    }


def _mock_model_save_helper(mock_params, model_save_type):
    """helper to create a mock model save instance"""
    with patch.object(model_save_type, "__init__", lambda s, p, r: None):
        mdl = model_save_type(None, None)
    mdl._dataset = "dataset"
    mdl._scenario = "scenario"
    mdl._create_datetime = "20220101_012345"
    mdl._run_id = "dataset__scenario__20220101_012345"
    mdl._model_runs = 2
    mdl._params = mock_params
    mdl._model = None
    mdl._base_results_path = "base_results_path"
    mdl._results_path = "results_path"
    mdl._temp_path = "temp_path"
    mdl._ar_path = "temp_path/aggregated_results"
    mdl._cf_path = "temp_path/change_factors"
    mdl._item_base = {
        "id": mdl._run_id,
        "dataset": mdl._dataset,
        "scenario": mdl._scenario,
        "create_datetime": mdl._create_datetime,
        "model_runs": mdl._model_runs,
        "submitted_by": "username",
        "start_year": 2018,
        "end_year": 2020,
    }
    mdl._save_results = True
    return mdl


@pytest.fixture
def mock_model_save(mock_params):
    """create a mock ModelSave instance"""
    return _mock_model_save_helper(mock_params, ModelSave)


@pytest.fixture
def mock_local_model_save(mock_params):
    """create a mock LocalSave instance"""
    return _mock_model_save_helper(mock_params, LocalSave)


@pytest.fixture
def mock_cosmos_model_save(mock_params):
    """create a mock CosmsosDBSave instance"""
    mdl = _mock_model_save_helper(mock_params, CosmosDBSave)
    mdl._database = "db"
    mdl._cosmos_endpoint = "endpoint"
    mdl._cosmos_key = "key"
    return mdl


@pytest.fixture
def mock_change_factors():
    """create a mock change factors dataframe"""
    return pd.DataFrame({"change_factor": ["a"]})


def test_model_save_init(mocker, mock_params):
    """test that init set's everything up"""
    # arrange
    mocker.patch("os.path.join", wraps=lambda *args: "/".join(args))
    makedirs_mock = mocker.patch("os.makedirs")
    mkdtemp_mock = mocker.patch("model.model_save.mkdtemp", return_value="mkdtemp")
    # act
    m = ModelSave(mock_params, "results_path", "temp_path", "save_results")
    # assert
    assert m._dataset == "dataset"
    assert m._scenario == "scenario"
    assert m._create_datetime == "20220101_012345"
    assert m._run_id == "dataset__scenario__20220101_012345"
    assert m._model_runs == 1024
    assert m._params == mock_params
    assert m._model == None
    assert m._base_results_path == "results_path"
    assert (
        m._results_path
        == "dataset=dataset/scenario=scenario/create_datetime=20220101_012345"
    )
    assert m._temp_path == "temp_path"
    assert m._ar_path == "temp_path/aggregated_results"
    assert m._cf_path == "temp_path/change_factors"
    assert m._item_base == {
        "id": m._run_id,
        "dataset": m._dataset,
        "scenario": m._scenario,
        "create_datetime": m._create_datetime,
        "model_runs": m._model_runs,
        "submitted_by": "username",
        "start_year": 2018,
        "end_year": 2020,
        "app_version": "1.2",
    }
    assert m._save_results == "save_results"
    assert makedirs_mock.call_count == 2
    assert makedirs_mock.call_args_list[0] == call(m._ar_path, exist_ok=True)
    assert makedirs_mock.call_args_list[1] == call(m._cf_path, exist_ok=True)
    # test that it also handles creating a temporary dir if one isn't passed
    m = ModelSave(mock_params, "results_path")
    assert m._temp_path == "mkdtemp"
    mkdtemp_mock.assert_called_once()


def test_set_model(mock_model_save):
    """it should update self._model"""
    mock_model_save.set_model("model")
    assert mock_model_save._model == "model"


def test_run_model_baseline(mocker, mock_model_save):
    """it should just aggregate the baseline data"""
    model = Mock()
    model.model_type = "ip"
    model.aggregate.return_value = "aggregated_results"
    mock_model_save._model = model
    dill_mock = mocker.patch("dill.dump")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_model_save.run_model(-1)
        mock_file.assert_called_with("temp_path/aggregated_results/ip_-1.dill", "wb")
        dill_mock.assert_called_once()
        assert dill_mock.call_args_list[0][0][0] == "aggregated_results"


def test_run_model_dont_save_results(mocker, mock_model_save, mock_change_factors):
    """it should run the model and aggregate the results, but not save the full results"""  # arrange
    model = Mock()
    model.model_type = "ip"
    model.run.return_value = (mock_change_factors, "results")
    model.aggregate.return_value = "aggregated_results"
    mock_model_save._model = model
    mock_model_save._save_results = False
    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    dill_mock = mocker.patch("dill.dump")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_model_save.run_model(0)
        model.save_results.assert_not_called()
        to_parquet_mock.assert_called_once()
        to_parquet_mock.call_args_list[0][0][
            0
        ] == "temp_path/change_factors/ip_0.parquet"
        mock_file.assert_called_with("temp_path/aggregated_results/ip_0.dill", "wb")
        dill_mock.assert_called_once()
        assert dill_mock.call_args_list[0][0][0] == "aggregated_results"


def test_run_model_save_results(mocker, mock_model_save, mock_change_factors):
    """it should run the model and aggregate the results, saving the full results"""  # arrange
    mocker.patch("os.path.join", wraps=lambda *args: "/".join(args))
    mocker.patch("os.makedirs")
    model = Mock()
    model.model_type = "ip"
    model.run.return_value = (mock_change_factors, "results")
    model.aggregate.return_value = "aggregated_results"
    mock_model_save._model = model
    mock_model_save._save_results = True
    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    dill_mock = mocker.patch("dill.dump")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_model_save.run_model(0)
        model.save_results.assert_called_once()
        assert model.save_results.call_args_list[0][0][0] == "results"
        assert (
            model.save_results.call_args_list[0][0][1]("ip")
            == "base_results_path/model_results/activity_type='ip'/results_path/model_run=0"
        )
        os.makedirs.assert_called_once_with(
            "base_results_path/model_results/activity_type='ip'/results_path/model_run=0",
            exist_ok=True,
        )
        to_parquet_mock.assert_called_once()
        to_parquet_mock.call_args_list[0][0][
            0
        ] == "temp_path/change_factors/ip_0.parquet"
        mock_file.assert_called_with("temp_path/aggregated_results/ip_0.dill", "wb")
        dill_mock.assert_called_once()
        assert dill_mock.call_args_list[0][0][0] == "aggregated_results"


def test_post_runs(mocker, mock_model_save):
    """it should save params and clean up temporary files"""
    # arrange
    makedirs_mock = mocker.patch("os.makedirs")
    json_mock = mocker.patch("json.dump")
    rmtree_mock = mocker.patch("shutil.rmtree")
    mock_model_save._model = Mock()
    mock_model_save._model.run_params = "run_params"
    mock_model_save._params = "params"
    with patch("builtins.open", mock_open()) as mock_file:
        mock_model_save  # act
        mock_model_save.post_runs()
        # assert
        assert makedirs_mock.call_count == 2
        assert makedirs_mock.call_args_list[0] == call(
            "base_results_path/params", exist_ok=True
        )
        assert makedirs_mock.call_args_list[1] == call(
            "base_results_path/run_params", exist_ok=True
        )
        #
        assert json_mock.call_count == 2
        assert json_mock.call_args_list[0][0][0] == "params"
        assert json_mock.call_args_list[1][0][0] == "run_params"
        #
        mock_file.call_args_list[0][0] == call(
            "base_results_path/params/dataset__scenario__20220101_012345.json", "w"
        )
        mock_file.call_args_list[1][0] == call(
            "base_results_path/run_params/dataset__scenario__20220101_012345.json", "w"
        )
        #
        rmtree_mock.assert_called_once_with("temp_path")


def test_combine_aggregated_results(mocker, mock_model_save):
    """it should combine all of the saved aggregated results"""
    dill_load_mock = mocker.patch("dill.load", return_value=None)
    expected_flipped_results = {
        "default": [
            {"pod": "aae_abc", "values": [1, 2, 3]},
            {"pod": "ip_abc", "values": [4, 5, 6]},
            {"pod": "op_abc", "values": [7, 8, 9]},
        ],
        "other": [
            {"pod": "ip_abc", "values": [6, 5, 4]},
            {"pod": "op_abc", "values": [9, 8, 7]},
        ],
    }
    mock_model_save._flip_results = Mock(return_value=expected_flipped_results)
    mock_model_save._model = Mock()
    mock_model_save._model.run_params = {"variant": "variants"}
    mock_model_save._item_base = {"item_base": None}

    with patch("builtins.open", mock_open()) as mock_file:
        actual = mock_model_save._combine_aggregated_results()
        mock_file.call_args_list[0][0] == call(
            "temp_path/aggregated_results/aae_-1.json", "rb"
        )
        assert dill_load_mock.call_count == (mock_model_save._model_runs + 2) * 3
        mock_model_save._flip_results.assert_called_once_with(
            {k: [None] * 4 for k in ["aae", "ip", "op"]}
        )
        assert actual == {
            "item_base": None,
            "available_aggregations": {
                "aae": ["default"],
                "ip": ["default", "other"],
                "op": ["default", "other"],
            },
            "selected_variants": "variants",
            "results": expected_flipped_results,
        }


def test_flip_result():
    result = namedtuple("result", ["pod", "measure"])
    results = {
        "aae": [
            {"default": {result("aae_abc", "attendances"): i}}
            for i in [1, 2] + list(range(1, 101))
        ],
        "ip": [
            {
                k: {result("ip_abc", "attendances"): i}
                for k in ["default", "bed_occupancy"]
            }
            for i in [1, 2] + list(range(1, 101))
        ],
        "op": [
            {k: {result("op_abc", "attendances"): i} for k in ["default", "other"]}
            for i in [1, 2] + list(range(1, 101))
        ],
        "op_conversion": [
            {k: {result("op_abc", "attendances"): i} for k in ["default", "other"]}
            for i in [1, 2] + list(range(1, 101))
        ],
    }
    actual = ModelSave._flip_results(results)
    expected = {
        "default": [
            {
                "pod": "aae_abc",
                "measure": "attendances",
                "baseline": 1,
                "principal": 2,
                "median": 50.5,
                "lwr_ci": 5.95,
                "upr_ci": 95.05,
                "model_runs": list(range(1, 101)),
            },
            {
                "pod": "ip_abc",
                "measure": "attendances",
                "baseline": 1,
                "principal": 2,
                "median": 50.5,
                "lwr_ci": 5.95,
                "upr_ci": 95.05,
                "model_runs": list(range(1, 101)),
            },
            {
                "pod": "op_abc",
                "measure": "attendances",
                "baseline": 2,
                "principal": 4,
                "median": 101.0,
                "lwr_ci": 11.9,
                "upr_ci": 190.1,
                "model_runs": list(range(2, 202, 2)),
            },
        ],
        "bed_occupancy": [
            {
                "pod": "ip_abc",
                "measure": "attendances",
                "baseline": 1,
                "principal": 2,
                "median": 50.5,
                "lwr_ci": 5.95,
                "upr_ci": 95.05,
                "model_runs": list(range(1, 101)),
            }
        ],
        "other": [
            {
                "pod": "op_abc",
                "measure": "attendances",
                "baseline": 2,
                "principal": 4,
                "median": 101.0,
                "lwr_ci": 11.9,
                "upr_ci": 190.1,
            },
        ],
    }
    assert actual == expected


def test_combine_change_factors(mocker, mock_model_save):
    """test that it finds all of the files, reads them all, then combines them"""
    # arrange
    m = Mock()
    pq_mock = mocker.patch("pyarrow.parquet.ParquetDataset", return_value=m)
    m.read_pandas().to_pandas.return_value = "change factors"
    # act
    results = mock_model_save._combine_change_factors()
    # assert
    pq_mock.assert_called_once_with(mock_model_save._cf_path, use_legacy_dataset=False)
    assert results == "change factors"


def test_local_init(mocker):
    """it should call the super init method"""
    mocker.patch("model.model_save.super")
    LocalSave(1, 2, 3)
    # no asserts to perform, so long as this method doesn't fail


def test_local_post_runs(mocker, mock_local_model_save):
    """test that it correctly runs the post runs"""
    mdl = mock_local_model_save
    makedirs_mock = mocker.patch("os.makedirs")
    json_mock = mocker.patch("json.dump")
    mdl._combine_aggregated_results = Mock(return_value="combined_aggregated_results")
    mdl._combine_change_factors = Mock()
    super_mock = mocker.patch("model.model_save.super")
    with patch("builtins.open", mock_open()) as mock_file:
        mdl.post_runs()

        mdl._combine_aggregated_results.assert_called_once()
        json_mock.assert_called_once()
        assert json_mock.call_args_list[0][0][0] == "combined_aggregated_results"

        mock_file.assert_called_with(
            f"base_results_path/aggregated_results/{mdl._run_id}.json",
            "w",
            encoding="UTF-8",
        )

        assert makedirs_mock.call_count == 2
        assert makedirs_mock.call_args_list[0] == call(
            "base_results_path/aggregated_results", exist_ok=True
        )
        assert makedirs_mock.call_args_list[1] == call(
            "base_results_path/change_factors", exist_ok=True
        )

        mdl._combine_change_factors.assert_called_once()
        mdl._combine_change_factors().to_csv.assert_called_once_with(
            f"base_results_path/change_factors/{mdl._run_id}.csv", index=False
        )

        assert super_mock().post_runs.called_once()


def test_cosmos_init(mocker):
    """it should call the super init method"""
    # arrange
    mocker.patch("model.model_save.super")
    dotenv_mock = mocker.patch("model.model_save.load_dotenv")
    mocker.patch("os.getenv", wraps=lambda x: x)
    # act
    mdl = CosmosDBSave(1, 2, 3)
    # assert
    dotenv_mock.assert_called_once()
    assert mdl._database == "COSMOS_DB"
    assert mdl._cosmos_endpoint == "COSMOS_ENDPOINT"
    assert mdl._cosmos_key == "COSMOS_KEY"


def test_cosmos_post_runs(mocker, mock_cosmos_model_save):
    """test that it calls the cosmos functions and the base post runs method"""
    # arrange
    mdl = mock_cosmos_model_save
    mdl._upload_results = Mock()
    mdl._upload_change_factors = Mock()
    super_mock = mocker.patch("model.model_save.super")
    # act
    mdl.post_runs()
    # assert
    mdl._upload_results.assert_called_once()
    mdl._upload_change_factors.assert_called_once()
    super_mock().post_runs.assert_called_once()


def test_cosmos_upload_results(mock_cosmos_model_save):
    """it should upload the results to cosmos"""
    # arrange
    mdl = mock_cosmos_model_save
    mdl._combine_aggregated_results = Mock(return_value={"aggregated_results": None})
    mdl._get_database_container = Mock()
    # act
    mdl._upload_results()
    # assert
    mdl._combine_aggregated_results.assert_called_once()
    mdl._get_database_container.assert_called_once_with("results")
    mdl._get_database_container().upsert_item.assert_called_once_with(
        {"id": mdl._run_id, "aggregated_results": None}, partition_key=mdl._run_id
    )


def test_cosmos_change_factor_to_dict(mock_cosmos_model_save):
    """it should convert the change factor into a dict"""
    # arrange
    change_factor = pd.DataFrame(
        {
            "change_factor": ["baseline", "a", "b", "c", "c"] * 3,
            "strategy": ["-", "-", "-", "d", "e"] * 3,
            "measure": ["a"] * 15,
            "activity_type": ["aae"] * 15,
            "model_run": sum([[i] * 5 for i in range(3)], []),
            "value": sum(
                [[0] + [j + i * 4 + 1 for j in range(4)] for i in range(3)], []
            ),
        }
    )
    expected = [
        {"change_factor": "a", "principal": 1, "value": [5, 9]},
        {"change_factor": "b", "principal": 2, "value": [6, 10]},
        {"change_factor": "baseline", "baseline": 0},
        {"change_factor": "c", "strategy": "d", "principal": 3, "value": [7, 11]},
        {"change_factor": "c", "strategy": "e", "principal": 4, "value": [8, 12]},
    ]
    # act
    actual = mock_cosmos_model_save._change_factor_to_dict(change_factor)
    # assert
    assert actual == expected


def test_cosmos_upload_change_factors(mock_cosmos_model_save):
    """it should upload the change factors to cosmos"""
    # arrange
    mdl = mock_cosmos_model_save
    mdl._combine_change_factors = Mock(
        return_value=pd.DataFrame(
            {
                "change_factor": ["baseline", "a", "b", "c", "c"] * 3,
                "strategy": ["-", "-", "-", "d", "e"] * 3,
                "measure": ["a"] * 15,
                "activity_type": ["aae"] * 15,
                "model_run": sum([[i] * 5 for i in range(3)], []),
                "value": sum(
                    [[0] + [j + i * 4 + 1 for j in range(4)] for i in range(3)], []
                ),
            }
        )
    )
    mdl._change_factor_to_dict = Mock(return_value=None)
    mdl._get_database_container = Mock()
    expected_item = {
        **mdl._item_base,
        "aae": [{"measure": "a", "change_factors": None}],
    }
    # act
    mdl._upload_change_factors()
    # assert
    mdl._combine_change_factors.assert_called_once()
    mdl._get_database_container.assert_called_once_with("change_factors")
    mdl._get_database_container().upsert_item.assert_called_once_with(
        expected_item, partition_key=mdl._run_id
    )


def test_cosmos_get_database_container(mocker, mock_cosmos_model_save):
    """it should return a Cosmos Container Client"""
    # arrange
    gdc_mock = Mock()
    cosmos_client_mock = mocker.patch("model.model_save.CosmosClient")
    cosmos_client_mock().get_database_client.return_value = gdc_mock
    gdc_mock.get_container_client.return_value = "the container"
    # act
    actual = mock_cosmos_model_save._get_database_container("container")
    # assert
    assert actual == "the container"
    assert cosmos_client_mock.call_args_list[1] == call("endpoint", "key")
    cosmos_client_mock().get_database_client.assert_called_once_with("db")
    gdc_mock.get_container_client.assert_called_once_with("container")
