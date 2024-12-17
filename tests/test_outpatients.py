"""test outpatients model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.outpatients import OutpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(OutpatientsModel, "__init__", lambda s, p, d, h, r: None):
        mdl = OutpatientsModel(None, None, None, None)
    mdl._data_loader = Mock()
    mdl.model_type = "op"
    mdl.params = {
        "dataset": "synthetic",
        "model_runs": 3,
        "seed": 1,
        "demographic_factors": {
            "file": "demographics_file.csv",
            "variant_probabilities": {"a": 0.6, "b": 0.4},
        },
        "start_year": 2018,
        "end_year": 2020,
        "health_status_adjustment": [0.8, 1.0],
        "waiting_list_adjustment": "waiting_list_adjustment",
        "expat": {"op": {"Other": [0.7, 0.9]}},
        "repat_local": {"op": {"Other": [1.0, 1.2]}},
        "repat_nonlocal": {"op": {"Other": [1.3, 1.5]}},
        "non-demographic_adjustment": {
            "a": {"a_a": [1, 1.2], "a_b": [1, 1.2]},
            "b": {"b_a": [1, 1.2], "b_b": [1, 1.2]},
        },
        "inpatient_factors": {
            "admission_avoidance": {
                "a_a": {"interval": [0.4, 0.6]},
                "a_b": {"interval": [0.4, 0.6]},
            },
            "los_reduction": {
                "b_a": {"interval": [0.4, 0.6]},
                "b_b": {"interval": [0.4, 0.6]},
            },
        },
        "outpatient_factors": {
            "a": {"a_a": {"interval": [0.4, 0.6]}, "a_b": {"interval": [0.4, 0.6]}},
            "b": {"b_a": {"interval": [0.4, 0.6]}, "b_b": {"interval": [0.4, 0.6]}},
        },
        "op_factors": {
            "a": {"a_a": {"interval": [0.4, 0.6]}, "a_b": {"interval": [0.4, 0.6]}},
            "b": {"b_a": {"interval": [0.4, 0.6]}, "b_b": {"interval": [0.4, 0.6]}},
        },
    }
    mdl._data_path = "data/synthetic"
    # create a mock object for the hsa gams
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    mdl.hsa_gams = {(i, j): hsa_mock for i in ["op_a_a", "op_b_b"] for j in [1, 2]}
    # create a minimal data object for testing
    mdl.data = pd.DataFrame(
        {
            "rn": list(range(1, 21)),
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
            "hsagrp": [x for _ in range(1, 11) for x in ["op_a_a", "op_b_b"]],
        }
    )
    return mdl


# methods


def test_init_calls_super_init(mocker):
    """test that the model calls the super method"""
    # arrange
    super_mock = mocker.patch("model.outpatients.super")
    # act
    OutpatientsModel("params", "data_path", "hsa", "run_params")
    # assert
    super_mock.assert_called_once()


def test_get_data(mock_model):
    # arrange
    mdl = mock_model
    mdl._data_loader.get_op.return_value = "op data"

    # act
    actual = mdl._get_data()

    # assert
    assert actual == "op data"
    mdl._data_loader.get_op.assert_called_once_with()


def test_add_pod_to_data(mock_model):
    # arrange
    mock_model.data = pd.DataFrame(
        {
            "is_first": [True, True, False, False],
            "has_procedures": [True, False, True, False],
        }
    )
    # act
    mock_model._add_pod_to_data()

    # assert
    assert mock_model.data["pod"].to_list() == [
        "op_procedure",
        "op_first",
        "op_procedure",
        "op_follow-up",
    ]


def test_get_data_counts(mock_model):
    # arrange
    mdl = mock_model
    data = mdl.data
    data["attendances"] = list(range(1, 21))
    data["tele_attendances"] = list(range(21, 41))
    # act
    actual = mdl.get_data_counts(data)
    # assert
    assert actual.tolist() == [
        [float(i) for i in range(1, 21)],
        [float(i) for i in range(21, 41)],
    ]


def test_load_strategies(mock_model):
    # arrange
    mdl = mock_model
    mdl.data["has_procedures"] = [True] * 10 + [False] * 10
    mdl.data["is_first"] = ([True] * 5 + [False] * 5) * 2
    mdl.data["is_cons_cons_ref"] = [True] * 10 + [False] * 10
    mdl.data["type"] = ["a", "b", "c", "d", "e"] * 4
    mdl.data["is_gp_ref"] = [False] * 10 + [True] * 10
    # act
    mdl._load_strategies()
    # assert
    assert mdl.strategies["activity_avoidance"]["strategy"].to_list() == [
        f"{i}_{j}"
        for i in ["followup_reduction"]
        + ["consultant_to_consultant_reduction"] * 2
        + ["gp_referred_first_attendance_reduction"]
        for j in ["a", "b", "c", "d", "e"]
    ]
    assert mdl.strategies["activity_avoidance"]["sample_rate"].to_list() == [1] * 20


def test_convert_to_tele(mock_model):
    """test that it mutates the data"""
    # arrange
    mdl = mock_model

    mr_mock = Mock()
    mr_mock.rng.binomial.return_value = np.array(
        [10, 15, 0, 20, 25, 0, 30, 35, 0, 40, 45, 0]
    )
    data = pd.DataFrame(
        {
            "rn": range(12),
            "pod": (["a"] * 3 + ["b"] * 3) * 2,
            "sitetret": ["c"] * 6 + ["d"] * 6,
            "has_procedures": [False, False, True] * 4,
            "type": ["a", "b", "a"] * 4,
            "attendances": [20, 25, 30] * 4,
            "tele_attendances": [5, 10, 0] * 4,
        }
    )
    mr_mock.run_params = {
        "efficiencies": {"op": {"convert_to_tele_a": 0.25, "convert_to_tele_b": 0.5}}
    }
    mr_mock.model.strategies = {
        "efficiencies": pd.Series(
            {
                k: "convert_to_tele_a" if k % 2 else "convert_to_tele_b"
                for k in data["rn"]
            }
        )
    }

    # act
    actual_data, actual_step_counts = mdl._convert_to_tele(data.copy(), mr_mock)

    # assert
    assert actual_data["attendances"].to_list() == [
        10,
        10,
        30,
        0,
        0,
        30,
        -10,
        -10,
        30,
        -20,
        -20,
        30,
    ]
    assert actual_data["tele_attendances"].to_list() == [
        15,
        25,
        0,
        25,
        35,
        0,
        35,
        45,
        0,
        45,
        55,
        0,
    ]

    assert mr_mock.rng.binomial.call_args == call(
        data["attendances"].to_list(), [i for _ in range(6) for i in [0.5, 0.75]]
    )

    assert actual_step_counts.to_dict("list") == {
        "pod": ["a", "a", "b", "b"],
        "sitetret": ["c", "d", "c", "d"],
        "change_factor": [
            "efficiencies",
            "efficiencies",
            "efficiencies",
            "efficiencies",
        ],
        "strategy": [
            "convert_to_tele",
            "convert_to_tele",
            "convert_to_tele",
            "convert_to_tele",
        ],
        "attendances": [-25, -65, -45, -85],
        "tele_attendances": [25, 65, 45, 85],
    }


def test_apply_resampling(mocker, mock_model):
    # arrange
    row_samples = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # act
    data = mock_model.apply_resampling(row_samples, pd.DataFrame())
    # assert
    assert data["attendances"].to_list() == [1, 2, 3, 4]
    assert data["tele_attendances"].to_list() == [5, 6, 7, 8]


def test_efficiencies(mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    data = pd.DataFrame({"x": [1]})

    mdl._convert_to_tele = Mock(return_value=("data", "step_counts"))

    # act
    actual = mdl.efficiencies(data, "model_run")

    # assert
    assert actual == ("data", "step_counts")

    mdl._convert_to_tele.assert_called_once()
    assert mdl._convert_to_tele.call_args[0][0].to_dict("list") == {"x": [1]}
    assert mdl._convert_to_tele.call_args[0][1] == "model_run"


def test_process_data(mock_model):
    # arrange
    df = pd.DataFrame(
        {
            "sitetret": ["trust"] * 4,
            "is_first": [True, True, False, False],
            "has_procedures": [False, True, False, True],
            "tretspef": [1, 1, 1, 1],
            "tretspef_raw": [1, 1, 1, 1],
            "rn": [1, 2, 3, 4],
            "attendances": [5, 6, 7, 8],
            "tele_attendances": [9, 10, 11, 12],
            "age": [1, 1, 1, 1],
            "age_group": [1, 1, 1, 1],
            "sex": [1, 1, 1, 1],
            "pod": ["op_first", "op_procedure", "op_follow-up", "op_procedure"],
        }
    )
    expected = {
        "pod": [
            k for k in ["op_first", "op_follow-up", "op_procedure"] for _ in [0, 1]
        ],
        "sitetret": ["trust"] * 6,
        "measure": ["attendances", "tele_attendances"] * 3,
        "sex": [1] * 6,
        "age": [1] * 6,
        "age_group": [1] * 6,
        "tretspef": [1] * 6,
        "tretspef_raw": [1] * 6,
        "value": [5, 9, 7, 11, 14, 22],
    }
    # act
    actual = mock_model.process_data(df)
    # assert
    assert actual.to_dict("list") == expected


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: model_results.to_dict(orient="list")}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl.process_data = Mock(return_value="processed_data")

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = "model_results"

    # act
    actual_mr, actual_aggs = mdl.aggregate(mr_mock)

    # assert
    mdl.process_data.assert_called_once_with("model_results")
    assert actual_mr == "processed_data"
    assert actual_aggs == [
        ["sex", "tretspef"],
        ["tretspef_raw"],
    ]


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame(
        {"rn": [0], "attendances": [1], "tele_attendances": [2]}
    )
    mr_mock.avoided_activity = pd.DataFrame(
        {"rn": [0], "attendances": [1], "tele_attendances": [0]}
    )

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    mock_model.save_results(mr_mock, path_fn)
    assert to_parquet_mock.call_args_list == [
        call("op/0.parquet"),
        call("op_avoided/0.parquet"),
    ]


def test_calculate_avoided_activity(mock_model):
    # arrange
    data = pd.DataFrame({"rn": [0], "attendances": [4], "tele_attendances": [3]})
    data_resampled = pd.DataFrame(
        {"rn": [0], "attendances": [2], "tele_attendances": [1]}
    )
    # act
    actual = mock_model.calculate_avoided_activity(data, data_resampled)
    # assert
    assert actual.to_dict(orient="list") == {
        "rn": [0],
        "attendances": [2],
        "tele_attendances": [2],
    }
