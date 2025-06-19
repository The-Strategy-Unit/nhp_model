"""test a&e model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring,unnecessary-lambda-assignment

from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from nhp.model.aae import AaEModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(AaEModel, "__init__", lambda s, p, d, h, r: None):
        mdl = AaEModel(None, None, None, None)
    mdl._data_loader = Mock()
    mdl.model_type = "aae"
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
        "expat": {
            "aae": {"ambulance": [0.7, 0.9]},
            "repat_local": {"aae": {"ambulance": [1.0, 1.2]}},
            "repat_nonlocal": {"aae": {"ambulance": [1.3, 1.5]}},
        },
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
        "aae_factors": {
            "a": {"a_a": {"interval": [0.4, 0.6]}, "a_b": {"interval": [0.4, 0.6]}},
            "b": {"b_a": {"interval": [0.4, 0.6]}, "b_b": {"interval": [0.4, 0.6]}},
        },
    }
    mdl._data_path = "data/synthetic"
    # create a mock object for the hsa gams
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    mdl.hsa_gams = {(i, j): hsa_mock for i in ["aae_a_a", "aae_b_b"] for j in [1, 2]}
    # create a minimal data object for testing
    mdl.data = pd.DataFrame(
        {
            "rn": list(range(1, 21)),
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
            "hsagrp": [x for _ in range(1, 11) for x in ["aae_a_a", "aae_b_b"]],
        }
    )
    return mdl


# methods


def test_init_calls_super_init(mocker):
    """test that the model calls the super method"""
    # arrange
    super_mock = mocker.patch("nhp.model.aae.super")
    # act
    AaEModel("params", "data_path", "hsa", "run_params")
    # assert
    super_mock.assert_called_once()


def test_get_data(mock_model):
    # arrange
    mdl = mock_model
    mdl._data_loader.get_aae.return_value = "aae data"

    # act
    actual = mdl._get_data()

    # assert
    assert actual == "aae data"
    mdl._data_loader.get_aae.assert_called_once_with()


def test_add_pod_to_data(mock_model):
    # arrange
    mock_model.data = pd.DataFrame({"aedepttype": ["01", "02", "03", "04"]})
    # act
    mock_model._add_pod_to_data()

    # assert
    assert mock_model.data["pod"].to_list() == [
        "aae_type-01",
        "aae_type-02",
        "aae_type-03",
        "aae_type-04",
    ]


def test_get_data_counts(mock_model):
    # arrange
    mdl = mock_model
    data = mdl.data
    data["arrivals"] = list(range(1, 21))
    # act
    actual = mdl.get_data_counts(data)
    # assert
    assert actual.tolist() == [[float(i) for i in range(1, 21)]]


def test_load_strategies(mock_model):
    # arrange
    mdl = mock_model
    mdl.data["is_frequent_attender"] = [False] * 0 + [True] * 4 + [False] * 16
    mdl.data["is_left_before_treatment"] = [False] * 4 + [True] * 4 + [False] * 12
    mdl.data["is_low_cost_referred_or_discharged"] = (
        [False] * 12 + [True] * 4 + [False] * 4
    )
    mdl.data["is_discharged_no_treatment"] = [False] * 16 + [True] * 4
    # act
    mdl._load_strategies()
    # assert
    assert mdl.strategies["activity_avoidance"]["strategy"].to_list() == [
        "frequent_attenders_a_a",
        "frequent_attenders_b_b",
        "frequent_attenders_a_a",
        "frequent_attenders_b_b",
        "left_before_seen_a_a",
        "left_before_seen_b_b",
        "left_before_seen_a_a",
        "left_before_seen_b_b",
        "low_cost_discharged_a_a",
        "low_cost_discharged_b_b",
        "low_cost_discharged_a_a",
        "low_cost_discharged_b_b",
        "discharged_no_treatment_a_a",
        "discharged_no_treatment_b_b",
        "discharged_no_treatment_a_a",
        "discharged_no_treatment_b_b",
    ]
    assert mdl.strategies["activity_avoidance"]["sample_rate"].to_list() == [1] * 16


def test_apply_resampling(mocker, mock_model):
    # arrange
    row_samples = np.array([[1, 2, 3, 4]])
    # act
    data = mock_model.apply_resampling(row_samples, pd.DataFrame())
    # assert
    assert data["arrivals"].to_list() == [1, 2, 3, 4]


def test_efficiencies(mock_model):
    """test the efficiencies method (pass)"""
    # arrange

    # act
    actual = mock_model.efficiencies("data", None)

    # assert
    assert actual == ("data", None)


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: model_results.to_dict(orient="list")}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl.process_results = Mock(return_value="processed_data")

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = "model_results"

    # act
    actual_mr, actual_aggs = mdl.aggregate(mr_mock)

    # assert
    mdl.process_results.assert_called_once_with("model_results")
    assert actual_mr == "processed_data"
    assert actual_aggs == [
        ["acuity"],
        ["attendance_category"],
    ]


def test_process_results(mock_model):
    # arrange
    data = pd.DataFrame(
        {
            "sitetret": ["trust"] * 4,
            "acuity": ["a", "a", "b", "b"],
            "attendance_category": [1, 1, 2, 2],
            "age": [1, 2, 3, 4],
            "age_group": [1] * 4,
            "sex": [1] * 4,
            "pod": ["aae_type-01", "aae_type-01", "aae_type-02", "aae_type-02"],
            "is_ambulance": [True, False, True, False],
            "value": [1, 2, 3, 4],
        }
    )

    expected = {
        "pod": ["aae_type-01", "aae_type-01", "aae_type-02", "aae_type-02"],
        "sitetret": ["trust"] * 4,
        "acuity": ["a", "a", "b", "b"],
        "measure": ["ambulance", "walk-in"] * 2,
        "sex": [1] * 4,
        "age": [1, 2, 3, 4],
        "age_group": [1] * 4,
        "attendance_category": [1, 1, 2, 2],
        "value": [1, 2, 3, 4],
    }
    # act
    actual = mock_model.process_results(data)

    # assert
    assert actual.to_dict("list") == expected


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame({"rn": [0], "arrivals": [1]})
    mr_mock.avoided_activity = pd.DataFrame({"rn": [0], "arrivals": [1]})

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    mock_model.save_results(mr_mock, path_fn)
    assert to_parquet_mock.call_args_list[0] == call("aae/0.parquet")
    assert to_parquet_mock.call_args_list[1] == call("aae_avoided/0.parquet")


def test_calculate_avoided_activity(mock_model):
    # arrange
    data = pd.DataFrame({"rn": [0, 1], "arrivals": [4, 3]})
    data_resampled = pd.DataFrame({"rn": [0, 1], "arrivals": [2, 1]})
    # act
    actual = mock_model.calculate_avoided_activity(data, data_resampled)
    # assert
    assert actual.to_dict(orient="list") == {"rn": [0, 1], "arrivals": [2, 2]}
