"""test a&e model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.aae import AaEModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(AaEModel, "__init__", lambda s, p, d: None):
        mdl = AaEModel(None, None)
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
        "bed_occupancy": {
            "a": {"a": [0.4, 0.6], "b": 0.7},
            "b": {"a": [0.4, 0.6], "b": 0.8},
        },
    }
    mdl._data_path = "data/synthetic"
    # create a mock object for the hsa gams
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    mdl._hsa_gams = {(i, j): hsa_mock for i in ["aae_a_a", "aae_b_b"] for j in [1, 2]}
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
    super_mock = mocker.patch("model.aae.super")
    ubd_mock = mocker.patch("model.aae.AaEModel._update_baseline_data")
    gdc_mock = mocker.patch("model.aae.AaEModel._get_data_counts", return_value=1)
    lst_mock = mocker.patch("model.aae.AaEModel._load_strategies")
    # act
    mdl = AaEModel("params", "data_path")
    # assert
    super_mock.assert_called_once()
    ubd_mock.assert_called_once()
    gdc_mock.assert_called_once_with(None)
    lst_mock.assert_called_once()
    assert mdl._baseline_counts == 1


def test_update_baseline_data(mock_model):
    # arrange
    mdl = mock_model
    mdl.data["is_ambulance"] = [True] * 10 + [False] * 10
    # act
    mdl._update_baseline_data()
    # assert
    assert mdl.data["group"].to_list() == ["ambulance"] * 10 + ["walk-in"] * 10
    assert mdl.data["tretspef"].to_list() == ["Other"] * 20


def test_get_data_counts(mock_model):
    # arrange
    mdl = mock_model
    data = mdl.data
    data["arrivals"] = list(range(1, 21))
    # act
    actual = mdl._get_data_counts(data)
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
    # act
    mdl._load_strategies()
    # assert
    assert mdl._strategies["activity_avoidance"]["strategy"].to_list() == [
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
    ]
    assert mdl._strategies["activity_avoidance"]["sample_rate"].to_list() == [1] * 12


def test_apply_resampling(mocker, mock_model):
    # arrange
    row_samples = np.array([[1, 2, 3, 4]])
    gdc_mock = mocker.patch("model.aae.AaEModel._get_data_counts", return_value=1)
    # act
    data, counts = mock_model._apply_resampling(row_samples, pd.DataFrame())
    # assert
    assert data["arrivals"].to_list() == [1, 2, 3, 4]
    assert counts == 1
    gdc_mock.assert_called_once()


def test_run(mocker, mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    mdl._baseline_counts = 1

    rr_mock = Mock()
    m = mocker.patch("model.aae.ActivityAvoidance", return_value=rr_mock)
    rr_mock.demographic_adjustment.return_value = rr_mock
    rr_mock.health_status_adjustment.return_value = rr_mock
    rr_mock.expat_adjustment.return_value = rr_mock
    rr_mock.repat_adjustment.return_value = rr_mock
    rr_mock.baseline_adjustment.return_value = rr_mock
    rr_mock.activity_avoidance.return_value = rr_mock
    rr_mock.apply_resampling.return_value = rr_mock

    # act
    mdl._run("model_run")

    # assert
    m.assert_called_once_with("model_run", 1)
    rr_mock.demographic_adjustment.assert_called_once()
    rr_mock.health_status_adjustment.assert_called_once()
    rr_mock.expat_adjustment.assert_called_once()
    rr_mock.repat_adjustment.assert_called_once()
    rr_mock.baseline_adjustment.assert_called_once()
    rr_mock.activity_avoidance.assert_called_once()
    rr_mock.apply_resampling.assert_called_once()


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""
    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame(
        {
            "sitetret": ["trust"] * 4,
            "age_group": [1] * 4,
            "sex": [1] * 4,
            "aedepttype": ["01", "01", "02", "02"],
            "is_ambulance": [True, False, True, False],
            "value": [1, 2, 3, 4],
        }
    )
    # act
    results = mdl.aggregate(mr_mock)
    # assert
    assert mdl._create_agg.call_count == 2
    assert results == {"default": 1, "sex+age_group": 1}
    #
    mr_call = pd.DataFrame(
        {
            "pod": ["aae_type-01", "aae_type-01", "aae_type-02", "aae_type-02"],
            "sitetret": ["trust"] * 4,
            "measure": ["ambulance", "walk-in"] * 2,
            "sex": [1] * 4,
            "age_group": [1] * 4,
            "value": [1, 2, 3, 4],
        }
    )
    assert mdl._create_agg.call_args_list[0][0][0].equals(mr_call)
    assert mdl._create_agg.call_args_list[1][0][0].equals(mr_call)


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame({"rn": [0], "arrivals": [1]})

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    mock_model.save_results(mr_mock, path_fn)
    to_parquet_mock.assert_called_once_with("aae/0.parquet")
