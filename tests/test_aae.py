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
    mocker.patch("model.aae.super")
    AaEModel("params", "data_path")
    # no asserts to perform, so long as this method doesn't fail


def test_low_cost_discharged(mock_model):
    """test that it calls factor helper"""
    mdl = mock_model
    mdl._factor_helper = Mock(return_value="low cost discharged")
    assert (
        mdl._low_cost_discharged("data", {"low_cost_discharged": "lcd"})
        == "low cost discharged"
    )
    mdl._factor_helper.assert_called_once_with(
        "data", "lcd", {"is_low_cost_referred_or_discharged": 1}
    )


def test_left_before_seen(mock_model):
    """test that it calls factor helper"""
    mdl = mock_model
    mdl._factor_helper = Mock(return_value="left before seen")
    assert (
        mdl._left_before_seen("data", {"left_before_seen": "lbs"}) == "left before seen"
    )
    mdl._factor_helper.assert_called_once_with(
        "data", "lbs", {"is_left_before_treatment": 1}
    )


def test_frequent_attenders(mock_model):
    """test that is calls factor helper"""
    mdl = mock_model
    mdl._factor_helper = Mock(return_value="frequent attenders")
    assert (
        mdl._frequent_attenders("data", {"frequent_attenders": "fa"})
        == "frequent attenders"
    )
    mdl._factor_helper.assert_called_once_with(
        "data", "fa", {"is_frequent_attender": 1}
    )


def test_expat_adjustment():
    """ "test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame({"is_ambulance": [True, False]})
    run_params = {"expat": {"aae": {"ambulance": 0.8}}}
    # act
    actual = AaEModel._expat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [0.8, 1.0]


def test_repat_adjustment():
    """test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame(
        {
            "is_ambulance": [True, False] * 2,
            "is_main_icb": [i for i in [True, False] for _ in range(2)],
        }
    )
    run_params = {
        "repat_local": {"aae": {"ambulance": 1.1}},
        "repat_nonlocal": {"aae": {"ambulance": 1.3}},
    }
    # act
    actual = AaEModel._repat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [1.1, 1.0, 1.3, 1.0]


def test_baseline_adjustment():
    """test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame({"rn": list(range(2)), "is_ambulance": [True, False]})
    run_params = {"baseline_adjustment": {"aae": {"ambulance": 1, "walk-in": 2}}}
    # act
    actual = AaEModel._baseline_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [1, 2]


def test_step_counts(mock_model):
    """test that it estimates the step counts correctly"""
    # arrange
    mdl = mock_model
    arrivals = pd.Series([1, 2, 3])
    arrivals_after = 15
    factors = {"a": [0.5, 0.75, 0.875], "b": [1.0, 1.5, 2.0]}
    # act
    actual = mdl._step_counts(arrivals, arrivals_after, factors)
    # assert
    assert actual == {"baseline": 6, "a": -6.1875, "b": 15.1875}


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    rng = Mock()
    rng.poisson.return_value = np.array([7, 8])
    mdl._step_counts = Mock(return_value={"a": 1, "b": 2})
    mdl._low_cost_discharged = Mock(return_value=[5, 6])
    mdl._left_before_seen = Mock(return_value=[7, 8])
    mdl._frequent_attenders = Mock(return_value=[9, 10])
    mdl._expat_adjustment = Mock(return_value=pd.Series([11, 12]))
    mdl._repat_adjustment = Mock(return_value=pd.Series([13, 14]))
    mdl._baseline_adjustment = Mock(return_value=pd.Series([15, 16]))
    data = pd.DataFrame({"rn": [1, 2], "hsagrp": [3, 4], "arrivals": [5, 6]})
    run_params = {"aae_factors": "aae_factors"}
    # act
    change_factors, model_results = mdl._run(
        rng, data, run_params, pd.Series({1: 1, 2: 2}), [3, 4]
    )
    # assert
    assert change_factors.to_dict("list") == {
        "change_factor": ["a", "b"],
        "strategy": ["-"] * 2,
        "measure": ["arrivals"] * 2,
        "value": [1, 2],
    }
    assert model_results.to_dict("list") == {"rn": [1, 2], "arrivals": [7, 8]}
    rng.poisson.assert_called_once()
    assert rng.poisson.call_args_list[0][0][0].to_list() == [10135125, 61931520]
    mdl._low_cost_discharged.assert_called_once()
    mdl._left_before_seen.assert_called_once()
    mdl._frequent_attenders.assert_called_once()
    mdl._expat_adjustment.assert_called_once()
    mdl._repat_adjustment.assert_called_once()
    mdl._baseline_adjustment.assert_called_once()
    mdl._step_counts.assert_called_once()


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""
    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    model_results = pd.DataFrame(
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
    results = mdl.aggregate(model_results, 1)
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

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    results = pd.DataFrame({"rn": [0], "arrivals": [1]})
    mock_model.save_results(results, path_fn)
    to_parquet_mock.assert_called_once_with("aae/0.parquet")
