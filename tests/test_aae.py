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


def test_run_poisson_step(mock_model):
    """test that it applies the poisson step and mutates the data and step counts objects"""
    rng = Mock()
    rng.poisson = Mock(wraps=lambda xs: [2 * x for x in xs])
    data = pd.DataFrame({"arrivals": [1, 2, 3, 4, 5]})
    step_counts = {"baseline": 15}
    mock_model._run_poisson_step(rng, data, "test", [0, 1, 2, 3, 4], step_counts)
    assert data.arrivals.to_list() == [4, 12, 24, 40]
    assert step_counts["test"] == 65
    rng.poisson.assert_called_once()


def test_run_binomial_step(mock_model):
    """test that it applies the poisson step and mutates the data and step counts objects"""
    rng = Mock()
    rng.binomial = Mock(wraps=lambda xs, ys: [2 * (x - y) for x, y in zip(xs, ys)])
    data = pd.DataFrame({"arrivals": [5, 6, 7, 8, 9]})
    step_counts = {"baseline": 15}
    mock_model._run_binomial_step(rng, data, "test", [5, 4, 3, 2, 1], step_counts)
    assert data.arrivals.to_list() == [4, 8, 12, 16]
    assert step_counts["test"] == 25
    rng.binomial.assert_called_once()


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    mdl._run_poisson_step = Mock()
    mdl._run_binomial_step = Mock()
    mdl._low_cost_discharged = Mock()
    mdl._left_before_seen = Mock()
    mdl._frequent_attenders = Mock()
    data = pd.DataFrame({"rn": [1, 2], "hsagrp": [3, 4], "arrivals": [5, 6]})
    run_params = {"aae_factors": "aae_factors"}
    # act
    change_factors, model_results = mdl._run(
        None, data, run_params, pd.Series({1: 1, 2: 2}), "hsa_f"
    )
    # assert
    assert change_factors.equals(
        pd.DataFrame(
            {
                "change_factor": ["baseline"],
                "strategy": ["-"],
                "measure": ["arrivals"],
                "value": np.array([11]),
            }
        )
    )
    assert model_results.equals(data.drop("hsagrp", axis="columns"))
    assert mdl._run_poisson_step.call_count == 2
    assert mdl._run_binomial_step.call_count == 3
    mdl._low_cost_discharged.assert_called_once()
    mdl._left_before_seen.assert_called_once()
    mdl._frequent_attenders.assert_called_once()


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    model_results = pd.DataFrame(
        {
            "aedepttype": ["01", "01", "02", "02"],
            "aearrivalmode": ["1", "2", "1", "2"],
            "value": [1, 2, 3, 4],
        }
    )
    results = mdl.aggregate(model_results, 1)
    assert mdl._create_agg.call_count == 2
    assert results == {"default": 1, "sex+age_group": 1}
    #
    mr_call = pd.DataFrame(
        {
            "aedepttype": ["01", "01", "02", "02"],
            "aearrivalmode": ["1", "2", "1", "2"],
            "value": [1, 2, 3, 4],
            "pod": ["aae_type-01", "aae_type-01", "aae_type-02", "aae_type-02"],
            "measure": ["ambulance", "walk-in"] * 2,
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
