"""test outpatients model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from model.outpatients import OutpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(OutpatientsModel, "__init__", lambda s, p, d: None):
        mdl = OutpatientsModel(None, None)
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
    mocker.patch("model.outpatients.super")
    OutpatientsModel("params", "data_path")
    # no asserts to perform, so long as this method doesn't fail


def test_waiting_list_adjutment(mock_model):
    """test that it returns the wla numpy array"""
    data = pd.DataFrame({"tretspef": [1] * 2 + [2] * 8 + [3] * 4 + [4] * 1})
    mdl = mock_model
    mdl.params["waiting_list_adjustment"] = {"op": {1: 1, 2: 2, 3: 3, 5: 1}}
    actual = mdl._waiting_list_adjustment(data)
    expected = [1.5] * 2 + [1.25] * 8 + [1.75] * 4 + [1]
    assert np.array_equal(actual, expected)


def test_followup_reduction(mock_model):
    """test that it calls factor helper"""
    mdl = mock_model
    mdl._factor_helper = Mock(return_value="followup reduction")
    assert (
        mdl._followup_reduction("data", {"followup_reduction": "fupr"})
        == "followup reduction"
    )
    mdl._factor_helper.assert_called_once_with(
        "data", "fupr", {"has_procedures": 0, "is_first": 0}
    )


def test_consultant_to_consultant_reduction(mock_model):
    """test that it calls factor helper"""
    mdl = mock_model
    mdl._factor_helper = Mock(return_value="c2c reduction")
    assert (
        mdl._consultant_to_consultant_reduction(
            "data", {"consultant_to_consultant_reduction": "c2c"}
        )
        == "c2c reduction"
    )
    mdl._factor_helper.assert_called_once_with("data", "c2c", {"is_cons_cons_ref": 1})


def test_convert_to_tele(mock_model):
    """test that it mutates the data"""
    rng = Mock()
    rng.binomial.return_value = 10
    data = pd.DataFrame(
        {
            "has_procedures": [False, False, True],
            "type": ["a", "b", "a"],
            "attendances": [20, 25, 30],
            "tele_attendances": [5, 10, 0],
        }
    )
    mdl = mock_model
    mdl._convert_to_tele(rng, data, {"convert_to_tele": {"a": 1, "b": 2}})
    assert data["attendances"].to_list() == [10, 15, 30]
    assert data["tele_attendances"].to_list() == [15, 20, 0]
    assert rng.binomial.called_once_with(pd.Series([20, 25]), [1, 2])


def test_run_poisson_step(mock_model):
    """test that it applies the poisson step and mutates the data and calls update step counts"""
    # arrange
    rng = Mock()
    rng.poisson = Mock(wraps=lambda xs: [2 * x for x in xs])
    data = pd.DataFrame({"attendances": [0, 1, 2, 3], "tele_attendances": [4, 0, 5, 6]})
    mdl = mock_model
    mdl._update_step_counts = Mock()
    # act
    mdl._run_poisson_step(rng, data, "test", [1, 2, 3, 0], "step_counts")
    # assert
    assert data.attendances.to_list() == [0, 4, 12]
    assert data.tele_attendances.to_list() == [8, 0, 30]
    assert rng.poisson.call_count == 2
    mdl._update_step_counts.assert_called_once_with(data, "test", "step_counts")


def test_run_binomial_step(mock_model):
    """test that it applies the binomial step and mutates the data and calls update step counts"""
    # arrange
    rng = Mock()
    rng.binomial = Mock(wraps=lambda xs, ys: [2 * (x - y) for x, y in zip(xs, ys)])
    data = pd.DataFrame({"attendances": [3, 6, 7, 1], "tele_attendances": [4, 5, 8, 1]})
    mdl = mock_model
    mdl._update_step_counts = Mock()
    # act
    mdl._run_binomial_step(rng, data, "test", [3, 5, 2, 1], "step_counts")
    # assert
    assert data.attendances.to_list() == [0, 2, 10]
    assert data.tele_attendances.to_list() == [2, 0, 12]
    assert rng.binomial.call_count == 2
    mdl._update_step_counts.assert_called_once_with(data, "test", "step_counts")


def test_update_step_counts(mock_model):
    """test that it updates the step counts object"""
    # arrange
    step_counts = {"baseline": {"attendances": 6, "tele_attendances": 15}}
    data = pd.DataFrame({"attendances": [2, 3, 4], "tele_attendances": [6, 7, 8]})
    # act
    mock_model._update_step_counts(data, "test", step_counts)
    # assert
    assert step_counts["test"] == {"attendances": 3, "tele_attendances": 6}


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    mdl._run_poisson_step = Mock()
    mdl._run_binomial_step = Mock()
    mdl._waiting_list_adjustment = Mock()
    mdl._followup_reduction = Mock()
    mdl._consultant_to_consultant_reduction = Mock()
    mdl._convert_to_tele = Mock()
    mdl._update_step_counts = Mock()
    mdl._expat_adjustment = Mock()
    mdl._repat_adjustment = Mock()
    data = pd.DataFrame(
        {
            "rn": [1, 2],
            "hsagrp": [3, 4],
            "attendances": [5, 6],
            "tele_attendances": [7, 8],
        }
    )
    run_params = {"outpatient_factors": "outpatient_factors"}
    # act
    change_factors, model_results = mdl._run(
        None, data, run_params, pd.Series({1: 1, 2: 2}), "hsa_f"
    )
    # assert
    assert change_factors.equals(
        pd.DataFrame(
            {
                "change_factor": ["baseline"] * 2,
                "strategy": ["-"] * 2,
                "measure": ["attendances", "tele_attendances"],
                "value": np.array([11, 15]),
            }
        )
    )
    assert model_results.equals(data.drop("hsagrp", axis="columns"))
    assert mdl._run_poisson_step.call_count == 4
    assert mdl._run_binomial_step.call_count == 3
    mdl._waiting_list_adjustment.assert_called_once()
    mdl._followup_reduction.assert_called_once()
    mdl._consultant_to_consultant_reduction.assert_called_once()
    mdl._convert_to_tele.assert_called_once()
    mdl._update_step_counts.assert_called_once()
    mdl._expat_adjustment.assert_called_once()
    mdl._repat_adjustment.assert_called_once()


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    model_results = pd.DataFrame(
        {
            "is_first": [True, True, False, False],
            "has_procedures": [False, True, False, True],
            "tretspef": [1, 1, 1, 1],
            "rn": [1, 2, 3, 4],
            "attendances": [5, 6, 7, 8],
            "tele_attendances": [9, 10, 11, 12],
            "age_group": [1, 1, 1, 1],
            "sex": [1, 1, 1, 1],
        }
    )
    results = mdl.aggregate(model_results, 1)
    assert mdl._create_agg.call_count == 3
    assert results == {"default": 1, "sex+age_group": 1, "sex+tretspef": 1}
    #
    mr_call = pd.DataFrame(
        {
            "pod": [
                k for k in ["op_first", "op_follow-up", "op_procedure"] for _ in [0, 1]
            ],
            "measure": ["attendances", "tele_attendances"] * 3,
            "sex": [1] * 6,
            "age_group": [1] * 6,
            "tretspef": [1] * 6,
            "value": [5, 9, 7, 11, 14, 22],
        }
    )
    assert mdl._create_agg.call_args_list[0][0][0].equals(mr_call)
    assert mdl._create_agg.call_args_list[1][0][0].equals(mr_call)
    assert mdl._create_agg.call_args_list[2][0][0].equals(mr_call)


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    results = pd.DataFrame({"rn": [0], "attendances": [1], "tele_attendances": [2]})
    mock_model.save_results(results, path_fn)
    to_parquet_mock.assert_called_once_with("op/0.parquet")
