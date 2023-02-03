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
    # arrange
    mdl = mock_model
    rng = Mock()
    rng.binomial.return_value = np.array([10])
    data = pd.DataFrame(
        {
            "has_procedures": [False, False, True],
            "type": ["a", "b", "a"],
            "attendances": [20, 25, 30],
            "tele_attendances": [5, 10, 0],
        }
    )
    step_counts = {}
    # act
    mdl._convert_to_tele(rng, data, {"convert_to_tele": {"a": 1, "b": 2}}, step_counts)
    # assert
    assert data["attendances"].to_list() == [10, 15, 30]
    assert data["tele_attendances"].to_list() == [15, 20, 0]
    assert rng.binomial.called_once_with(pd.Series([20, 25]), [1, 2])
    assert step_counts == {
        "convert_to_tele": {"attendances": -10, "tele_attendances": 10}
    }


def test_expat_adjustment():
    """ "test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame({"tretspef": ["100", "120", "Other"]})
    run_params = {"expat": {"op": {"100": 0.8, "120": 0.9}}}
    # act
    actual = OutpatientsModel._expat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [0.8, 0.9, 1.0]


def test_repat_adjustment():
    """test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame(
        {
            "tretspef": ["100", "120", "Other"] * 2,
            "is_main_icb": [i for i in [True, False] for _ in range(3)],
        }
    )
    run_params = {
        "repat_local": {"op": {"100": 1.1, "120": 1.2}},
        "repat_nonlocal": {"op": {"100": 1.3, "120": 1.4}},
    }
    # act
    actual = OutpatientsModel._repat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [1.1, 1.2, 1.0, 1.3, 1.4, 1.0]


def test_baseline_adjustment():
    """test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame(
        {
            "rn": list(range(12)),
            "tretspef": ["100", "200", "Other"] * 4,
            "is_first": ([True] * 3 + [False] * 3) * 2,
            "has_procedures": [False] * 6 + [True] * 6,
        }
    )
    run_params = {
        "baseline_adjustment": {
            "op": {
                "first": {"100": 1, "200": 2, "Other": 3},
                "followup": {"100": 4, "200": 5, "Other": 6},
                "procedure": {"100": 7, "200": 8, "Other": 9},
            }
        }
    }
    # act
    actual = OutpatientsModel._baseline_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9]


def test_step_counts(mock_model):
    """test that it estimates the step counts correctly"""
    # arrange
    mdl = mock_model
    attendances_before = pd.Series([1, 2])
    tele_attendances_before = pd.Series([3, 4])
    attendances_after = 6
    tele_attendances_after = 16
    factors = {"a": [0.75, 0.875], "b": [1.5, 2.0]}
    # act
    actual = mdl._step_counts(
        attendances_before,
        tele_attendances_before,
        attendances_after,
        tele_attendances_after,
        factors,
    )
    # arrange
    assert actual == {
        "baseline": {"attendances": 3, "tele_attendances": 7},
        "a": {
            "attendances": -0.9230769230769231,
            "tele_attendances": -3.333333333333334,
        },
        "b": {
            "attendances": 3.9230769230769234,
            "tele_attendances": 12.333333333333334,
        },
    }


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    mdl = mock_model
    rng = Mock()
    rng.poisson = Mock(wraps=lambda x: x)
    mdl._waiting_list_adjustment = Mock(return_value=np.array([1, 2]))
    mdl._followup_reduction = Mock(return_value=np.array([3, 4]))
    mdl._consultant_to_consultant_reduction = Mock(return_value=np.array([5, 6]))
    mdl._expat_adjustment = Mock(return_value=pd.Series([7, 8]))
    mdl._repat_adjustment = Mock(return_value=pd.Series([9, 10]))
    mdl._baseline_adjustment = Mock(return_value=pd.Series([11, 12]))
    mdl._step_counts = Mock(
        return_value={"baseline": {"attendances": 1, "tele_attendances": 2}}
    )
    mdl._convert_to_tele = Mock()
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
        rng, data, run_params, pd.Series({1: 1, 2: 2}), np.array([11, 12])
    )
    # assert
    assert change_factors.to_dict("list") == {
        "change_factor": ["baseline"] * 2,
        "strategy": ["-"] * 2,
        "measure": ["attendances", "tele_attendances"],
        "value": [1, 2],
    }
    assert model_results.to_dict("list") == {
        "rn": [1, 2],
        "attendances": [571725, 6635520],
        "tele_attendances": [800415, 8847360],
    }
    mdl._waiting_list_adjustment.assert_called_once()
    mdl._followup_reduction.assert_called_once()
    mdl._consultant_to_consultant_reduction.assert_called_once()
    mdl._convert_to_tele.assert_called_once()
    mdl._expat_adjustment.assert_called_once()
    mdl._repat_adjustment.assert_called_once()
    mdl._baseline_adjustment.assert_called_once()
    assert rng.poisson.call_count == 2
    mdl._step_counts.assert_called_once()
    assert mdl._step_counts.call_args_list[0][0][0].to_list() == [5, 6]
    assert mdl._step_counts.call_args_list[0][0][1].to_list() == [7, 8]
    assert mdl._step_counts.call_args_list[0][0][2] == 7207245
    assert mdl._step_counts.call_args_list[0][0][3] == 9647775
    assert {
        k: v.tolist() for k, v in mdl._step_counts.call_args_list[0][0][4].items()
    } == {
        "health_status_adjustment": [11, 12],
        "population_factors": [1, 2],
        "expatriation": [7, 8],
        "repatriation": [9, 10],
        "baseline_adjustment": [11, 12],
        "waiting_list_adjustment": [1, 2],
        "followup_reduction": [3, 4],
        "consultant_to_consultant_referrals": [5, 6],
    }


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
    # act
    results = mdl.aggregate(model_results, 1)
    # assert
    assert mdl._create_agg.call_count == 3
    assert results == {"default": 1, "sex+age_group": 1, "sex+tretspef": 1}
    mr_call = pd.DataFrame(
        {
            "pod": [
                k for k in ["op_first", "op_follow-up", "op_procedure"] for _ in [0, 1]
            ],
            "sitetret": ["trust"] * 6,
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
