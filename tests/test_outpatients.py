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
        "op_factors": {
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
    mdl._hsa_gams = {(i, j): hsa_mock for i in ["op_a_a", "op_b_b"] for j in [1, 2]}
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
    ubd_mock = mocker.patch("model.outpatients.OutpatientsModel._update_baseline_data")
    gdc_mock = mocker.patch(
        "model.outpatients.OutpatientsModel._get_data_counts", return_value=1
    )
    lst_mock = mocker.patch("model.outpatients.OutpatientsModel._load_strategies")
    # act
    mdl = OutpatientsModel("params", "data_path")
    # assert
    super_mock.assert_called_once()
    ubd_mock.assert_called_once()
    gdc_mock.assert_called_once_with(None)
    lst_mock.assert_called_once()
    assert mdl._baseline_counts == 1


def test_update_baseline_data(mock_model):
    # arrange
    mdl = mock_model
    mdl.data["has_procedures"] = [True] * 10 + [False] * 10
    mdl.data["is_first"] = ([True] * 5 + [False] * 5) * 2
    # act
    mdl._update_baseline_data()
    # assert
    assert (
        mdl.data["group"].to_list()
        == ["procedure"] * 10 + ["first"] * 5 + ["followup"] * 5
    )
    assert mdl.data["is_wla"].to_list() == [True] * 20


def test_get_data_counts(mock_model):
    # arrange
    mdl = mock_model
    data = mdl.data
    data["attendances"] = list(range(1, 21))
    data["tele_attendances"] = list(range(21, 41))
    # act
    actual = mdl._get_data_counts(data)
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
    # act
    mdl._load_strategies()
    # assert
    assert mdl._strategies["activity_avoidance"]["strategy"].to_list() == [
        f"{i}_{j}"
        for i in ["followup_reduction"] + ["consultant_to_consultant_reduction"] * 2
        for j in ["a", "b", "c", "d", "e"]
    ]
    assert mdl._strategies["activity_avoidance"]["sample_rate"].to_list() == [1] * 15


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
        ("efficiencies", "convert_to_tele"): {
            "attendances": -10,
            "tele_attendances": 10,
        }
    }


def test_run(mocker, mock_model):
    """test that it runs the model steps"""
    # arrange
    model_run = Mock()
    model_run.rng = "rng"
    model_run.run_params = {"outpatient_factors": "outpatient_factors"}
    mdl = mock_model
    mdl._baseline_counts = 0
    rr_mock = Mock()
    mocker.patch("model.outpatients.RowResampling", return_value=rr_mock)
    rr_mock.demographic_adjustment.return_value = rr_mock
    rr_mock.health_status_adjustment.return_value = rr_mock
    rr_mock.expat_adjustment.return_value = rr_mock
    rr_mock.repat_adjustment.return_value = rr_mock
    rr_mock.waiting_list_adjustment.return_value = rr_mock
    rr_mock.baseline_adjustment.return_value = rr_mock
    rr_mock.activity_avoidance.return_value = rr_mock
    rr_mock.apply_resampling.return_value = (
        pd.DataFrame({"rn": [1], "hsagrp": [1]}),
        {
            ("baseline", "-"): np.array([1, 2]),
            ("demographic_adjustment", "-"): np.array([3, 4]),
        },
    )
    mdl._convert_to_tele = Mock()
    # act
    change_factors, model_results = mdl._run(model_run)
    # assert
    assert change_factors.to_dict("list") == {
        "change_factor": ["baseline", "demographic_adjustment"] * 2,
        "strategy": ["-"] * 4,
        "measure": ["attendances"] * 2 + ["tele_attendances"] * 2,
        "value": [1, 3, 2, 4],
    }
    assert model_results.to_dict("list") == {"rn": [1]}
    mdl._convert_to_tele.assert_called_once()


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
