"""test inpatients model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from model.inpatients import InpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(InpatientsModel, "__init__", lambda s, p, d: None):
        mdl = InpatientsModel(None, None)
    mdl.model_type = "ip"
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
        "theatres": {
            "change_utilisation": {"a": [1.01, 1.03], "b": [1.02, 1.04]},
            "change_availability": [1.03, 1.05],
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


@pytest.fixture
def mock_losr():
    return pd.DataFrame(
        {
            "type": [x for x in ["all", "zero"] for _ in [0, 1]] + ["bads"] * 3,
            "baseline_target_rate": [pd.NA] * 4 + [0.25, 0.50, 0.75],
            "op_dc_split": [pd.NA] * 4 + [0, 1, 0.5],
            "losr_f": [1 - 1 / (2**x) for x in range(7)],
        },
        index=["a", "b", "c", "d", "e", "f", "g"],
    )


# methods


def test_init_calls_super_init(mocker):
    """test that the model calls the super method and loads the strategies"""
    mocker.patch("model.inpatients.super")
    mocker.patch(
        "model.inpatients.InpatientsModel._load_parquet", wraps=lambda _: Mock()
    )
    mocker.patch("model.inpatients.InpatientsModel._load_theatres_data")
    mdl = InpatientsModel("params", "data_path")
    # no asserts to perform, so long as this method doesn't fail
    assert mdl._load_parquet.call_count == 2
    assert (
        mdl._load_parquet.call_args_list[0][0][0] == "ip_admission_avoidance_strategies"
    )
    assert mdl._load_parquet.call_args_list[1][0][0] == "ip_los_reduction_strategies"
    mdl._load_theatres_data.assert_called_once()


def test_load_theatres_data(mocker, mock_model):
    """test that it loads the json file correctly"""
    json_mock = mocker.patch(
        "json.load",
        return_value={
            "theatres": 10,
            "four_hour_sessions": {"100": 1, "200": 2, "Other": 3},
        },
    )
    with patch("builtins.open", mock_open()) as mock_file:
        mock_model._load_theatres_data()
        json_mock.assert_called_once()
        mock_file.assert_called_with(
            "data/synthetic/theatres.json", "r", encoding="UTF-8"
        )
        assert mock_model._theatres_data["theatres"] == 10
        assert mock_model._theatres_data["four_hour_sessions"].equals(
            pd.Series({"100": 1, "200": 2, "Other": 3}, name="four_hour_sessions")
        )


def test_los_reduction(mock_model):
    """test that the method returns a dataframe"""
    mdl = mock_model
    actual = mdl._los_reduction(
        {"inpatient_factors": {"los_reduction": {"b_a": 0.45, "b_b": 0.55}}}
    )
    assert actual.equals(
        pd.DataFrame(
            {"interval": [[0.4, 0.6], [0.4, 0.6]], "losr_f": [0.45, 0.55]},
            index=["b_a", "b_b"],
        )
    )


def test_random_strategy(mocker, mock_model):
    """test that it selects one random strategy"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = np.asarray([1, 1, 0, 1, 1, 0, 1, 1])
    rng.bit_generator = np.random.default_rng(1).bit_generator
    mocker.patch("pandas.DataFrame.sample", wraps=lambda x, *_: x)
    mdl = mock_model
    mdl._strategies = {
        "a": pd.DataFrame(
            {
                "strategy": ["a", "b", "c", "NULL", "b", "c", "d", "NULL"],
                "sample_rate": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            index=[1, 1, 1, 1, 2, 2, 2, 2],
        )
    }
    mdl.params = {"inpatient_factors": {"a": {"a": 1, "b": 2, "c": 3}}}
    # act
    actual = mdl._random_strategy(rng, "a")
    # assert
    assert actual.equals(pd.Series(["NULL", "a"], index=[2, 1]))


def test_waiting_list_adjustment(mock_model):
    """test that it returns the wla numpy array"""
    data = pd.DataFrame(
        {
            "tretspef": [1] * 2 + [2] * 8 + [3] * 4 + [4] * 1,
            "admimeth": ["11", "21"] * 7 + ["11"],
        }
    )
    mdl = mock_model
    mdl.params["waiting_list_adjustment"] = {"ip": {1: 1, 2: 2, 3: 3, 5: 1}}
    actual = mdl._waiting_list_adjustment(data)
    expected = [1.5, 1] + [1.25, 1] * 4 + [1.75, 1] * 2 + [1]
    assert np.array_equal(actual, expected)


def test_non_demographic_adjustment(mock_model):
    """test that is returns the nda numpy array"""
    run_params = {
        "non-demographic_adjustment": {
            "elective": {
                0: 1,
                1: 2,
            },
            "maternity": {
                0: 3,
                1: 4,
            },
            "non-elective": {
                0: 5,
                1: 6,
            },
        }
    }
    data = pd.DataFrame(
        {
            "admimeth": ["11", "12", "21", "22", "31", "32", "81", "82"],
            "age_group": [0, 1] * 4,
        }
    )
    actual = mock_model._non_demographic_adjustment(data, run_params)
    expected = [1, 2, 5, 6, 3, 4, 5, 6]
    assert list(actual) == expected


def test_admission_avoidance_step(mock_model):
    """test admission avoidance step returns the correct data and step counts"""
    # arrange
    x = list(range(9))
    rng = Mock()
    rng.binomial.return_value = np.asarray([1] * 6 + [0] * 3)
    admission_avoidance = pd.Series(
        ["a_a", "a_b", "NULL"] * 3, index=x, name="admission_avoidance_strategy"
    )
    data = pd.DataFrame({"rn": x, "speldur": x})
    run_params = {
        "inpatient_factors": {"admission_avoidance": {"a_a": 0.25, "a_b": 0.75}}
    }
    step_counts = {}
    # act
    data = mock_model._admission_avoidance_step(
        rng, data, admission_avoidance, run_params, step_counts
    )
    # assert
    assert data["rn"].tolist() == list(range(6))
    assert data["speldur"].tolist() == list(range(6))
    assert step_counts == {
        ("admission_avoidance", "NULL"): {"admissions": -1, "beddays": -9},
        ("admission_avoidance", "a_a"): {"admissions": -1, "beddays": -7},
        ("admission_avoidance", "a_b"): {"admissions": -1, "beddays": -8},
    }


def test_losr_all(mock_model, mock_losr):
    """test that it reduces the speldur column for 'all' types"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = list(range(6))
    losr = mock_losr
    data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "a", "b"] * 3)
    step_counts = {}
    # act
    mock_model._losr_all(data, losr, rng, step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 0, 3, 3, 1, 4, 6, 2, 5]
    assert step_counts == {
        ("los_reduction", "a"): {"admissions": 0, "beddays": -9},
        ("los_reduction", "b"): {"admissions": 0, "beddays": -3},
    }
    assert rng.binomial.call_args_list[0][0][0].to_list() == [1, 4, 7, 2, 5, 8]
    assert rng.binomial.call_args_list[0][0][1].to_list() == [0, 0, 0, 0.5, 0.5, 0.5]


def test_losr_bads(mock_model, mock_losr):
    """test that it reduces the speldur column for 'bads' types"""
    rng = Mock()
    # 1: rvr < ur0
    # 2: rvr < ur0 + ur1
    # 3: rvr >= ur0 + ur1
    rng.uniform.return_value = (
        [0.08, 0.08, 0.09] + [0.06, 0.07, 1.0] + [0.06, 0.07, 0.55]
    )
    losr = mock_losr
    data = pd.DataFrame(
        {"speldur": list(range(12)), "classpat": ["1"] * 6 + ["2"] * 3 + ["1"] * 3},
        index=[x for x in ["x", "e", "f", "g"] for _ in range(3)],
    )
    step_counts = {}
    # act
    mock_model._losr_bads(data, losr, rng, step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 1, 2, 3, 4, 0, 0, 0, 0, 9, 0, 0]
    assert step_counts == {
        ("los_reduction", "e"): {"admissions": -1, "beddays": -5},
        ("los_reduction", "f"): {"admissions": -1, "beddays": -21},
        ("los_reduction", "g"): {"admissions": -1, "beddays": -21},
    }
    assert rng.uniform.call_args_list[0][1] == {"size": 9}


def test_losr_zero(mock_model, mock_losr):
    """test that it reduces the speldur column for 'zero' types"""
    # arrange
    rng = Mock()
    rng.uniform.return_value = [0.75, 0.7, 0.8, 0.875, 0.8, 0.9]
    losr = mock_losr
    data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "c", "d"] * 3)
    step_counts = {}
    # act
    mock_model._losr_to_zero(data, losr, rng, "zero", step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 1, 2, 3, 0, 0, 6, 7, 8]
    assert step_counts == {
        ("los_reduction", "c"): {"admissions": 0, "beddays": -4},
        ("los_reduction", "d"): {"admissions": 0, "beddays": -5},
    }
    assert rng.uniform.call_args_list[0][1] == {"size": 6}


def test_run_poisson_step(mock_model):
    """test that it applies the poisson step, returns the mutates the data, and updates step counts"""
    # arrange
    rng = Mock()
    rng.poisson = Mock(wraps=lambda xs: [2 * x for x in xs])
    data = pd.DataFrame({"rn": [0, 1, 2, 3], "speldur": [0, 1, 2, 3]})
    mdl = mock_model
    step_counts = {}
    # act
    data = mdl._run_poisson_step(rng, data, "test", [1, 2, 3, 0], step_counts)
    # assert
    assert data.rn.to_list() == [0] * 2 + [1] * 4 + [2] * 6
    assert data.speldur.to_list() == [0] * 2 + [1] * 4 + [2] * 6
    assert rng.poisson.call_count == 1
    assert step_counts == {("test", "-"): {"admissions": 8, "beddays": 18}}


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    data = pd.DataFrame({"rn": [1, 2], "speldur": [0, 1], "hsagrp": ["a", "b"]})
    mdl = mock_model
    mdl._random_strategy = Mock(
        return_value=pd.Series([1, 2], index=[1, 2]).rename_axis("rn")
    )
    mdl._los_reduction = Mock()
    mdl._run_poisson_step = Mock(wraps=lambda x, y, *args: y)
    mdl._waiting_list_adjustment = Mock()
    mdl._non_demographic_adjustment = Mock()
    mdl._admission_avoidance_step = Mock(wraps=lambda x, y, *args: y)
    mdl._losr_all = Mock()
    mdl._losr_to_zero = Mock()
    mdl._losr_bads = Mock()
    # act
    change_factors, model_results = mdl._run(
        None, data, "run_params", pd.Series({1: 1, 2: 2}), "hsa_f"
    )
    # assert
    assert change_factors.equals(
        pd.DataFrame(
            {
                "change_factor": ["baseline"] * 2,
                "strategy": ["-"] * 2,
                "measure": ["admissions", "beddays"],
                "value": np.array([2, 3]),
            }
        )
    )
    assert model_results.equals(
        data.drop("hsagrp", axis="columns").reset_index(drop=True)
    )
    assert mdl._run_poisson_step.call_count == 4
    assert mdl._random_strategy.call_count == 2
    mdl._los_reduction.assert_called_once()
    mdl._waiting_list_adjustment.assert_called_once()
    mdl._non_demographic_adjustment.assert_called_once()
    mdl._admission_avoidance_step.assert_called_once()
    mdl._losr_all.assert_called_once()
    assert mdl._losr_to_zero.call_count == 2
    mdl._losr_bads.assert_called_once()


def test_bed_occupancy(mocker, mock_model):
    """test that it aggregates the bed occupancy data"""
    # arrange
    mock_model.params["bed_occupancy"] = {
        "specialty_mapping": {
            "General and Acute": {"100": "a", "200": "b", "300": "b", "500": "c"}
        }
    }
    read_csv = mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "specialty_code": ["100", "200", "300", "400"],
                "specialty_group": ["General and Acute"] * 4,
                "available": [10, 20, 30, 40],
                "occupied": [5, 15, 25, 35],
            }
        ),
    )
    mock_model.data = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "2", "4"],
            "mainspef": ["100", "200", "300", "100", "200"],
            "speldur": [1, 2, 3, 4, 5],
        }
    )
    model_results = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "1", "1", "2", "4"] * 2,
            "mainspef": ["100", "100", "200", "200", "300", "100", "200"] * 2,
            "pod": [x for x in ["ip_admission", "ip_daycase"] for _ in range(7)],
            "measure": ["beddays"] * 14,
            "value": [1, 2, 3, 4, 5, 6, 7] * 2,
        }
    )
    # act
    actual = mock_model._bed_occupancy(
        model_results, {"day+night": {"a": 0.75, "b": 0.875}}, 0
    )
    # assert
    assert {tuple(k): v for k, v in actual.items()} == {
        ("ip", "day+night", "a"): (9 * 5) / (2 * 0.75),
        ("ip", "day+night", "b"): (19 * 40) / (13 * 0.875),
    }
    read_csv.assert_called_once_with(
        "data/synthetic/kh03.csv",
        dtype={
            "specialty_code": np.character,
            "specialty_group": np.character,
            "available": np.float64,
            "occupied": np.float64,
        },
    )


def test_bed_occupancy_baseline(mocker, mock_model):
    """test that it just uses the kh03 available data"""
    # arrange
    mock_model.params["bed_occupancy"] = {
        "specialty_mapping": {
            "General and Acute": {"100": "a", "200": "b", "300": "b", "500": "c"}
        }
    }
    read_csv = mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "specialty_code": ["100", "200", "300", "400"],
                "specialty_group": ["General and Acute"] * 4,
                "available": [10, 20, 30, 40],
                "occupied": [5, 15, 25, 35],
            }
        ),
    )
    mock_model.data = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "2", "4"],
            "mainspef": ["100", "200", "300", "100", "200"],
            "speldur": [1, 2, 3, 4, 5],
        }
    )
    model_results = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "1", "1", "2", "4"] * 2,
            "mainspef": ["100", "100", "200", "200", "300", "100", "200"] * 2,
            "pod": [x for x in ["ip_admission", "ip_daycase"] for _ in range(7)],
            "measure": ["beddays"] * 14,
            "value": [1, 2, 3, 4, 5, 6, 7] * 2,
        }
    )
    # act
    actual = mock_model._bed_occupancy(
        model_results, {"day+night": {"a": 0.75, "b": 0.875}}, -1
    )
    # assert
    assert {tuple(k): v for k, v in actual.items()} == {
        ("ip", "day+night", "a"): 10,
        ("ip", "day+night", "b"): 50,
    }


def test_theatres_available(mock_model):
    """test that it aggregates the theatres data"""
    # arrange
    mock_model.data = pd.DataFrame(
        [
            {"tretspef": t, "admimeth": a, "has_procedure": p}
            for t in ["100", "110", "200", "Other"]
            for a in ["11", "21"]
            for p in [1, 0]
        ]
    )
    mock_model._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other": 300}, name="four_hour_sessions"
        ),
    }
    model_results = pd.concat([mock_model.data] * 3)
    # act
    theatres_available = mock_model._theatres_available(
        model_results,
        {
            "change_utilisation": {"100": 2, "110": 2.5, "Other": 3},
            "change_availability": 5,
        },
        0,
    )
    # assert
    assert {tuple(k): v for k, v in theatres_available.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 150.0,
        ("ip_theatres", "four_hour_sessions", "110"): 240.0,
        ("ip_theatres", "four_hour_sessions", "Other"): 300.0,
        ("ip_theatres", "theatres"): 2.3,
    }


def test_theatres_available_baseline(mock_model):
    """test that it returns the baseline theatres data"""
    # arrange
    mock_model.data = pd.DataFrame(
        [
            {"tretspef": t, "admimeth": a, "has_procedure": p}
            for t in ["100", "110", "200", "Other"]
            for a in ["11", "21"]
            for p in [1, 0]
        ]
    )
    mock_model._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other": 300}, name="four_hour_sessions"
        ),
    }
    model_results = pd.concat([mock_model.data] * 3)
    # act
    theatres_available = mock_model._theatres_available(
        model_results,
        {
            "change_utilisation": {"100": 2, "110": 2.5, "Other": 3},
            "change_availability": 5,
        },
        -1,
    )
    # assert
    assert {tuple(k): v for k, v in theatres_available.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 100,
        ("ip_theatres", "four_hour_sessions", "110"): 200,
        ("ip_theatres", "four_hour_sessions", "Other"): 300,
        ("ip_theatres", "theatres"): 10,
    }


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl._get_run_params = Mock(
        return_value={"bed_occupancy": "run_params", "theatres": "theatres"}
    )
    mdl._bed_occupancy = Mock(return_value=2)
    mdl._theatres_available = Mock(return_value=3)
    xs = list(range(6)) * 2
    model_results = pd.DataFrame(
        {
            "classpat": ["1", "2", "3", "4", "5", "-1"] * 2,
            "admimeth": [x for x in ["1", "2"] for _ in range(6)],
            "speldur": xs,
            "rn": xs,
            "age_group": xs,
            "sex": xs,
            "tretspef": xs,
            "mainspef": xs,
        }
    )
    results = mdl.aggregate(model_results, 1)
    assert mdl._create_agg.call_count == 3
    assert results == {
        "default": 1,
        "sex+age_group": 1,
        "sex+tretspef": 1,
        "bed_occupancy": 2,
        "theatres_available": 3,
    }
    #
    mr_call = {
        "pod": ["op_procedure"] * 2
        + [
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_elective_daycase",
            "ip_elective_admission",
            "ip_elective_birth-episode",
            "ip_non-elective_admission",
            "ip_elective_daycase",
            "ip_elective_daycase",
            "ip_non-elective_admission",
            "ip_non-elective_birth-episode",
        ]
        * 2,
        "age_group": [5, 5] + [0, 1, 2, 3, 4] * 4,
        "rn": [5, 5] + [0, 1, 2, 3, 4] * 4,
        "sex": [5, 5] + [0, 1, 2, 3, 4] * 4,
        "mainspef": [5, 5] + [0, 1, 2, 3, 4] * 4,
        "tretspef": [5, 5] + [0, 1, 2, 3, 4] * 4,
        "measure": ["attendances"] * 2 + ["admissions"] * 10 + ["beddays"] * 10,
        "value": [1] * 12 + [1, 2, 3, 4, 5] * 2,
    }
    for i in [0, 1, 2]:
        for k in mr_call.keys():
            assert (
                mdl._create_agg.call_args_list[i][0][0][k].to_list() == mr_call[k]
            ), f"{i=},{k=}"

    mdl._get_run_params.assert_called_once_with(1)
    mdl._bed_occupancy.assert_called_once()
    mdl._theatres_available.assert_called_once()
    for k in mr_call.keys():
        assert mdl._bed_occupancy.call_args_list[0][0][0][k].to_list() == mr_call[k][2:]
    assert mdl._bed_occupancy.call_args_list[0][0][1] == "run_params"


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    results = pd.DataFrame(
        {
            "rn": [0],
            "classpat": [1],
            "age": [2],
            "sex": [3],
            "tretspef": [4],
            "speldur": [5],
        }
    )
    mock_model.save_results(results, path_fn)
    assert to_parquet_mock.call_count == 2
    assert to_parquet_mock.call_args_list[0] == call("op_conversion/0.parquet")
    assert to_parquet_mock.call_args_list[1] == call("ip/0.parquet")
