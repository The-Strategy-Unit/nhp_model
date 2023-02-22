"""test model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

import re
from collections import namedtuple
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.model import Model


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(Model, "__init__", lambda s, m, p, d, c: None):
        mdl = Model(None, None, None, None)
    mdl.model_type = "aae"
    mdl.params = {
        "input_data": "synthetic",
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
        "expat": {"ip": {"elective": {"Other": [0.7, 0.9]}}},
        "repat_local": {"ip": {"elective": {"Other": [1.0, 1.2]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1.3, 1.5]}}},
        "baseline_adjustment": {"ip": {"elective": {"Other": [1.5, 1.7]}}},
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
def mock_run_params():
    """generate the expected run params"""
    return {
        "variant": ["a", "a", "b", "a"],
        "seeds": [1, 2, 3, 4],
        "health_status_adjustment": [0.9, 1, 2, 3],
        "waiting_list_adjustment": "waiting_list_adjustment",
        "expat": {"ip": {"elective": {"Other": [0.8, 4, 5, 6]}}},
        "repat_local": {"ip": {"elective": {"Other": [1.1, 7, 8, 9]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1.4, 10, 11, 12]}}},
        "baseline_adjustment": {"ip": {"elective": {"Other": [1.6, 13, 14, 15]}}},
        "non-demographic_adjustment": {
            "a": {"a_a": [1.1, 16, 17, 18], "a_b": [1.1, 19, 20, 21]},
            "b": {"b_a": [1.1, 22, 23, 24], "b_b": [1.1, 25, 26, 27]},
        },
        "inpatient_factors": {
            "admission_avoidance": {"a_a": [0.5, 28, 29, 30], "a_b": [0.5, 31, 32, 33]},
            "los_reduction": {"b_a": [0.5, 34, 35, 36], "b_b": [0.5, 37, 38, 39]},
        },
        "outpatient_factors": {
            "a": {"a_a": [0.5, 40, 41, 42], "a_b": [0.5, 43, 44, 45]},
            "b": {"b_a": [0.5, 46, 47, 48], "b_b": [0.5, 49, 50, 51]},
        },
        "aae_factors": {
            "a": {"a_a": [0.5, 52, 53, 54], "a_b": [0.5, 55, 56, 57]},
            "b": {"b_a": [0.5, 58, 59, 60], "b_b": [0.5, 61, 62, 63]},
        },
        "bed_occupancy": {
            "a": {"a": [0.5, 73, 74, 75], "b": [0.7, 0.7, 0.7, 0.7]},
            "b": {"a": [0.5, 76, 77, 78], "b": [0.8, 0.8, 0.8, 0.8]},
        },
        "theatres": {
            "change_utilisation": {"a": [1.02, 64, 65, 66], "b": [1.03, 67, 68, 69]},
            "change_availability": [1.04, 70, 71, 72],
        },
    }


@pytest.fixture
def mock_model_results():
    return pd.DataFrame(
        {
            "pod": [i for i in range(3) for _ in range(4)],
            "sitetret": ["trust"] * 12,
            "measure": [0, 1] * 6,
            "col1": [0, 0, 1, 1] * 3,
            "col2": [0, 0, 0, 1] * 3,
            "value": list(range(1, 13)),
        }
    )


# __init__()


@pytest.mark.parametrize("model_type", ["aae", "ip", "op"])
def test_model_init_sets_values(mocker, model_type):
    """test model constructor works as expected"""
    # arrange
    params = {"dataset": "synthetic", "create_datetime": "20220101_012345"}
    mock_data = pd.DataFrame({"rn": [2, 1], "age": [2, 1]})

    mocker.patch("model.model.Model._load_parquet", return_value=mock_data)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model._load_hsa_gams")
    mocker.patch("model.model.Model._generate_run_params")

    mocker.patch("os.path.join", lambda *args: "/".join(args))

    # act
    mdl = Model(model_type, params, "data")

    # assert
    assert mdl.model_type == model_type
    assert mdl.params == params
    assert mdl._data_path == "data/synthetic"
    assert mdl.data.to_dict(orient="list") == {
        "rn": [1, 2],
        "age": [1, 2],
        "age_group": ["age_groups"] * 2,
    }
    mdl._load_demog_factors.assert_called_once()
    mdl._load_hsa_gams.assert_called_once()
    mdl._generate_run_params.assert_called_once()


def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None)


def test_model_init_sets_create_datetime(mocker):
    """it sets the create_datetime item in params if not already set"""
    # arrange
    params = {"dataset": "synthetic"}
    mock_data = pd.DataFrame({"rn": [2, 1], "age": [2, 1]})

    mocker.patch("model.model.Model._load_parquet", return_value=mock_data)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model._load_hsa_gams")
    mocker.patch("model.model.Model._generate_run_params")

    mocker.patch("os.path.join", lambda *args: "/".join(args))

    # act
    mdl = Model("aae", params, "data")

    # assert
    assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


# _load_parquet()


def test_load_parquet(mocker, mock_model):
    """test that load parquet properly loads files"""
    m = Mock()
    m.to_pandas.return_value = "data"
    mocker.patch("pyarrow.parquet.read_pandas", return_value=m)
    mdl = mock_model
    assert mdl._load_parquet("ip") == "data"
    m.expect_called_with_args("data/ip.parquet")
    m.to_pandas.assert_called_once()


# _load_demog_factors()


@pytest.mark.parametrize(
    "start_year, end_year", [("2018", "2019"), ("2018", "2020"), ("2019", "2020")]
)
def test_demog_factors_loads_correctly(mocker, mock_model, start_year, end_year):
    """test that the demographic factors are loaded correctly"""
    mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "variant": ["a"] * 10 + ["b"] * 10,
                "age": list(range(1, 6)) * 4,
                "sex": ([1] * 5 + [2] * 5) * 2,
                "2018": list(range(1, 21)),
                "2019": list(range(11, 31)),
                "2020": list(range(21, 41)),
            }
        ),
    )
    mdl = mock_model
    mdl.params["start_year"] = start_year
    mdl.params["end_year"] = end_year
    mdl._load_demog_factors()

    # choose the values for the ranges to test
    if start_year == "2018":
        a_range = range(1, 11)
        b_range = range(11, 21)
        diff = 10 if end_year == "2019" else 20
    else:
        a_range = range(11, 21)
        b_range = range(21, 31)
        diff = 10

    assert mdl.demog_factors["a"].to_list() == [(i + diff) / i for i in a_range]
    assert mdl.demog_factors["b"].to_list() == [(i + diff) / i for i in b_range]


# _load_hsa_gams()
def test_load_hsa_gams(mocker, mock_model):
    # arrange
    mocker.patch("pickle.load", return_value="pkl_load")

    # act
    with patch("builtins.open", mock_open(read_data="hsa_gams")) as mock_file:
        mock_model._load_hsa_gams()

    # assert
    assert mock_model.hsa_gams == "pkl_load"
    mock_file.assert_called_with("data/synthetic/hsa_gams.pkl", "rb")


# _generate_run_params()


def test_generate_run_params(mocker, mock_model, mock_run_params):
    """test that _generate_run_params returns the run parameters"""
    n = 0

    def get_next_n(*args):  # pylint: disable=unused-argument
        nonlocal n
        n += 1
        return n

    rng = Mock()
    rng.choice.return_value = np.array(["a", "b", "a"])
    rng.integers.return_value = np.array([1, 2, 3, 4])
    rng.normal = Mock(wraps=get_next_n)
    mocker.patch("numpy.random.default_rng", return_value=rng)

    mocker.patch("model.model.inrange", wraps=lambda x, low, high: x)
    # arrange
    mdl = mock_model
    mdl._variants = ["a", "b"]
    mdl._probabilities = [0.6, 0.4]

    # act
    mdl._generate_run_params()
    # assert
    assert mdl.run_params == mock_run_params


# _get_run_params()


@pytest.mark.parametrize(
    "model_run, expected_run_params",
    [
        (
            0,
            {
                "variant": "a",
                "health_status_adjustment": 0.9,
                "seed": 1,
                "non-demographic_adjustment": {
                    "a": {"a_a": 1.1, "a_b": 1.1},
                    "b": {"b_a": 1.1, "b_b": 1.1},
                },
                "expat": {"ip": {"elective": {"Other": 0.8}}},
                "repat_local": {"ip": {"elective": {"Other": 1.1}}},
                "repat_nonlocal": {"ip": {"elective": {"Other": 1.4}}},
                "baseline_adjustment": {"ip": {"elective": {"Other": 1.6}}},
                "inpatient_factors": {
                    "admission_avoidance": {"a_a": 0.5, "a_b": 0.5},
                    "los_reduction": {"b_a": 0.5, "b_b": 0.5},
                },
                "outpatient_factors": {
                    "a": {"a_a": 0.5, "a_b": 0.5},
                    "b": {"b_a": 0.5, "b_b": 0.5},
                },
                "aae_factors": {
                    "a": {"a_a": 0.5, "a_b": 0.5},
                    "b": {"b_a": 0.5, "b_b": 0.5},
                },
                "bed_occupancy": {
                    "a": {"a": 0.5, "b": 0.7},
                    "b": {"a": 0.5, "b": 0.8},
                },
                "theatres": {
                    "change_utilisation": {"a": 1.02, "b": 1.03},
                    "change_availability": 1.04,
                },
                "waiting_list_adjustment": "waiting_list_adjustment",
            },
        ),
        (
            2,
            {
                "variant": "b",
                "seed": 3,
                "health_status_adjustment": 2,
                "non-demographic_adjustment": {
                    "a": {"a_a": 17, "a_b": 20},
                    "b": {"b_a": 23, "b_b": 26},
                },
                "expat": {"ip": {"elective": {"Other": 5}}},
                "repat_local": {"ip": {"elective": {"Other": 8}}},
                "repat_nonlocal": {"ip": {"elective": {"Other": 11}}},
                "baseline_adjustment": {"ip": {"elective": {"Other": 14}}},
                "inpatient_factors": {
                    "admission_avoidance": {"a_a": 29, "a_b": 32},
                    "los_reduction": {"b_a": 35, "b_b": 38},
                },
                "outpatient_factors": {
                    "a": {"a_a": 41, "a_b": 44},
                    "b": {"b_a": 47, "b_b": 50},
                },
                "aae_factors": {
                    "a": {"a_a": 53, "a_b": 56},
                    "b": {"b_a": 59, "b_b": 62},
                },
                "bed_occupancy": {
                    "a": {"a": 74, "b": 0.7},
                    "b": {"a": 77, "b": 0.8},
                },
                "theatres": {
                    "change_utilisation": {"a": 65, "b": 68},
                    "change_availability": 71,
                },
                "waiting_list_adjustment": "waiting_list_adjustment",
            },
        ),
    ],
)
def test_get_run_params(mock_model, mock_run_params, model_run, expected_run_params):
    """tests _get_run_params gets the right params for a model run"""
    # arrange
    mdl = mock_model
    mdl.run_params = mock_run_params
    # act
    actual = mdl._get_run_params(model_run)
    # assert
    assert actual == expected_run_params


# run()


def test_run(mocker, mock_model):
    """test run calls the _run method correctly"""
    # arrange
    mr_mock = mocker.patch("model.model.ModelRun", return_value="model_run")
    mdl = mock_model
    mdl._run = Mock()

    mdl._baseline_counts = 1
    mdl._data_mask = "data mask"

    rr_mock = mocker.patch("model.model.ActivityAvoidance")
    rr_mock.return_value = rr_mock
    rr_mock.demographic_adjustment.return_value = rr_mock
    rr_mock.health_status_adjustment.return_value = rr_mock
    rr_mock.expat_adjustment.return_value = rr_mock
    rr_mock.repat_adjustment.return_value = rr_mock
    rr_mock.waiting_list_adjustment.return_value = rr_mock
    rr_mock.baseline_adjustment.return_value = rr_mock
    rr_mock.non_demographic_adjustment.return_value = rr_mock
    rr_mock.activity_avoidance.return_value = rr_mock
    rr_mock.apply_resampling.return_value = rr_mock

    # act
    actual = mdl.run(0)

    # assert
    assert actual == "model_run"

    rr_mock.assert_called_once_with("model_run", 1, row_mask="data mask")
    rr_mock.demographic_adjustment.assert_called_once()
    rr_mock.health_status_adjustment.assert_called_once()
    rr_mock.expat_adjustment.assert_called_once()
    rr_mock.repat_adjustment.assert_called_once()
    rr_mock.waiting_list_adjustment.assert_called_once()
    rr_mock.baseline_adjustment.assert_called_once()
    rr_mock.non_demographic_adjustment.assert_called_once()
    rr_mock.activity_avoidance.assert_called_once()
    rr_mock.apply_resampling.assert_called_once()

    mr_mock.assert_called_once_with(mdl, 0)
    mdl._run.assert_called_once_with("model_run")


def test__run(mock_model):
    """test the _run method"""
    # it does nothing, test is purely for coverage purposes
    mock_model._run(None)


# _create_agg()


@pytest.mark.parametrize(
    "cols, name, include_measure, expected",
    [
        (
            None,
            None,
            True,
            lambda r: {
                "default": {
                    r("trust", 0, 0): 4,
                    r("trust", 0, 1): 6,
                    r("trust", 1, 0): 12,
                    r("trust", 1, 1): 14,
                    r("trust", 2, 0): 20,
                    r("trust", 2, 1): 22,
                }
            },
        ),
        (
            None,
            "thing",
            True,
            lambda r: {
                "thing": {
                    r("trust", 0, 0): 4,
                    r("trust", 0, 1): 6,
                    r("trust", 1, 0): 12,
                    r("trust", 1, 1): 14,
                    r("trust", 2, 0): 20,
                    r("trust", 2, 1): 22,
                }
            },
        ),
        (
            ["col1"],
            None,
            True,
            lambda r: {
                "col1": {
                    r("trust", 0, 0, 0): 1,
                    r("trust", 0, 0, 1): 3,
                    r("trust", 0, 1, 0): 2,
                    r("trust", 0, 1, 1): 4,
                    r("trust", 1, 0, 0): 5,
                    r("trust", 1, 0, 1): 7,
                    r("trust", 1, 1, 0): 6,
                    r("trust", 1, 1, 1): 8,
                    r("trust", 2, 0, 0): 9,
                    r("trust", 2, 0, 1): 11,
                    r("trust", 2, 1, 0): 10,
                    r("trust", 2, 1, 1): 12,
                }
            },
        ),
        (
            ["col1", "col2"],
            None,
            False,
            lambda r: {
                "col1+col2": {
                    r("trust", 0, 0, 0): 3,
                    r("trust", 0, 1, 0): 3,
                    r("trust", 0, 1, 1): 4,
                    r("trust", 1, 0, 0): 11,
                    r("trust", 1, 1, 0): 7,
                    r("trust", 1, 1, 1): 8,
                    r("trust", 2, 0, 0): 19,
                    r("trust", 2, 1, 0): 11,
                    r("trust", 2, 1, 1): 12,
                }
            },
        ),
    ],
)
def test_create_agg(
    mock_model, mock_model_results, cols, name, include_measure, expected
):
    """test that it aggregates the data correctly"""
    actual = mock_model._create_agg(mock_model_results, cols, name, include_measure)
    cols = ["sitetret", "pod"] + (["measure"] if include_measure else []) + (cols or [])
    assert actual == expected(namedtuple("results", cols))
