"""test model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

import re
from unittest.mock import Mock, patch

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
        "activity_avoidance": {
            "ip": {
                "a_a": {"interval": [0.4, 0.6]},
                "a_b": {"interval": [0.4, 0.6]},
            },
            "op": {"a_a": {"interval": [0.4, 0.6]}, "a_b": {"interval": [0.4, 0.6]}},
            "aae": {"a_a": {"interval": [0.4, 0.6]}, "a_b": {"interval": [0.4, 0.6]}},
        },
        "efficiencies": {
            "ip": {
                "b_a": {"interval": [0.4, 0.6]},
                "b_b": {"interval": [0.4, 0.6]},
            },
            "op": {"b_a": {"interval": [0.4, 0.6]}, "b_b": {"interval": [0.4, 0.6]}},
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
        "activity_avoidance": {
            "ip": {"a_a": [0.5, 28, 29, 30], "a_b": [0.5, 31, 32, 33]},
            "op": {"a_a": [0.5, 34, 35, 36], "a_b": [0.5, 37, 38, 39]},
            "aae": {"a_a": [0.5, 40, 41, 42], "a_b": [0.5, 43, 44, 45]},
        },
        "efficiencies": {
            "ip": {"b_a": [0.5, 46, 47, 48], "b_b": [0.5, 49, 50, 51]},
            "op": {"b_a": [0.5, 52, 53, 54], "b_b": [0.5, 55, 56, 57]},
        },
        "theatres": {
            "change_utilisation": {"a": [1.02, 58, 59, 60], "b": [1.03, 61, 62, 63]},
            "change_availability": [1.04, 64, 65, 66],
        },
        "bed_occupancy": {
            "a": {"a": [0.5, 67, 68, 69], "b": [0.7, 0.7, 0.7, 0.7]},
            "b": {"a": [0.5, 70, 71, 72], "b": [0.8, 0.8, 0.8, 0.8]},
        },
    }


@pytest.fixture
def mock_model_results():
    """creates mock of model results object"""
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
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")
    mocker.patch("model.model.Model._get_data_mask", return_value="data_mask")
    mocker.patch("model.model.Model._get_data_counts", return_value="data_counts")

    mocker.patch("os.path.join", lambda *args: "/".join(args))

    # act
    mdl = Model(model_type, params, "data", "hsa", "run_params")

    # assert
    assert mdl.model_type == model_type
    assert mdl.params == params
    assert mdl._data_path == "data/synthetic"
    assert mdl.data.to_dict(orient="list") == {
        "rn": [1, 2],
        "age": [1, 2],
        "age_group": ["age_groups"] * 2,
    }
    mdl._load_strategies.assert_called_once()
    mdl._load_demog_factors.assert_called_once()
    assert mdl.hsa == "hsa"
    mdl.generate_run_params.assert_not_called()
    assert mdl.run_params == "run_params"
    assert mdl.data_mask == "data_mask"
    assert mdl.baseline_counts == "data_counts"
    mdl._get_data_counts.call_args_list[0][0][0].equals(mdl.data)


def test_model_init_calls_generate_run_params(mocker):
    """test model constructor works as expected"""
    # arrange
    params = {"dataset": "synthetic", "create_datetime": "20220101_012345"}
    mock_data = pd.DataFrame({"rn": [2, 1], "age": [2, 1]})

    mocker.patch("model.model.Model._load_parquet", return_value=mock_data)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params", return_value="generated")
    mocker.patch("model.model.Model._get_data_mask", return_value="data_mask")
    mocker.patch("model.model.Model._get_data_counts", return_value="data_counts")

    mocker.patch("os.path.join", lambda *args: "/".join(args))

    # act
    mdl = Model("aae", params, "data", "hsa")

    # assert
    mdl.generate_run_params.assert_called_once()
    assert mdl.run_params == "generated"


def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None, None, None)


def test_model_init_sets_create_datetime(mocker):
    """it sets the create_datetime item in params if not already set"""
    # arrange
    params = {"dataset": "synthetic"}
    mock_data = pd.DataFrame({"rn": [2, 1], "age": [2, 1]})

    mocker.patch("model.model.Model._load_parquet", return_value=mock_data)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")

    mocker.patch("os.path.join", lambda *args: "/".join(args))

    # act
    mdl = Model("aae", params, "data", "run_params")

    # assert
    assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


# _load_parquet()


def test_load_parquet(mocker, mock_model):
    """test that load parquet properly loads files"""
    # arrange
    m = mocker.patch("pandas.read_parquet", return_value="read_parquet")
    mdl = mock_model

    # act
    actual = mdl._load_parquet("ip")

    # assert
    assert actual == "read_parquet"
    m.expect_called_with_args("data/ip.parquet")


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
    actual = mdl.generate_run_params(mdl.params)

    # assert
    assert actual == mock_run_params


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
                "activity_avoidance": {
                    "ip": {"a_a": 0.5, "a_b": 0.5},
                    "op": {"a_a": 0.5, "a_b": 0.5},
                    "aae": {"a_a": 0.5, "a_b": 0.5},
                },
                "efficiencies": {
                    "ip": {"b_a": 0.5, "b_b": 0.5},
                    "op": {"b_a": 0.5, "b_b": 0.5},
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
                "activity_avoidance": {
                    "ip": {"a_a": 29, "a_b": 32},
                    "op": {"a_a": 35, "a_b": 38},
                    "aae": {"a_a": 41, "a_b": 44},
                },
                "efficiencies": {
                    "ip": {"b_a": 47, "b_b": 50},
                    "op": {"b_a": 53, "b_b": 56},
                },
                "theatres": {
                    "change_utilisation": {"a": 59, "b": 62},
                    "change_availability": 65,
                },
                "bed_occupancy": {
                    "a": {"a": 68, "b": 0.7},
                    "b": {"a": 71, "b": 0.8},
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


# _create_agg()


@pytest.mark.parametrize(
    "cols, name, include_measure, expected",
    [
        (
            None,
            None,
            True,
            {
                "default": {
                    frozenset({("pod", 0), ("measure", 0), ("sitetret", "trust")}): 4,
                    frozenset({("pod", 0), ("sitetret", "trust"), ("measure", 1)}): 6,
                    frozenset({("pod", 1), ("measure", 0), ("sitetret", "trust")}): 12,
                    frozenset({("pod", 1), ("sitetret", "trust"), ("measure", 1)}): 14,
                    frozenset({("pod", 2), ("sitetret", "trust"), ("measure", 0)}): 20,
                    frozenset({("pod", 2), ("sitetret", "trust"), ("measure", 1)}): 22,
                }
            },
        ),
        (
            None,
            "thing",
            True,
            {
                "thing": {
                    frozenset({("sitetret", "trust"), ("measure", 0), ("pod", 0)}): 4,
                    frozenset({("sitetret", "trust"), ("measure", 1), ("pod", 0)}): 6,
                    frozenset({("pod", 1), ("sitetret", "trust"), ("measure", 0)}): 12,
                    frozenset({("pod", 1), ("sitetret", "trust"), ("measure", 1)}): 14,
                    frozenset({("pod", 2), ("sitetret", "trust"), ("measure", 0)}): 20,
                    frozenset({("pod", 2), ("sitetret", "trust"), ("measure", 1)}): 22,
                }
            },
        ),
        (
            ["col1"],
            None,
            True,
            {
                "col1": {
                    frozenset(
                        {("measure", 0), ("col1", 0), ("sitetret", "trust"), ("pod", 0)}
                    ): 1,
                    frozenset(
                        {("col1", 1), ("measure", 0), ("sitetret", "trust"), ("pod", 0)}
                    ): 3,
                    frozenset(
                        {("col1", 0), ("measure", 1), ("sitetret", "trust"), ("pod", 0)}
                    ): 2,
                    frozenset(
                        {("col1", 1), ("measure", 1), ("sitetret", "trust"), ("pod", 0)}
                    ): 4,
                    frozenset(
                        {("pod", 1), ("measure", 0), ("col1", 0), ("sitetret", "trust")}
                    ): 5,
                    frozenset(
                        {("pod", 1), ("measure", 0), ("sitetret", "trust"), ("col1", 1)}
                    ): 7,
                    frozenset(
                        {("pod", 1), ("col1", 0), ("measure", 1), ("sitetret", "trust")}
                    ): 6,
                    frozenset(
                        {("pod", 1), ("measure", 1), ("sitetret", "trust"), ("col1", 1)}
                    ): 8,
                    frozenset(
                        {("pod", 2), ("measure", 0), ("col1", 0), ("sitetret", "trust")}
                    ): 9,
                    frozenset(
                        {("pod", 2), ("measure", 0), ("sitetret", "trust"), ("col1", 1)}
                    ): 11,
                    frozenset(
                        {("col1", 0), ("pod", 2), ("measure", 1), ("sitetret", "trust")}
                    ): 10,
                    frozenset(
                        {("pod", 2), ("measure", 1), ("sitetret", "trust"), ("col1", 1)}
                    ): 12,
                }
            },
        ),
        (
            ["col1", "col2"],
            None,
            False,
            {
                "col1+col2": {
                    frozenset(
                        {("col1", 0), ("pod", 0), ("sitetret", "trust"), ("col2", 0)}
                    ): 3,
                    frozenset(
                        {("col1", 1), ("pod", 0), ("sitetret", "trust"), ("col2", 0)}
                    ): 3,
                    frozenset(
                        {("col1", 1), ("pod", 0), ("sitetret", "trust"), ("col2", 1)}
                    ): 4,
                    frozenset(
                        {("col1", 0), ("col2", 0), ("sitetret", "trust"), ("pod", 1)}
                    ): 11,
                    frozenset(
                        {("col1", 1), ("col2", 0), ("sitetret", "trust"), ("pod", 1)}
                    ): 7,
                    frozenset(
                        {("col1", 1), ("sitetret", "trust"), ("pod", 1), ("col2", 1)}
                    ): 8,
                    frozenset(
                        {("col1", 0), ("sitetret", "trust"), ("col2", 0), ("pod", 2)}
                    ): 19,
                    frozenset(
                        {("col1", 1), ("sitetret", "trust"), ("col2", 0), ("pod", 2)}
                    ): 11,
                    frozenset(
                        {("col1", 1), ("sitetret", "trust"), ("pod", 2), ("col2", 1)}
                    ): 12,
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
    assert actual == expected


def test_go(mocker, mock_model):
    """test the go method"""
    # arrange
    mdl = mock_model
    mr_mock = Mock()
    mocker.patch("model.model.ModelRun", return_value=mr_mock)
    mr_mock.get_aggregate_results.return_value = "aggregate_results"

    # act
    actual = mdl.go(0)

    # assert
    assert actual == "aggregate_results"
