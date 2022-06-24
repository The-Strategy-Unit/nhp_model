"""test model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

import re
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
        "non-demographic_adjustment": {
            "a": {"a_a": [1, 1.2], "a_b": [1, 1.2]},
            "b": {"b_a": [1, 1.2], "b_b": [1, 1.2]},
        },
        "strategy_params": {
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


@pytest.fixture
def mock_run_params():
    """generate the expected run params"""
    return {
        "variant": ["a", "a", "b", "a"],
        "seeds": [1, 2, 3, 4],
        "health_status_adjustment": [0.9, 1, 2, 3],
        "waiting_list_adjustment": "waiting_list_adjustment",
        "non-demographic_adjustment": {
            "a": {"a_a": [1.1, 4, 5, 6], "a_b": [1.1, 7, 8, 9]},
            "b": {"b_a": [1.1, 10, 11, 12], "b_b": [1.1, 13, 14, 15]},
        },
        "strategy_params": {
            "admission_avoidance": {"a_a": [0.5, 16, 17, 18], "a_b": [0.5, 19, 20, 21]},
            "los_reduction": {"b_a": [0.5, 22, 23, 24], "b_b": [0.5, 25, 26, 27]},
        },
        "outpatient_factors": {
            "a": {"a_a": [0.5, 28, 29, 30], "a_b": [0.5, 31, 32, 33]},
            "b": {"b_a": [0.5, 34, 35, 36], "b_b": [0.5, 37, 38, 39]},
        },
        "aae_factors": {
            "a": {"a_a": [0.5, 40, 41, 42], "a_b": [0.5, 43, 44, 45]},
            "b": {"b_a": [0.5, 46, 47, 48], "b_b": [0.5, 49, 50, 51]},
        },
        "bed_occupancy": {
            "a": {"a": [0.5, 52, 53, 54], "b": [0.7, 0.7, 0.7, 0.7]},
            "b": {"a": [0.5, 55, 56, 57], "b": [0.8, 0.8, 0.8, 0.8]},
        },
    }


# __init__()


@pytest.mark.parametrize("model_type", ["aae", "ip", "op"])
def test_model_init_sets_values(mocker, model_type):
    """test model constructor works as expected"""
    params = {"input_data": "synthetic", "create_datetime": "20220101_012345"}

    mocker.patch("model.model.Model._load_parquet", return_value={"age": 1})
    mocker.patch("model.model.Model._load_demog_factors", return_value=None)
    mocker.patch("model.model.Model._generate_run_params", return_value=None)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("pickle.load", return_value="load_hsa")
    mocker.patch("os.path.join", lambda *args: "/".join(args))

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        mdl = Model(model_type, params, "data")
        assert mdl.model_type == model_type
        assert mdl.params == params
        assert mdl._data_path == "data/synthetic"
        mock_file.assert_called_with("data/synthetic/hsa_gams.pkl", "rb")
        assert mdl._hsa_gams == "load_hsa"
        assert mdl.data == {"age": 1, "age_group": "age_groups"}
        mdl._load_demog_factors.assert_called_once()
        mdl._generate_run_params.assert_called_once()


def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None)


def test_model_init_sets_create_datetime(mocker):
    """it sets the create_datetime item in params if not already set"""
    params = {"input_data": "synthetic"}

    mocker.patch("model.model.Model._load_parquet", return_value={"age": 1})
    mocker.patch("model.model.Model._load_demog_factors", return_value=None)
    mocker.patch("model.model.Model._generate_run_params", return_value=None)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("pickle.load", return_value="load_hsa")
    mocker.patch("os.path.join", lambda *args: "/".join(args))

    with patch("builtins.open", mock_open(read_data="data")):
        mdl = Model("aae", params, "data")
        assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


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

    assert mdl._demog_factors.equals(
        pd.DataFrame(
            {
                "a": [(i + diff) / i for i in a_range for _ in [0, 1]],
                "b": [(i + diff) / i for i in b_range for _ in [0, 1]],
                "rn": [i + j for i in range(1, 11) for j in [0, 10]],
            }
        ).set_index("rn")
    )

    assert mdl._variants == ["a", "b"]
    assert mdl._probabilities == [0.6, 0.4]


# _health_status_adjustment()


def test_health_status_adjustment(mock_model):
    """test that the health status adjustment factor is created correctly"""
    mdl = mock_model
    mdl.params["life_expectancy"] = {
        "f": [1.1, 1.0, 0.9],
        "m": [0.8, 0.7, 0.6],
        "min_age": 50,
        "max_age": 52,
    }
    mdl.data.age += 48
    run_params = {"health_status_adjustment": 0.8}
    expected = [
        # sex = 1, age = [49,53)
        1.0,
        49.36 / 50.0,
        50.44 / 51.0,
        51.52 / 52.0,
        1.0,
        # sex = 2, age = [49,53)
        1.0,
        49.12 / 50.0,
        50.20 / 51.0,
        51.28 / 52.0,
        1.0,
    ] * 2
    actual = mdl._health_status_adjustment(mdl.data, run_params)
    assert list(actual) == expected


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


# _factor_helper()


def test_factor_helper(mock_model):
    """test that _factor_helper returns the correct values"""
    mdl = mock_model
    actual = mdl._factor_helper(mdl.data, {"a": 0.9}, {"sex": 1}).tolist()
    expected = ([0.9, 1.0, 0.9, 1.0, 0.9] + [1.0] * 5) * 2
    assert actual == expected


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

    mocker.patch("model.model.inrange", wraps=lambda x, *args: x)
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
                "strategy_params": {
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
            },
        ),
        (
            2,
            {
                "variant": "b",
                "seed": 3,
                "health_status_adjustment": 2,
                "non-demographic_adjustment": {
                    "a": {"a_a": 5, "a_b": 8},
                    "b": {"b_a": 11, "b_b": 14},
                },
                "strategy_params": {
                    "admission_avoidance": {"a_a": 17, "a_b": 20},
                    "los_reduction": {"b_a": 23, "b_b": 26},
                },
                "outpatient_factors": {
                    "a": {"a_a": 29, "a_b": 32},
                    "b": {"b_a": 35, "b_b": 38},
                },
                "aae_factors": {
                    "a": {"a_a": 41, "a_b": 44},
                    "b": {"b_a": 47, "b_b": 50},
                },
                "bed_occupancy": {
                    "a": {"a": 53, "b": 0.7},
                    "b": {"a": 56, "b": 0.8},
                },
            },
        ),
    ],
)
def test_get_run_params(mock_model, mock_run_params, model_run, expected_run_params):
    """tests _get_run_params gets the right params for a model run"""
    mdl = mock_model
    mdl.run_params = mock_run_params
    assert mdl._get_run_params(model_run) == expected_run_params
