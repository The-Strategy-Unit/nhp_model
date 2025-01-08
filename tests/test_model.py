"""test model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring

import re
from unittest.mock import Mock, call, patch

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
        "covid_adjustment": [1.0, 1.2],
        "waiting_list_adjustment": {
            "ip": {"100": 1, "120": 2},
            "op": {"100": 3, "120": 4},
        },
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
        "health_status_adjustment": [1, 2, 3, 4, 5],
        "covid_adjustment": [1, 1, 2, 3],
        "waiting_list_adjustment": {
            "ip": {"100": [1] * 4, "120": [2] * 4},
            "op": {"100": [3] * 4, "120": [4] * 4},
        },
        "expat": {"ip": {"elective": {"Other": [1, 4, 5, 6]}}},
        "repat_local": {"ip": {"elective": {"Other": [1, 7, 8, 9]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1, 10, 11, 12]}}},
        "baseline_adjustment": {"ip": {"elective": {"Other": [1, 13, 14, 15]}}},
        "non-demographic_adjustment": {
            "a": {"a_a": [1, 16, 17, 18], "a_b": [1, 19, 20, 21]},
            "b": {"b_a": [1, 22, 23, 24], "b_b": [1, 25, 26, 27]},
        },
        "activity_avoidance": {
            "ip": {"a_a": [1, 28, 29, 30], "a_b": [1, 31, 32, 33]},
            "op": {"a_a": [1, 34, 35, 36], "a_b": [1, 37, 38, 39]},
            "aae": {"a_a": [1, 40, 41, 42], "a_b": [1, 43, 44, 45]},
        },
        "efficiencies": {
            "ip": {"b_a": [1, 46, 47, 48], "b_b": [1, 49, 50, 51]},
            "op": {"b_a": [1, 52, 53, 54], "b_b": [1, 55, 56, 57]},
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
    params = {
        "dataset": "synthetic",
        "start_year": "2020",
        "create_datetime": "20220101_012345",
    }

    mocker.patch("model.model.Model._load_data")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")
    lp_m = mocker.patch("model.model.load_params")
    hsa_m = mocker.patch("model.model.HealthStatusAdjustmentInterpolated")
    nhp_data_mock = Mock()
    nhp_data_mock.return_value = "nhp_data"

    # act
    mdl = Model(model_type, ["measures"], params, nhp_data_mock, "hsa", "run_params")

    # assert
    assert mdl.model_type == model_type
    assert mdl.params == params
    assert mdl._data_loader is None
    nhp_data_mock.assert_called_once_with("2020", "synthetic")
    mdl._load_data.assert_called_once()
    mdl._load_strategies.assert_called_once()
    mdl._load_demog_factors.assert_called_once()
    assert mdl.hsa == "hsa"
    mdl.generate_run_params.assert_not_called()
    assert mdl.run_params == "run_params"
    lp_m.assert_not_called()
    hsa_m.assert_not_called()


def test_model_init_calls_generate_run_params(mocker):
    """test model constructor works as expected"""
    # arrange
    params = {
        "dataset": "synthetic",
        "start_year": "2020",
        "create_datetime": "20220101_012345",
    }

    mocker.patch("model.model.Model._load_data")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params", return_value="generated")

    # act
    mdl = Model("aae", "arrivals", params, Mock(), "hsa")

    # assert
    mdl.generate_run_params.assert_called_once()
    assert mdl.run_params == "generated"


def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None, None)


def test_model_init_sets_create_datetime(mocker):
    """it sets the create_datetime item in params if not already set"""
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("model.model.Model._load_data")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")

    # act
    mdl = Model("aae", "arrivals", params, Mock(), "hsa", "run_params")

    # assert
    assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


def test_model_init_loads_params_if_string(mocker):
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("model.model.Model._load_data")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")
    lp_m = mocker.patch("model.model.load_params")
    lp_m.return_value = params

    # act
    mdl = Model("aae", "arrivals", "params_path", Mock(), "hsa")

    # assert
    lp_m.assert_called_once_with("params_path")
    assert mdl.params == params


def test_model_init_initialises_hsa_if_none(mocker):
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("model.model.Model._load_data")
    mocker.patch("model.model.Model._load_strategies")
    mocker.patch("model.model.Model._load_demog_factors")
    mocker.patch("model.model.Model.generate_run_params")
    hsa_m = mocker.patch("model.model.HealthStatusAdjustmentInterpolated")
    hsa_m.return_value = "hsa"

    # act
    mdl = Model("aae", "arrivals", params, Mock(return_value="nhp_data"))

    # assert
    hsa_m.assert_called_once_with("nhp_data", "2020")
    assert mdl.hsa == "hsa"


# add_ndggrp_to_data


def test_add_ndggrp_to_data(mock_model):
    # arrange
    mdl = mock_model
    mdl.data = pd.DataFrame({"group": ["a", "b", "c"]})

    # act
    mock_model._add_ndggrp_to_data()

    # assert
    assert mdl.data["ndggrp"].to_list() == ["a", "b", "c"]


# measures


def test_measures(mock_model):
    # arrange
    mdl = mock_model
    mdl._measures = ["x", "y"]

    # act
    actual = mdl.measures

    # assert
    assert actual == ["x", "y"]


# _load_data


def test_load_data(mocker, mock_model):
    # arrange
    mdl = mock_model
    mdl._measures = ["x", "y"]
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch(
        "model.model.Model.get_data_counts", return_value=np.array([[1, 2], [3, 4]])
    )
    mocker.patch("model.model.Model._add_pod_to_data")
    mocker.patch("model.model.Model._add_ndggrp_to_data")

    mdl._get_data = Mock(
        return_value=pd.DataFrame(
            {"rn": [2, 1], "age": [2, 1], "pod": ["a", "b"], "sitetret": ["c", "d"]}
        )
    )

    # act
    mdl._load_data()

    # assert
    assert mdl.data.to_dict(orient="list") == {
        "rn": [1, 2],
        "age": [1, 2],
        "pod": ["b", "a"],
        "sitetret": ["d", "c"],
        "age_group": ["age_groups"] * 2,
    }
    assert mdl.baseline_counts.tolist() == [[1, 2], [3, 4]]
    mdl.get_data_counts.call_args_list[0][0][0].equals(mdl.data)
    mdl._add_pod_to_data.assert_called_once_with()
    mdl._add_ndggrp_to_data.assert_called_once_with()

    assert mdl.baseline_step_counts.to_dict("list") == {
        "pod": ["a", "b"],
        "sitetret": ["c", "d"],
        "x": [2, 1],
        "y": [4, 3],
        "change_factor": ["baseline", "baseline"],
        "strategy": ["-", "-"],
    }


# _load_strategies()


def test_load_stratergies(mock_model):
    # arrange
    # act
    mock_model._load_strategies()
    # assert
    assert mock_model.strategies is None


# _load_demog_factors()


@pytest.mark.parametrize(
    "year, expected_demog, expected_birth",
    [
        (
            2018,
            np.arange(21, 41) / np.arange(1, 21),
            np.arange(21, 31) / np.arange(1, 11),
        ),
        (
            2019,
            np.arange(21, 41) / np.arange(11, 31),
            np.arange(21, 31) / np.arange(11, 21),
        ),
    ],
)
def test_demog_factors_loads_correctly(
    mock_model, year, expected_demog, expected_birth
):
    """test that the demographic factors are loaded correctly"""
    # arrange
    nhp_data_mock = Mock()
    nhp_data_mock.get_demographic_factors.return_value = pd.DataFrame(
        {
            "variant": ["a"] * 10 + ["b"] * 10,
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
            "2018": list(range(1, 21)),
            "2019": list(range(11, 31)),
            "2020": list(range(21, 41)),
        }
    )
    nhp_data_mock.get_birth_factors.return_value = pd.DataFrame(
        {
            "variant": ["a"] * 5 + ["b"] * 5,
            "age": list(range(1, 6)) * 2,
            "sex": [1] * 10,
            "2018": list(range(1, 11)),
            "2019": list(range(11, 21)),
            "2020": list(range(21, 31)),
        }
    )
    mdl = mock_model
    mdl._data_loader = nhp_data_mock
    mdl.params["start_year"] = year

    # act
    mdl._load_demog_factors()

    # assert
    assert np.equal(mdl.demog_factors["2020"], expected_demog).all()
    assert np.equal(mdl.birth_factors["2020"], expected_birth).all()

    nhp_data_mock.get_demographic_factors.assert_called_once_with()
    nhp_data_mock.get_birth_factors.assert_called_once_with()


# _generate_run_params()


def test_generate_run_params(mocker, mock_model, mock_run_params):
    """test that _generate_run_params returns the run parameters"""
    # arrange
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

    hsa_m = mocker.patch("model.model.HealthStatusAdjustment.generate_params")
    hsa_m.return_value = [1, 2, 3, 4, 5]

    mocker.patch("model.model.inrange", wraps=lambda x, low, high: x)

    mdl = mock_model
    mdl._variants = ["a", "b"]
    mdl._probabilities = [0.6, 0.4]

    # act
    actual = mdl.generate_run_params(mdl.params)

    # assert
    hsa_m.assert_called_once_with(2018, 2020, ["a", "a", "b", "a"], rng, 3)
    assert actual == mock_run_params


# _get_run_params()


@pytest.mark.parametrize(
    "model_run, expected_run_params",
    [
        (
            0,
            {
                "variant": "a",
                "health_status_adjustment": 1,
                "seed": 1,
                "covid_adjustment": 1,
                "non-demographic_adjustment": {
                    "a": {"a_a": 1, "a_b": 1},
                    "b": {"b_a": 1, "b_b": 1},
                },
                "expat": {"ip": {"elective": {"Other": 1}}},
                "repat_local": {"ip": {"elective": {"Other": 1}}},
                "repat_nonlocal": {"ip": {"elective": {"Other": 1}}},
                "baseline_adjustment": {"ip": {"elective": {"Other": 1}}},
                "activity_avoidance": {
                    "ip": {"a_a": 1, "a_b": 1},
                    "op": {"a_a": 1, "a_b": 1},
                    "aae": {"a_a": 1, "a_b": 1},
                },
                "efficiencies": {
                    "ip": {"b_a": 1, "b_b": 1},
                    "op": {"b_a": 1, "b_b": 1},
                },
                "waiting_list_adjustment": {
                    "ip": {"100": 1, "120": 2},
                    "op": {"100": 3, "120": 4},
                },
                "year": 2020,
            },
        ),
        (
            2,
            {
                "variant": "b",
                "seed": 3,
                "health_status_adjustment": 3,
                "covid_adjustment": 2,
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
                "waiting_list_adjustment": {
                    "ip": {"100": 1, "120": 2},
                    "op": {"100": 3, "120": 4},
                },
                "year": 2020,
            },
        ),
    ],
)
def test_get_run_params(
    mock_model,
    mock_run_params,
    model_run,
    expected_run_params,
):
    """tests _get_run_params gets the right params for a model run"""
    # arrange
    mdl = mock_model
    mdl.run_params = mock_run_params

    # act
    actual = mdl._get_run_params(model_run)
    # assert
    assert actual == expected_run_params


# get_data_counts


def test_get_data_counts(mock_model):
    with pytest.raises(NotImplementedError):
        mock_model.get_data_counts(None)


# activity_avoidance


def test_activity_avoidance_no_params(mock_model):
    # arrange
    mdl = mock_model
    mdl.strategies = {"activity_avoidance": None}

    mdl.model_type = "ip"

    mr_mock = Mock()
    mr_mock.run_params = {"activity_avoidance": {"ip": {}}}

    # act
    actual = mdl.activity_avoidance("data", mr_mock)

    # assert
    assert actual == ("data", None)


@pytest.mark.parametrize(
    "binomial_rv, expected_binomial_args, expected_factors",
    [
        (
            [1] * 9,
            {
                0: 0.01171875,
                1: 0.046875,
                2: 0.1171875,
                3: 1.0,
            },
            {
                "a": [0.125, 1.0, 1.0, 1.0],
                "b": [0.25, 0.25, 1.0, 1.0],
                "c": [0.375, 0.375, 0.375, 1.0],
                "d": [1.0, 0.5, 0.5, 1.0],
                "e": [1.0, 1.0, 0.625, 1.0],
            },
        ),
        ([0] * 9, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, {}),
    ],
)
def test_activity_avoidance(
    mock_model, binomial_rv, expected_binomial_args, expected_factors
):
    # arrange
    mdl = mock_model

    data = pd.DataFrame({"rn": [1, 2, 3, 4]})

    mdl.get_data_counts = Mock(return_value=np.array([2, 3, 4, 5]))
    mdl.apply_resampling = Mock(return_value="apply_resampling")

    mr_mock = Mock()
    mr_mock.rng.binomial.side_effect = [np.array(binomial_rv), np.array([1, 2, 3, 4])]

    mdl.strategies = {
        "activity_avoidance": pd.DataFrame(
            {
                "strategy": ["a", "b", "c"] + ["b", "c", "d"] + ["c", "d", "e"],
                "sample_rate": [0.5, 1.0, 1.0] + [1.0, 1.0, 1.0] + [1.0, 1.0, 0.5],
            },
            index=pd.Index([1, 1, 1, 2, 2, 2, 3, 3, 3], name="rn"),
        )
    }

    mdl.model_type = "ip"
    mr_mock.run_params = {
        "activity_avoidance": {
            "ip": {"a": 1 / 8, "b": 2 / 8, "c": 3 / 8, "d": 4 / 8, "e": 5 / 8}
        }
    }

    mr_mock.fix_step_counts.return_value = pd.DataFrame({"change_factor": [1]})

    # act
    actual_data, actual_step_counts = mdl.activity_avoidance(data, mr_mock)

    # assert
    assert actual_data == "apply_resampling"
    assert actual_step_counts.to_dict("list") == {
        "strategy": [1],
        "change_factor": ["activity_avoidance"],
    }

    assert mr_mock.rng.binomial.call_args_list[0][0][0] == 1
    assert mr_mock.rng.binomial.call_args_list[0][0][1].to_dict() == {
        1: 1.0,
        2: 1.0,
        3: 0.5,
    }

    assert mr_mock.rng.binomial.call_args_list[1][0][0].tolist() == [2, 3, 4, 5]
    assert (
        mr_mock.rng.binomial.call_args_list[1][0][1].to_dict() == expected_binomial_args
    )

    assert mr_mock.fix_step_counts.call_args[0][0].to_dict("list") == {
        "rn": [1, 2, 3, 4]
    }
    assert mr_mock.fix_step_counts.call_args[0][1].tolist() == [1, 2, 3, 4]
    assert mr_mock.fix_step_counts.call_args[0][2].to_dict("list") == expected_factors
    assert (
        mr_mock.fix_step_counts.call_args[0][3] == "activity_avoidance_interaction_term"
    )


# go


def test_go_save_full_model_results_false(mocker, mock_model):
    """test the go method"""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()

    mock_model.save_full_model_results = False
    mr_mock = Mock()
    mocker.patch("model.model.ModelIteration", return_value=mr_mock)
    mr_mock.get_aggregate_results.return_value = "aggregate_results"

    # act
    actual = mdl.go(0)

    # assert
    assert actual == "aggregate_results"
    mdl.save_results.assert_not_called()


def test_go_save_full_model_results_true(mocker, mock_model):
    """test the go method"""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()
    mdl.params["dataset"] = "synthetic"
    mdl.params["scenario"] = "test"
    mdl.params["create_datetime"] = "20240101_012345"

    makedirs_mock = mocker.patch("os.makedirs")

    mock_model.save_full_model_results = True
    mr_mock = Mock()
    mocker.patch("model.model.ModelIteration", return_value=mr_mock)
    mr_mock.get_aggregate_results.return_value = "aggregate_results"

    expected_path = "results/synthetic/test/20240101_012345/ip/model_run=1/"
    # act
    actual = mdl.go(1)

    # assert
    assert actual == "aggregate_results"
    mdl.save_results.assert_called()

    call_args = mdl.save_results.call_args[0]
    assert call_args[0] == mr_mock
    assert call_args[1]("ip") == expected_path
    makedirs_mock.assert_called_once_with(expected_path, exist_ok=True)


def test_go_save_full_model_results_true_baseline(mocker, mock_model):
    """test the go method"""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()
    mdl.params["dataset"] = "synthetic"
    mdl.params["id"] = "id"

    makedirs_mock = mocker.patch("os.makedirs")

    mock_model.save_full_model_results = True
    mr_mock = Mock()
    mocker.patch("model.model.ModelIteration", return_value=mr_mock)
    mr_mock.get_aggregate_results.return_value = "aggregate_results"

    # act
    actual = mdl.go(0)

    # assert
    assert actual == "aggregate_results"

    mdl.save_results.assert_not_called()
    makedirs_mock.assert_not_called()


# get_agg


@pytest.mark.parametrize(
    "results, cols, expected",
    [
        (
            pd.DataFrame(
                {
                    "pod": ["a"] * 4 + ["b"] * 4,
                    "sitetret": [i for i in ["c", "d"] for _ in [0, 1]] * 2,
                    "measure": ["e", "f"] * 4,
                    "value": range(8),
                }
            ),
            [],
            {
                r: i
                for (i, r) in enumerate(
                    [
                        (i, j, k)
                        for i in ["a", "b"]
                        for j in ["c", "d"]
                        for k in ["e", "f"]
                    ]
                )
            },
        ),
        (
            pd.DataFrame(
                {
                    "pod": (["a"] * 4 + ["b"] * 4) * 2,
                    "sitetret": [i for i in ["c", "d"] for _ in [0, 1]] * 4,
                    "measure": ["e", "f"] * 8,
                    "x": ["x"] * 8 + ["y"] * 8,
                    "value": range(16),
                }
            ),
            ["x"],
            {
                r: i
                for (i, r) in enumerate(
                    [
                        (i, j, x, k)
                        for x in ["x", "y"]
                        for i in ["a", "b"]
                        for j in ["c", "d"]
                        for k in ["e", "f"]
                    ]
                )
            },
        ),
    ],
)
def test_get_agg(mock_model, results, cols, expected):
    # arrange
    mdl = mock_model

    # act
    actual = mdl.get_agg(results, *cols)

    # assert
    assert actual.to_dict() == expected
