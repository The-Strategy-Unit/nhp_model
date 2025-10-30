"""Test model."""

import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from nhp.model.model import Model


# fixtures
@pytest.fixture
def mock_model():
    """Create a mock Model instance."""
    with patch.object(Model, "__init__", lambda s, m, p, d, c: None):
        mdl = Model(None, None, None, None)  # type: ignore
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
        "waiting_list_adjustment": {
            "ip": {"100": 1, "120": 2},
            "op": {"100": 3, "120": 4},
        },
        "expat": {"ip": {"elective": {"Other": [0.7, 0.9]}}},
        "repat_local": {"ip": {"elective": {"Other": [1.0, 1.2]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1.3, 1.5]}}},
        "baseline_adjustment": {"ip": {"elective": {"Other": [1.5, 1.7]}}},
        "inequalities": {"level_up": ["HRG1"], "level_down": ["HRG2"], "zero_sum": []},
        "non-demographic_adjustment": {
            "variant": "ndg-test",
            "variant-type": "year-on-year-growth",
            "values": {
                "a": {"a_a": [1, 1.2], "a_b": [1, 1.2]},
                "b": {"b_a": [1, 1.2], "b_b": [1, 1.2]},
            },
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
    """Generate the expected run params."""
    return {
        "variant": ["a", "a", "b", "a"],
        "seeds": [1, 2, 3, 4],
        "health_status_adjustment": [1, 2, 3, 4, 5],
        "non-demographic_adjustment": {
            "a": {"a_a": [1, 1, 2, 3], "a_b": [1, 4, 5, 6]},
            "b": {"b_a": [1, 7, 8, 9], "b_b": [1, 10, 11, 12]},
        },
        "waiting_list_adjustment": {
            "ip": {"100": [1, 1, 1, 1], "120": [2, 2, 2, 2]},
            "op": {"100": [3, 3, 3, 3], "120": [4, 4, 4, 4]},
        },
        "expat": {"ip": {"elective": {"Other": [1, 13, 14, 15]}}},
        "repat_local": {"ip": {"elective": {"Other": [1, 16, 17, 18]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1, 19, 20, 21]}}},
        "baseline_adjustment": {"ip": {"elective": {"Other": [1, 22, 23, 24]}}},
        "activity_avoidance": {
            "ip": {"a_a": [1, 25, 26, 27], "a_b": [1, 28, 29, 30]},
            "op": {"a_a": [1, 31, 32, 33], "a_b": [1, 34, 35, 36]},
            "aae": {"a_a": [1, 37, 38, 39], "a_b": [1, 40, 41, 42]},
        },
        "efficiencies": {
            "ip": {"b_a": [1, 43, 44, 45], "b_b": [1, 46, 47, 48]},
            "op": {"b_a": [1, 49, 50, 51], "b_b": [1, 52, 53, 54]},
        },
    }


@pytest.fixture
def mock_model_results():
    """Creates mock of model results object."""
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
    """Test model constructor works as expected."""
    # arrange
    params = {
        "dataset": "synthetic",
        "start_year": "2020",
        "create_datetime": "20220101_012345",
    }

    mocker.patch("nhp.model.model.Model._load_data")
    mocker.patch("nhp.model.model.Model._load_strategies")
    mocker.patch("nhp.model.model.Model._load_demog_factors")
    mocker.patch("nhp.model.model.Model._load_inequalities_factors")
    mocker.patch("nhp.model.model.Model.generate_run_params")
    lp_m = mocker.patch("nhp.model.model.load_params")
    hsa_m = mocker.patch("nhp.model.model.HealthStatusAdjustmentInterpolated")
    data_mock = Mock()
    data_mock.return_value = "data_loader"

    # act
    mdl = Model(model_type, ["measures"], params, data_mock, "hsa", "run_params")  # type: ignore

    # assert
    assert mdl.model_type == model_type
    assert mdl.params == params
    data_mock.assert_called_once_with("2020", "synthetic")
    mdl._load_data.assert_called_once_with("data_loader")  # type: ignore
    mdl._load_strategies.assert_called_once_with("data_loader")  # type: ignore
    mdl._load_demog_factors.assert_called_once_with("data_loader")  # type: ignore
    mdl._load_inequalities_factors.assert_called_once_with("data_loader")  # type: ignore
    assert mdl.hsa == "hsa"
    mdl.generate_run_params.assert_not_called()  # type: ignore
    assert mdl.run_params == "run_params"
    lp_m.assert_not_called()
    hsa_m.assert_not_called()


def test_model_init_calls_generate_run_params(mocker):
    """Test model constructor works as expected."""
    # arrange
    params = {
        "dataset": "synthetic",
        "start_year": "2020",
        "create_datetime": "20220101_012345",
    }

    mocker.patch("nhp.model.model.Model._load_data")
    mocker.patch("nhp.model.model.Model._load_strategies")
    mocker.patch("nhp.model.model.Model._load_demog_factors")
    mocker.patch("nhp.model.model.Model._load_inequalities_factors")
    mocker.patch("nhp.model.model.Model.generate_run_params", return_value="generated")

    # act
    mdl = Model("aae", "arrivals", params, Mock(), "hsa")  # type: ignore

    # assert
    mdl.generate_run_params.assert_called_once()  # type: ignore
    assert mdl.run_params == "generated"


def test_model_init_validates_model_type():
    """It raises an exception if an invalid model_type is passed."""
    with pytest.raises(AssertionError):
        Model("", None, None, None)  # type: ignore


def test_model_init_sets_create_datetime(mocker):
    """It sets the create_datetime item in params if not already set."""
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("nhp.model.model.Model._load_data")
    mocker.patch("nhp.model.model.Model._load_strategies")
    mocker.patch("nhp.model.model.Model._load_demog_factors")
    mocker.patch("nhp.model.model.Model._load_inequalities_factors")
    mocker.patch("nhp.model.model.Model.generate_run_params")

    # act
    mdl = Model("aae", "arrivals", params, Mock(), "hsa", "run_params")  # type: ignore

    # assert
    assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


def test_model_init_loads_params_if_string(mocker):
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("nhp.model.model.Model._load_data")
    mocker.patch("nhp.model.model.Model._load_strategies")
    mocker.patch("nhp.model.model.Model._load_demog_factors")
    mocker.patch("nhp.model.model.Model._load_inequalities_factors")
    mocker.patch("nhp.model.model.Model.generate_run_params")
    lp_m = mocker.patch("nhp.model.model.load_params")
    lp_m.return_value = params

    # act
    mdl = Model("aae", "arrivals", "params_path", Mock(), "hsa")  # type: ignore

    # assert
    lp_m.assert_called_once_with("params_path")
    assert mdl.params == params


def test_model_init_initialises_hsa_if_none(mocker):
    # arrange
    params = {"dataset": "synthetic", "start_year": "2020"}

    mocker.patch("nhp.model.model.Model._load_data")
    mocker.patch("nhp.model.model.Model._load_strategies")
    mocker.patch("nhp.model.model.Model._load_demog_factors")
    mocker.patch("nhp.model.model.Model._load_inequalities_factors")
    mocker.patch("nhp.model.model.Model.generate_run_params")
    hsa_m = mocker.patch("nhp.model.model.HealthStatusAdjustmentInterpolated")
    hsa_m.return_value = "hsa"

    # act
    mdl = Model("aae", "arrivals", params, Mock(return_value="data"))  # type: ignore

    # assert
    hsa_m.assert_called_once_with("data", "2020")
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


# _get_data


def test_get_data(mock_model):
    mdl = mock_model
    with pytest.raises(NotImplementedError):
        mdl._get_data(Mock())


# _load_data


def test_load_data(mocker, mock_model):
    # arrange
    mdl = mock_model
    mdl._measures = ["x", "y"]
    mocker.patch("nhp.model.model.Model.get_data_counts", return_value=np.array([[1, 2], [3, 4]]))
    data_loader = Mock()

    data = {"rn": [1, 2], "age": [1, 2], "pod": ["a", "b"], "sitetret": ["c", "d"]}

    mdl._get_data = Mock(return_value=pd.DataFrame(data))

    # act
    mdl._load_data(data_loader)

    # assert
    assert mdl.data.to_dict(orient="list") == data
    assert mdl.baseline_counts.tolist() == [[1, 2], [3, 4]]
    mdl.get_data_counts.call_args_list[0][0][0].equals(mdl.data)

    assert mdl.baseline_step_counts.to_dict("list") == {
        "pod": ["a", "b"],
        "sitetret": ["c", "d"],
        "x": [1, 2],
        "y": [3, 4],
        "change_factor": ["baseline", "baseline"],
        "strategy": ["-", "-"],
    }

    mdl._get_data.assert_called_once_with(data_loader)


def test_baseline_step_counts(mock_model):
    # arrange
    mock_model._baseline_step_counts = "test_baseline_step_counts"

    # act
    actual = mock_model.baseline_step_counts

    # assert
    assert actual == "test_baseline_step_counts"


# _load_strategies()


def test_load_strategies(mock_model):
    # arrange
    # act
    mock_model._load_strategies(None)
    # assert
    assert not mock_model.strategies


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
def test_demog_factors_loads_correctly(mock_model, year, expected_demog, expected_birth):
    """Test that the demographic factors are loaded correctly."""
    # arrange
    data_loader = Mock()
    data_loader.get_demographic_factors.return_value = pd.DataFrame(
        {
            "variant": ["a"] * 10 + ["b"] * 10,
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
            "2018": list(range(1, 21)),
            "2019": list(range(11, 31)),
            "2020": list(range(21, 41)),
        }
    )
    data_loader.get_birth_factors.return_value = pd.DataFrame(
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
    mdl.params["start_year"] = year

    # act
    mdl._load_demog_factors(data_loader)

    # assert
    assert np.equal(mdl.demog_factors["2020"], expected_demog).all()
    assert np.equal(mdl.birth_factors["2020"], expected_birth).all()

    data_loader.get_demographic_factors.assert_called_once_with()
    data_loader.get_birth_factors.assert_called_once_with()


# _generate_run_params()


def test_generate_run_params(mocker, mock_model, mock_run_params):
    """Test that _generate_run_params returns the run parameters."""
    # arrange
    n = 0

    def get_next_n(*args):
        nonlocal n
        n += 1  # type: ignore
        return n

    rng = Mock()
    rng.choice.return_value = np.array(["a", "b", "a"])
    rng.integers.return_value = np.array([1, 2, 3, 4])
    rng.normal = Mock(wraps=get_next_n)
    mocker.patch("numpy.random.default_rng", return_value=rng)

    hsa_m = mocker.patch("nhp.model.model.HealthStatusAdjustment.generate_params")
    hsa_m.return_value = [1, 2, 3, 4, 5]

    mocker.patch("nhp.model.model.inrange", wraps=lambda x, low, high: x)

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
                "non-demographic_adjustment": {
                    "a": {"a_a": 2, "a_b": 5},
                    "b": {"b_a": 8, "b_b": 11},
                },
                "waiting_list_adjustment": {
                    "ip": {"100": 1, "120": 2},
                    "op": {"100": 3, "120": 4},
                },
                "expat": {"ip": {"elective": {"Other": 14}}},
                "repat_local": {"ip": {"elective": {"Other": 17}}},
                "repat_nonlocal": {"ip": {"elective": {"Other": 20}}},
                "baseline_adjustment": {"ip": {"elective": {"Other": 23}}},
                "activity_avoidance": {
                    "ip": {"a_a": 26, "a_b": 29},
                    "op": {"a_a": 32, "a_b": 35},
                    "aae": {"a_a": 38, "a_b": 41},
                },
                "efficiencies": {
                    "ip": {"b_a": 44, "b_b": 47},
                    "op": {"b_a": 50, "b_b": 53},
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
    """Tests _get_run_params gets the right params for a model run."""
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
def test_activity_avoidance(mock_model, binomial_rv, expected_binomial_args, expected_factors):
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
        "activity_avoidance": {"ip": {"a": 1 / 8, "b": 2 / 8, "c": 3 / 8, "d": 4 / 8, "e": 5 / 8}}
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
    assert mr_mock.rng.binomial.call_args_list[1][0][1].to_dict() == expected_binomial_args

    assert mr_mock.fix_step_counts.call_args[0][0].to_dict("list") == {"rn": [1, 2, 3, 4]}
    assert mr_mock.fix_step_counts.call_args[0][1].tolist() == [1, 2, 3, 4]
    assert mr_mock.fix_step_counts.call_args[0][2].to_dict("list") == expected_factors
    assert mr_mock.fix_step_counts.call_args[0][3] == "activity_avoidance_interaction_term"


# calculate_avoided_activity


def test_calculate_avoided_activity(mock_model):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_model.calculate_avoided_activity(None, None)


# go


def test_go_save_full_model_results_false(mocker, mock_model):
    """Test the go method."""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()

    mock_model.save_full_model_results = False
    mr_mock = Mock()
    mocker.patch("nhp.model.model.ModelIteration", return_value=mr_mock)
    mr_mock.get_aggregate_results.return_value = "aggregate_results"

    # act
    actual = mdl.go(0)

    # assert
    assert actual == "aggregate_results"
    mdl.save_results.assert_not_called()


def test_go_save_full_model_results_true(mocker, mock_model):
    """Test the go method."""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()
    mdl.params["dataset"] = "synthetic"
    mdl.params["scenario"] = "test"
    mdl.params["create_datetime"] = "20240101_012345"

    makedirs_mock = mocker.patch("os.makedirs")

    mock_model.save_full_model_results = True
    mr_mock = Mock()
    mocker.patch("nhp.model.model.ModelIteration", return_value=mr_mock)
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
    """Test the go method."""
    # arrange
    mdl = mock_model
    mdl.save_results = Mock()
    mdl.params["dataset"] = "synthetic"
    mdl.params["id"] = "id"

    makedirs_mock = mocker.patch("os.makedirs")

    mock_model.save_full_model_results = True
    mr_mock = Mock()
    mocker.patch("nhp.model.model.ModelIteration", return_value=mr_mock)
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
                    [(i, j, k) for i in ["a", "b"] for j in ["c", "d"] for k in ["e", "f"]]
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


def test_apply_resampling(mock_model):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_model.apply_resampling(None, None)


def test_efficiencies(mock_model):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_model.efficiencies(None, None)


def test_aggregate(mock_model):
    # arrange
    mdl = mock_model
    mdl.process_results = Mock(return_value="processed_results")
    mdl.get_agg = Mock(return_value="agg")
    mdl.specific_aggregations = Mock(return_value={"1": "agg", "2": "agg"})

    mi_mock = Mock()
    mi_mock.get_model_results.return_value = "results"

    # act
    actual = mdl.aggregate(mi_mock)

    # assert
    mi_mock.get_model_results.assert_called()
    mdl.process_results.assert_called_once_with("results")
    assert mdl.get_agg.call_args_list == [
        call("processed_results"),
        call("processed_results", "sex", "age_group"),
        call("processed_results", "age"),
    ]
    mdl.specific_aggregations.assert_called_once_with("processed_results")

    assert actual == {
        "default": "agg",
        "sex+age_group": "agg",
        "age": "agg",
        "1": "agg",
        "2": "agg",
    }


def test_load_inequalities_factors(mock_model):
    # arrange
    mdl = mock_model
    data_loader = Mock()
    data_loader.get_inequalities.return_value = pd.DataFrame(
        {
            "icb": ["ICB"] * 10,
            "sushrg_trimmed": ["HRG1"] * 5 + ["HRG2"] * 5,
            "imd_quintile": list(range(1, 6)) * 2,
            "activity_rate": [0.1] * 10,
            "fitted_line": [0.2] * 10,
            "level_up": [1] * 10,
            "level_down": [2] * 10,
            "zero_sum": [3] * 10,
        }
    )
    expected = pd.DataFrame(
        {
            "icb": ["ICB"] * 10,
            "sushrg_trimmed": ["HRG1"] * 5 + ["HRG2"] * 5,
            "imd_quintile": list(range(1, 6)) * 2,
            "factor": [1] * 5 + [2] * 5,
        }
    )
    # act
    mdl._load_inequalities_factors(data_loader)
    # assert
    pd.testing.assert_frame_equal(mdl.inequalities_factors, expected)


def test_load_inequalities_factors_when_empty(mock_model):
    # arrange
    mdl = mock_model
    mdl.params["inequalities"] = {}
    data_loader = Mock()
    data_loader.get_inequalities.return_value = pd.DataFrame(
        {
            "icb": ["ICB"] * 10,
            "sushrg_trimmed": ["HRG1"] * 5 + ["HRG2"] * 5,
            "imd_quintile": list(range(1, 6)) * 2,
            "activity_rate": [0.1] * 10,
            "fitted_line": [0.2] * 10,
            "level_up": [1] * 10,
            "level_down": [2] * 10,
            "zero_sum": [3] * 10,
        }
    )
    expected = pd.DataFrame([], columns=["icb", "sushrg_trimmed", "imd_quintile", "factor"])
    # act
    mdl._load_inequalities_factors(data_loader)
    # assert
    assert len(mdl.inequalities_factors) == 0


def test_process_results(mock_model):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_model.process_results(None)


def test_specific_aggregations(mock_model):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_model.specific_aggregations(None)
