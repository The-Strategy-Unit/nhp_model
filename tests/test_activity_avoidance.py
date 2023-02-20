"""test model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from model.activity_avoidance import ActivityAvoidance


# fixtures
@pytest.fixture
def mock_activity_avoidance():
    """create a mock Model instance"""
    with patch.object(ActivityAvoidance, "__init__", lambda m, c, r: None):
        mdl = ActivityAvoidance(None, None)

    return mdl


def test_init(mocker):
    # arrange
    model_run = Mock()
    model_run.data = "data"
    model_run._model = Mock()
    model_run._model.model_type = "ip"
    model_run.params = "params"
    model_run.run_params = "run_params"
    model_run.step_counts = "step_counts"
    model_run._model._demog_factors = "demog_factors"
    model_run._model._hsa_gams = "hsa_gams"
    model_run._model._strategies = {"activity_avoidance": "activity_avoidance"}

    usc_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update_step_counts"
    )

    # act
    actual = ActivityAvoidance(model_run, np.array([1]))

    # assert
    assert actual.data == "data"
    assert actual._activity_type == "ip"
    assert actual.params == "params"
    assert actual.run_params == "run_params"
    assert actual.step_counts == "step_counts"
    assert actual._demog_factors == "demog_factors"
    assert actual._hsa_gams == "hsa_gams"
    assert actual._strategies == "activity_avoidance"
    usc_mock.assert_called_once_with(("baseline", "-"))


@pytest.mark.parametrize("group, expected", [(None, ("f", "-")), ("a", ("a", "f"))])
def test_update(mocker, mock_activity_avoidance, group, expected):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = np.array([3.0, 4.0, 5.0, 6.0])
    aa_mock.data = pd.DataFrame({"a": [1, 2, 3, 4]})

    factor = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3], name="f")

    usc_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update_step_counts"
    )

    # act
    actual = aa_mock._update(factor, ["a"], group)

    # assert
    usc_mock.assert_called_once_with(expected)
    assert aa_mock._row_counts.tolist() == [3, 8, 15, 6]
    assert actual == aa_mock


def test_update_step_counts(mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._sum = np.array([3.0, 0.0])
    aa_mock._row_counts = np.array([[3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0]])
    aa_mock._row_mask = np.array([1.0, 1.0, 1.0, 0.0])
    aa_mock.step_counts = {}

    # act
    aa_mock._update_step_counts(("a", "-"))

    # assert
    assert aa_mock.step_counts[("a", "-")].tolist() == [9.0, 6.0]
    assert aa_mock._sum.tolist() == [12.0, 6.0]


def test_demographic_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock.run_params = {"variant": "a"}
    aa_mock._demog_factors = {"a": "f"}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.demographic_adjustment()

    # assert
    assert actual == "update"
    u_mock.assert_called_once_with("f", ["age", "sex"])


# _health_status_adjustment()


def test_health_status_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock.params = {
        "life_expectancy": {
            "f": [1.1, 1.0, 0.9],
            "m": [0.8, 0.7, 0.6],
            "min_age": 50,
            "max_age": 52,
        }
    }
    aa_mock.run_params = {"health_status_adjustment": 0.8}

    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x.to_numpy()})
    aa_mock._hsa_gams = {(i, j): hsa_mock for i in ["a", "b"] for j in [1, 2]}

    expected = [
        # sex = 1, age = [49,53)
        49.36 / 50.0,
        50.44 / 51.0,
        51.52 / 52.0,
        # sex = 2, age = [49,53)
        49.12 / 50.0,
        50.20 / 51.0,
        51.28 / 52.0,
    ] * 2

    keys = list(zip(["a"] * 6 + ["b"] * 6, ([1] * 3 + [2] * 3) * 2, [50, 51, 52] * 4))

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.health_status_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {k: v for k, v in zip(keys, expected)}
    assert f.name == "health_status_adjustment"
    assert cols == ["hsagrp", "sex", "age"]


def test_expat_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._activity_type = "ip"
    aa_mock.run_params = {
        "expat": {
            "ip": {
                "elective": {"100": 1, "200": 2},
                "non-elective": {"300": 3, "400": 4},
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.expat_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {
        ("elective", "100"): 1,
        ("elective", "200"): 2,
        ("non-elective", "300"): 3,
        ("non-elective", "400"): 4,
    }
    assert f.name == "expat"
    assert cols == ["group", "tretspef"]


def test_repat_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._activity_type = "ip"
    aa_mock.run_params = {
        "repat_local": {
            "ip": {
                "elective": {"100": 1, "200": 2},
                "non-elective": {"300": 3, "400": 4},
            }
        },
        "repat_nonlocal": {
            "ip": {
                "elective": {"100": 5, "200": 6},
                "non-elective": {"300": 7, "400": 8},
            }
        },
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.repat_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {
        (1, "elective", "100"): 1,
        (1, "elective", "200"): 2,
        (1, "non-elective", "300"): 3,
        (1, "non-elective", "400"): 4,
        (0, "elective", "100"): 5,
        (0, "elective", "200"): 6,
        (0, "non-elective", "300"): 7,
        (0, "non-elective", "400"): 8,
    }
    assert f.name == "repat"
    assert cols == ["is_main_icb", "group", "tretspef"]


def test_baseline_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._activity_type = "ip"
    aa_mock.run_params = {
        "baseline_adjustment": {
            "ip": {
                "elective": {"100": 1, "200": 2},
                "non-elective": {"300": 3, "400": 4},
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.baseline_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {
        ("elective", "100"): 1,
        ("elective", "200"): 2,
        ("non-elective", "300"): 3,
        ("non-elective", "400"): 4,
    }
    assert f.name == "baseline_adjustment"
    assert cols == ["group", "tretspef"]


def test_waiting_list_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._activity_type = "ip"

    aa_mock.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock.run_params = {
        "waiting_list_adjustment": {
            "ip": {
                "100": 1,
                "200": 4,
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.waiting_list_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {
        (True, "100"): 1 + 1 / 4,
        (True, "200"): 1 + 4 / 2,
    }
    assert f.name == "waiting_list_adjustment"
    assert cols == ["is_wla", "tretspef"]


def test_non_demographic_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._activity_type = "ip"

    aa_mock.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock.run_params = {
        "non-demographic_adjustment": {
            "elective": {
                "a": 1,
                "b": 2,
            },
            "non-elective": {
                "a": 3,
                "b": 4,
            },
        }
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.non_demographic_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {
        ("a", "elective"): 1,
        ("b", "elective"): 2,
        ("a", "non-elective"): 3,
        ("b", "non-elective"): 4,
    }
    assert f.name == "non-demographic_adjustment"
    assert cols == ["age_group", "group"]


def test_activity_avoidance():
    assert False


def test_apply_resampling():
    assert False
