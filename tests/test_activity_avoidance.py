"""test activity avoidance"""
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
    mdl._model_run = Mock()
    return mdl


def test_init(mocker):
    # arrange
    model_run = Mock()
    model_run.data = "data"
    model_run.model.model_type = "ip"
    model_run.params = "params"
    model_run.run_params = "run_params"
    model_run.step_counts = "step_counts"
    model_run.model.demog_factors = "demog_factors"
    model_run.model.hsa_gams = "hsa_gams"
    model_run.model.strategies = {"activity_avoidance": "activity_avoidance"}

    usc_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update_step_counts"
    )

    # act
    actual = ActivityAvoidance(model_run, np.array([1]), np.array([2]))

    # assert
    assert actual.data == "data"
    assert actual._activity_type == "ip"
    assert actual.params == "params"
    assert actual.run_params == "run_params"
    assert actual.step_counts == "step_counts"
    assert actual.demog_factors == "demog_factors"
    assert actual.hsa_gams == "hsa_gams"
    assert actual.strategies == "activity_avoidance"
    usc_mock.assert_called_once_with(("baseline", "-"))
    assert actual._row_counts.tolist() == [1]
    assert actual._row_mask.tolist() == [2]


@pytest.mark.parametrize("group, expected", [(None, ("f", "-")), ("a", ("a", "f"))])
def test_update(mocker, mock_activity_avoidance, group, expected):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = np.array([3.0, 4.0, 5.0, 6.0])
    aa_mock._model_run.data = pd.DataFrame({"a": [1, 2, 3, 4]})

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
    aa_mock._model_run.run_params = {"variant": "a"}
    aa_mock._model_run.model.demog_factors = {"a": "f"}

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

    aa_mock._model_run.run_params = {"health_status_adjustment": 2}

    activity_age = 1 / (np.arange(0, 12) + 1)

    aa_mock._model_run.model.hsa_precomputed_activity_ages = pd.DataFrame(
        {
            "hsagrp": (["a"] * 3 + ["b"] * 3) * 2,
            "sex": [1] * 6 + [2] * 6,
            "age": [50, 51, 52] * 4,
            "life_expectancy": [1, 2, 3] * 2 + [-1, -2, -3] * 2,
            "activity_age": activity_age,
        }
    ).set_index(["hsagrp", "sex", "age"])

    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x.to_numpy()})
    aa_mock._model_run.model.hsa_gams = {
        (i, j): hsa_mock for i in ["a", "b"] for j in [1, 2]
    }

    expected = {
        ("a", 1, 50): 48.0,
        ("a", 1, 51): 94.0,
        ("a", 1, 52): 138.0,
        ("a", 2, 50): 364.0,
        ("a", 2, 51): 440.0,
        ("a", 2, 52): 522.0,
        ("b", 1, 50): 192.0,
        ("b", 1, 51): 235.0,
        ("b", 1, 52): 276.0,
        ("b", 2, 50): 520.0,
        ("b", 2, 51): 605.0,
        ("b", 2, 52): 696.0,
    }

    keys = list(zip(["a"] * 6 + ["b"] * 6, ([1] * 3 + [2] * 3) * 2, [50, 51, 52] * 4))

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.health_status_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == expected
    assert f.name == "health_status_adjustment"
    assert cols == ["hsagrp", "sex", "age"]


def test_expat_adjustment_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {"expat": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.expat_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_expat_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
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


def test_repat_adjustment_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
        "repat_local": {"ip": {}},
        "repat_nonlocal": {"ip": {}},
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.repat_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_repat_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
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


def test_baseline_adjustment_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {"baseline_adjustment": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.baseline_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_baseline_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
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


def test_waiting_list_adjustment_aae(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "aae"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
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
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_waiting_list_adjustment_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {"waiting_list_adjustment": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.waiting_list_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_waiting_list_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
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


def test_non_demographic_adjustment_not_ip(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "op"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
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
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_non_demographic_adjustment_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {"non-demographic_adjustment": {}}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.non_demographic_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_non_demographic_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
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


def test_activity_avoidance_no_params(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run = Mock()
    aa_mock._model_run.model.strategies = {"activity_avoidance": None}

    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {"activity_avoidance": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )
    # act
    actual = aa_mock.activity_avoidance()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


@pytest.mark.parametrize(
    "binomial_rv, expected",
    [
        (
            [1] * 9,
            [
                ([1 / 8] * 1, "a"),
                ([2 / 8] * 2, "b"),
                ([3 / 8] * 3, "c"),
                ([4 / 8] * 2, "d"),
                ([5 / 8] * 1, "e"),
            ],
        ),
        (
            [0] * 9,
            [
                ([1] * 1, "a"),
                ([1] * 2, "b"),
                ([1] * 3, "c"),
                ([1] * 2, "d"),
                ([1] * 1, "e"),
            ],
        ),
    ],
)
def test_activity_avoidance(mocker, mock_activity_avoidance, binomial_rv, expected):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._model_run = Mock()
    aa_mock._model_run.rng.binomial.return_value = binomial_rv

    aa_mock._model_run.model.strategies = {
        "activity_avoidance": pd.DataFrame(
            {
                "strategy": ["a", "b", "c"] + ["b", "c", "d"] + ["c", "d", "e"],
                "sample_rate": [0.5, 1.0, 1.0] + [1.0, 1.0, 1.0] + [1.0, 1.0, 0.5],
            },
            index=pd.Index([1, 1, 1, 2, 2, 2, 3, 3, 3], name="rn"),
        )
    }

    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
        "activity_avoidance": {
            "ip": {"a": 1 / 8, "b": 2 / 8, "c": 3 / 8, "d": 4 / 8, "e": 5 / 8}
        }
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update_rn",
        return_value="update_rn",
    )

    # act
    actual = aa_mock.activity_avoidance()
    call_args = [(i[0][0].to_list(), i[0][1]) for i in u_mock.call_args_list]

    # assert
    assert actual == aa_mock
    call_args == expected


def test_apply_resampling(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = "row_counts"
    mr = aa_mock._model_run

    mr.data = "data"
    mr.rng.poisson.return_value = "poisson"
    mr.model.apply_resampling.return_value = ("data", np.array([[30.0, 50.0]]))

    aa_mock.step_counts = {
        ("baseline", "-"): np.array([20.0, 30.0]),
        ("a", "-"): np.array([5.0, 10.0]),  # 25, 40
        ("b", "-"): np.array([10.0, 5.0]),  # 45, 45
        ("c", "-"): np.array([-10.0, -5.0]),  # 25, 40
    }

    expected = {
        ("baseline", "-"): [20.0, 30.0],
        ("a", "-"): [60.0, 50.0],
        ("b", "-"): [120.0, 25.0],
        ("c", "-"): [-120.0, -25.0],
    }

    # act
    aa_mock.apply_resampling()

    # assert
    mr.rng.poisson.assert_called_once_with("row_counts")
    mr.model.apply_resampling.assert_called_once_with("poisson", "data")

    assert mr.data == "data"

    assert {k: v.tolist() for k, v in aa_mock.step_counts.items()} == expected
