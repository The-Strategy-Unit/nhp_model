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
    with patch.object(ActivityAvoidance, "__init__", lambda m, c: None):
        mdl = ActivityAvoidance(None)
    mdl._model_run = Mock()
    mdl._baseline_counts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(float)
    mdl.step_counts = {}
    return mdl


def test_init():
    # arrange
    model_run = Mock()
    model_run.data = "data"
    model_run.model.model_type = "ip"
    model_run.model.baseline_counts = np.array([[1, 2, 3], [4, 5, 6]])
    model_run.model.data_mask = [2]
    model_run.params = "params"
    model_run.run_params = "run_params"
    model_run.step_counts = {}
    model_run.model.demog_factors = "demog_factors"
    model_run.model.birth_factors = "birth_factors"
    model_run.model.hsa = "hsa"
    model_run.model.strategies = {"activity_avoidance": "activity_avoidance"}

    # act
    actual = ActivityAvoidance(model_run)

    # assert
    assert actual.data == "data"
    assert actual._activity_type == "ip"
    assert actual.params == "params"
    assert actual.run_params == "run_params"
    assert {k: v.tolist() for k, v in actual.step_counts.items()} == {
        ("baseline", "-"): [[2, 4, 6], [8, 10, 12]]
    }
    assert actual.demog_factors == "demog_factors"
    assert actual.birth_factors == "birth_factors"
    assert actual.hsa == "hsa"
    assert actual.strategies == "activity_avoidance"
    assert actual._row_counts.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert actual._baseline_counts.tolist() == [[2, 4, 6], [8, 10, 12]]


def test_update(mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = np.array([3.0, 4.0, 5.0, 6.0])
    aa_mock._model_run.data = pd.DataFrame({"a": [1, 2, 3, 4]})

    factor = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3], name="f")

    # act
    actual = aa_mock._update(factor, ["a"])

    # assert
    assert aa_mock._row_counts.tolist() == [3, 8, 15, 6]
    assert actual == aa_mock
    assert {k: v.tolist() for k, v in actual.step_counts.items()} == {
        ("f", "-"): [[0.0, 2.0, 6.0, 0.0], [0.0, 6.0, 14.0, 0.0]]
    }


def test_update_rn(mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = np.array([3.0, 4.0, 5.0, 6.0])
    aa_mock._model_run.data = pd.DataFrame({"rn": [1, 2, 3, 4]})

    factor = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3], name="f")

    # act
    actual = aa_mock._update_rn(factor, "a")

    # assert
    assert aa_mock._row_counts.tolist() == [3, 8, 15, 6]
    assert actual == aa_mock
    assert {k: v.tolist() for k, v in actual.step_counts.items()} == {
        ("activity_avoidance", "a"): [[0.0, 2.0, 6.0, 0.0], [0.0, 6.0, 14.0, 0.0]]
    }


def test_demographic_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.run_params = {"year": 2020, "variant": "a"}
    aa_mock._model_run.model.demog_factors = pd.DataFrame(
        {"2020": [1, 2, 3]},
        index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    )

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.demographic_adjustment()

    # assert
    assert actual == "update"
    u_mock.assert_called_once()

    assert u_mock.call_args[0][0].to_list() == [1, 2]
    assert u_mock.call_args[0][1] == ["age", "sex"]


def test_birth_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.run_params = {"year": 2020, "variant": "a"}
    # set up demog_factors/birth_factors: we end up dividing birth factors by the demog factors, so
    # we set these up here so (roughly) birth_factors == 1 / demog_factors.
    # the result of this should be x ** 2 for the values in birth_factors
    aa_mock._model_run.model.demog_factors = pd.DataFrame(
        {"2020": [1 / (x + 1) for x in range(16)]},
        index=pd.MultiIndex.from_tuples(
            [(v, s, a) for s in [1, 2] for v in ["a", "b"] for a in [1, 2, 3, 4]]
        ),
    )
    aa_mock._model_run.model.birth_factors = pd.DataFrame(
        {"2020": [(x + 9) for x in range(8)]},
        index=pd.MultiIndex.from_tuples(
            [(v, 2, a) for v in ["a", "b"] for a in [1, 2, 3, 4]]
        ),
    )

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.birth_adjustment()

    # assert
    assert actual == "update"
    u_mock.assert_called_once()

    assert u_mock.call_args[0][0].to_list() == [(x + 9) ** 2 for x in range(4)]
    assert u_mock.call_args[0][1] == ["group", "age", "sex"]


# _health_status_adjustment()


def test_health_status_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.run_params = {"health_status_adjustment": 2}

    aa_mock._model_run.model.hsa.run.return_value = "hsa"

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.health_status_adjustment()

    # assert
    assert actual == "update"
    aa_mock.hsa.run.assert_called_once_with({"health_status_adjustment": 2})
    u_mock.assert_called_once_with("hsa", ["hsagrp", "sex", "age"])


def test_covid_adjustment(mocker, mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = "ip"
    aa_mock._model_run.run_params = {
        "covid_adjustment": {"ip": {"elective": 1, "non-elective": 2}}
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.covid_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == {"elective": 1, "non-elective": 2}
    assert f.name == "covid_adjustment"
    assert cols == ["group"]


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
    assert f.to_dict() == {(True, "100"): 1, (True, "200"): 4}
    assert f.name == "waiting_list_adjustment"
    assert cols == ["is_wla", "tretspef"]


@pytest.mark.parametrize("model_type", ["ip", "op", "aae"])
def test_non_demographic_adjustment_no_params(
    mocker, mock_activity_avoidance, model_type
):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = model_type

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
        "non-demographic_adjustment": {"ip": {}, "op": {}, "aae": {}}
    }

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.non_demographic_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


@pytest.mark.parametrize(
    "model_type, expected, year",
    [
        ("ip", {"elective": 1, "non-elective": 2}, 2021),
        ("op", {"first": 3, "followup": 4}, 2021),
        ("aae", {"ambulance": 5, "walk-in": 6}, 2021),
        ("ip", {"elective": 1, "non-elective": 4}, 2022),
        ("op", {"first": 9, "followup": 16}, 2022),
        ("aae", {"ambulance": 25, "walk-in": 36}, 2022),
    ],
)
def test_non_demographic_adjustment(
    mocker, mock_activity_avoidance, model_type, expected, year
):
    # arrange
    aa_mock = mock_activity_avoidance
    aa_mock._model_run.model.model_type = model_type

    aa_mock._model_run.data = pd.DataFrame({"tretspef": ["100"] * 4 + ["200"] * 2})

    aa_mock._model_run.run_params = {
        "year": year,
        "non-demographic_adjustment": {
            "ip": {"elective": 1, "non-elective": 2},
            "op": {"first": 3, "followup": 4},
            "aae": {"ambulance": 5, "walk-in": 6},
        },
    }
    aa_mock._model_run.params = {"start_year": 2020}

    u_mock = mocker.patch(
        "model.activity_avoidance.ActivityAvoidance._update", return_value="update"
    )

    # act
    actual = aa_mock.non_demographic_adjustment()

    # assert
    assert actual == "update"
    f, cols = u_mock.call_args_list[0][0]
    assert f.to_dict() == expected
    assert f.name == "non-demographic_adjustment"
    assert cols == ["group"]


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
    assert call_args == expected


def test_apply_resampling(mock_activity_avoidance):
    # arrange
    aa_mock = mock_activity_avoidance

    aa_mock._row_counts = "row_counts"
    mr = aa_mock._model_run

    mr.data = "data"
    mr.rng.poisson.return_value = "poisson"
    mr.model.apply_resampling.return_value = ("data", np.array([[30.0, 50.0]]))

    aa_mock.step_counts = {
        ("baseline", "-"): np.array([[20.0], [30.0]]),
        ("a", "-"): np.array([[5.0], [10.0]]),  # 25, 40
        ("b", "-"): np.array([[10.0], [5.0]]),  # 45, 45
        ("c", "-"): np.array([[-10.0], [-5.0]]),  # 25, 40
    }

    expected = {
        ("baseline", "-"): [[20.0], [30.0]],
        ("a", "-"): [[23.33333333333333], [38.57142857142857]],
        ("b", "-"): [[46.66666666666666], [19.285714285714285]],
        ("c", "-"): [[-46.66666666666666], [-19.285714285714285]],
    }

    # act
    aa_mock.apply_resampling()

    # assert
    mr.rng.poisson.assert_called_once_with("row_counts")
    mr.model.apply_resampling.assert_called_once_with("poisson", "data")

    assert mr.data == "data"

    assert {k: v.tolist() for k, v in aa_mock.step_counts.items()} == expected
