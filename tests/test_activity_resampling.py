"""test activity avoidance"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from model.activity_resampling import ActivityResampling


# fixtures
@pytest.fixture
def mock_activity_resampling():
    """create a mock Model instance"""
    with patch.object(ActivityResampling, "__init__", lambda m, c: None):
        mdl = ActivityResampling(None)
    mdl._model_iteration = Mock()
    mdl._model_iteration.model.baseline_counts = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    ).astype(float)
    return mdl


def test_init():
    # arrange
    model_iteration = Mock()
    model_iteration.data = "data"
    model_iteration.model.model_type = "ip"
    model_iteration.model.baseline_counts = np.array([[1, 2, 3], [4, 5, 6]])
    model_iteration.params = "params"
    model_iteration.run_params = "run_params"
    model_iteration.model.demog_factors = "demog_factors"
    model_iteration.model.birth_factors = "birth_factors"
    model_iteration.model.hsa = "hsa"

    # act
    actual = ActivityResampling(model_iteration)

    # assert
    assert actual.data == "data"
    assert actual._activity_type == "ip"
    assert actual.params == "params"
    assert actual.run_params == "run_params"
    assert actual.demog_factors == "demog_factors"
    assert actual.birth_factors == "birth_factors"
    assert actual.hsa == "hsa"
    assert actual._baseline_counts.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_update(mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock.factors = []

    # aa_mock._row_counts = np.array([3.0, 4.0, 5.0, 6.0])
    aa_mock._model_iteration.data = pd.DataFrame({"a": [1, 2, 3, 4]})

    factor = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3], name="f")
    factor.index.names = ["a"]

    # act
    actual = aa_mock._update(factor)

    # assert
    assert aa_mock.factors[0].to_list() == [1, 2, 3, 1]
    assert actual == aa_mock


def test_demographic_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.run_params = {"year": 2020, "variant": "a"}
    aa_mock._model_iteration.model.demog_factors = pd.DataFrame(
        {"2020": [1, 2, 3]},
        index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    )

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.demographic_adjustment()

    # assert
    assert actual == "update"
    u_mock.assert_called_once()

    assert u_mock.call_args[0][0].equals(
        pd.Series([1, 2], name="demographic_adjustment")
    )


def test_birth_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.run_params = {"year": 2020, "variant": "a"}
    # set up demog_factors/birth_factors: we end up dividing birth factors by the demog factors, so
    # we set these up here so (roughly) birth_factors == 1 / demog_factors.
    # the result of this should be x ** 2 for the values in birth_factors
    aa_mock._model_iteration.model.demog_factors = pd.DataFrame(
        {"2020": [1 / (x + 1) for x in range(16)]},
        index=pd.MultiIndex.from_tuples(
            [(v, s, a) for s in [1, 2] for v in ["a", "b"] for a in [1, 2, 3, 4]]
        ),
    )
    aa_mock._model_iteration.model.birth_factors = pd.DataFrame(
        {"2020": [(x + 9) for x in range(8)]},
        index=pd.MultiIndex.from_tuples(
            [(v, 2, a) for v in ["a", "b"] for a in [1, 2, 3, 4]]
        ),
    )

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.birth_adjustment()

    # assert
    assert actual == "update"
    u_mock.assert_called_once()

    assert u_mock.call_args[0][0].to_dict() == {
        ("maternity", 2, 1): 81,
        ("maternity", 2, 2): 100,
        ("maternity", 2, 3): 121,
        ("maternity", 2, 4): 144,
    }


# _health_status_adjustment()


def test_health_status_adjustment_when_enabled(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.params = {"health_status_adjustment": True}
    aa_mock._model_iteration.run_params = {"health_status_adjustment": 2}

    aa_mock._model_iteration.model.hsa.run.return_value = "hsa"

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.health_status_adjustment()

    # assert
    assert actual == "update"
    aa_mock.hsa.run.assert_called_once_with({"health_status_adjustment": 2})
    u_mock.assert_called_once_with("hsa")


def test_health_status_adjustmen_when_disabled(mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.params = {"health_status_adjustment": False}
    aa_mock._model_iteration.run_params = {"health_status_adjustment": 2}

    aa_mock._model_iteration.model.hsa.run.return_value = "hsa"

    # act
    actual = aa_mock.health_status_adjustment()

    # assert
    assert actual == aa_mock
    aa_mock.hsa.run.assert_not_called()


def test_covid_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {
        "covid_adjustment": {"ip": {"elective": 1, "non-elective": 2}}
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.covid_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == {"elective": 1, "non-elective": 2}


def test_expat_adjustment_no_params(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {"expat": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.expat_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_expat_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {
        "expat": {
            "ip": {
                "elective": {"100": 1, "200": 2},
                "non-elective": {"300": 3, "400": 4},
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.expat_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == {
        ("elective", "100"): 1,
        ("elective", "200"): 2,
        ("non-elective", "300"): 3,
        ("non-elective", "400"): 4,
    }


def test_repat_adjustment_no_params(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {
        "repat_local": {"ip": {}},
        "repat_nonlocal": {"ip": {}},
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.repat_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_repat_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {
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
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.repat_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == {
        (1, "elective", "100"): 1,
        (1, "elective", "200"): 2,
        (1, "non-elective", "300"): 3,
        (1, "non-elective", "400"): 4,
        (0, "elective", "100"): 5,
        (0, "elective", "200"): 6,
        (0, "non-elective", "300"): 7,
        (0, "non-elective", "400"): 8,
    }


def test_baseline_adjustment_no_params(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {"baseline_adjustment": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.baseline_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_baseline_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"
    aa_mock._model_iteration.run_params = {
        "baseline_adjustment": {
            "ip": {
                "elective": {"100": 1, "200": 2},
                "non-elective": {"300": 3, "400": 4},
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.baseline_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == {
        ("elective", "100"): 1.0,
        ("elective", "200"): 2.0,
        ("non-elective", "300"): 3.0,
        ("non-elective", "400"): 4.0,
    }


def test_waiting_list_adjustment_aae(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "aae"

    aa_mock._model_iteration.data = pd.DataFrame(
        {"tretspef": ["100"] * 4 + ["200"] * 2}
    )

    aa_mock._model_iteration.run_params = {
        "waiting_list_adjustment": {
            "ip": {
                "100": 1,
                "200": 4,
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.waiting_list_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_waiting_list_adjustment_no_params(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"

    aa_mock._model_iteration.data = pd.DataFrame(
        {"tretspef": ["100"] * 4 + ["200"] * 2}
    )

    aa_mock._model_iteration.run_params = {"waiting_list_adjustment": {"ip": {}}}

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.waiting_list_adjustment()

    # assert
    assert actual == aa_mock
    u_mock.assert_not_called()


def test_waiting_list_adjustment(mocker, mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = "ip"

    aa_mock._model_iteration.data = pd.DataFrame(
        {"tretspef": ["100"] * 4 + ["200"] * 2}
    )

    aa_mock._model_iteration.run_params = {
        "waiting_list_adjustment": {
            "ip": {
                "100": 1,
                "200": 4,
            }
        }
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.waiting_list_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == {(True, "100"): 1, (True, "200"): 4}


@pytest.mark.parametrize("model_type", ["ip", "op", "aae"])
def test_non_demographic_adjustment_no_params(
    mocker, mock_activity_resampling, model_type
):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = model_type

    aa_mock._model_iteration.data = pd.DataFrame(
        {"tretspef": ["100"] * 4 + ["200"] * 2}
    )

    aa_mock._model_iteration.run_params = {
        "non-demographic_adjustment": {"ip": {}, "op": {}, "aae": {}}
    }

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
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
    mocker, mock_activity_resampling, model_type, expected, year
):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock._model_iteration.model.model_type = model_type

    aa_mock._model_iteration.data = pd.DataFrame(
        {"tretspef": ["100"] * 4 + ["200"] * 2}
    )

    aa_mock._model_iteration.run_params = {
        "year": year,
        "non-demographic_adjustment": {
            "ip": {"elective": 1, "non-elective": 2},
            "op": {"first": 3, "followup": 4},
            "aae": {"ambulance": 5, "walk-in": 6},
        },
    }
    aa_mock._model_iteration.params = {"start_year": 2020}

    u_mock = mocker.patch(
        "model.activity_resampling.ActivityResampling._update", return_value="update"
    )

    # act
    actual = aa_mock.non_demographic_adjustment()

    # assert
    assert actual == "update"
    assert u_mock.call_args[0][0].to_dict() == expected


def test_apply_resampling(mock_activity_resampling):
    # arrange
    aa_mock = mock_activity_resampling
    aa_mock.factors = [
        pd.Series([1, 2, 3, 4], name="a"),
        pd.Series([5, 6, 7, 8], name="b"),
    ]

    mr = aa_mock._model_iteration

    mr.data = "data"
    mr.rng.poisson.return_value = "poisson"
    mr.model.apply_resampling.return_value = "data"

    mr.fix_step_counts.return_value = pd.DataFrame({"x": [1]})

    # act
    actual = aa_mock.apply_resampling()

    # assert
    assert actual[0] == "data"
    assert actual[1].to_dict("list") == {"x": [1], "strategy": ["-"]}

    mr.rng.poisson.assert_called_once()
    assert mr.rng.poisson.call_args[0][0].tolist() == [
        [5.0, 24.0, 63.0, 128.0],
        [25.0, 72.0, 147.0, 256.0],
    ]

    mr.model.apply_resampling.assert_called_once_with("poisson", "data")

    mr.fix_step_counts.assert_called_once()
    args = mr.fix_step_counts.call_args[0]
    assert args[0] == "data"
    assert args[1] == "poisson"
    assert args[2].to_dict("list") == {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
    assert args[3] == "model_interaction_term"
