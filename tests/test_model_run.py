"""test run_model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from model.model_run import ModelRun


# fixtures
@pytest.fixture
def mock_model_run():
    """create a mock Model instance"""
    with patch.object(ModelRun, "__init__", lambda m, c, r: None):
        mr = ModelRun(None, None)
    mr.model = Mock()
    return mr


@pytest.mark.parametrize("run, rp_call", [(-1, 0), (0, 0), (1, 1)])
def test_init(mocker, run, rp_call):
    # arrange
    model = Mock()
    model.params = "params"
    model.data = Mock()
    model.data.copy.return_value = "data"
    model._get_run_params.return_value = {"seed": 1}

    rng_mock = mocker.patch("numpy.random.default_rng", return_value="rng")
    prp_mock = mocker.patch("model.model_run.ModelRun._patch_run_params")

    mocker.patch("model.model_run.ModelRun._run")

    # act
    actual = ModelRun(model, run)

    # assert
    assert actual.model == model
    assert actual.params == "params"
    assert actual.run_params == {"seed": 1}
    assert actual.rng == "rng"
    assert actual.data == "data"
    assert not actual.step_counts

    rng_mock.assert_called_once_with(1)
    prp_mock.assert_called_once_with()
    model._get_run_params.assert_called_once_with(rp_call)
    actual._run.assert_called_once()


def test_patch_run_params(mock_model_run):
    # arrange
    mr = mock_model_run
    mr.run_params = {
        "expat": {"op": {"100": 1, "200": 2}, "aae": {"ambulance": 3, "walk-in": 4}},
        "repat_local": {
            "op": {"100": 1, "200": 2},
            "aae": {"ambulance": 3, "walk-in": 4},
        },
        "repat_nonlocal": {
            "op": {"100": 1, "200": 2},
            "aae": {"ambulance": 3, "walk-in": 4},
        },
        "baseline_adjustment": {"aae": {"ambulance": 5, "walk-in": 6}},
    }

    # act
    mr._patch_run_params()

    # assert
    assert mr.run_params == {
        "expat": {
            "op": {
                "first": {"100": 1, "200": 2},
                "followup": {"100": 1, "200": 2},
                "procedure": {"100": 1, "200": 2},
            },
            "aae": {"ambulance": {"Other": 3}, "walk-in": {"Other": 4}},
        },
        "repat_local": {
            "op": {
                "first": {"100": 1, "200": 2},
                "followup": {"100": 1, "200": 2},
                "procedure": {"100": 1, "200": 2},
            },
            "aae": {"ambulance": {"Other": 3}, "walk-in": {"Other": 4}},
        },
        "repat_nonlocal": {
            "op": {
                "first": {"100": 1, "200": 2},
                "followup": {"100": 1, "200": 2},
                "procedure": {"100": 1, "200": 2},
            },
            "aae": {"ambulance": {"Other": 3}, "walk-in": {"Other": 4}},
        },
        "baseline_adjustment": {
            "aae": {"ambulance": {"Other": 5}, "walk-in": {"Other": 6}}
        },
    }


# run()


def test_run(mocker, mock_model_run):
    """test run calls the _run method correctly"""
    # arrange
    mr_mock = mock_model_run
    mr_mock.model_run = 0

    rr_mock = mocker.patch("model.model_run.ActivityAvoidance")
    rr_mock.return_value = rr_mock
    rr_mock.demographic_adjustment.return_value = rr_mock
    rr_mock.birth_adjustment.return_value = rr_mock
    rr_mock.health_status_adjustment.return_value = rr_mock
    rr_mock.covid_adjustment.return_value = rr_mock
    rr_mock.expat_adjustment.return_value = rr_mock
    rr_mock.repat_adjustment.return_value = rr_mock
    rr_mock.waiting_list_adjustment.return_value = rr_mock
    rr_mock.baseline_adjustment.return_value = rr_mock
    rr_mock.non_demographic_adjustment.return_value = rr_mock
    rr_mock.activity_avoidance.return_value = rr_mock
    rr_mock.apply_resampling.return_value = rr_mock

    # act
    mr_mock._run()

    # assert
    rr_mock.assert_called_once_with(mr_mock)
    rr_mock.demographic_adjustment.assert_called_once()
    rr_mock.birth_adjustment.assert_called_once()
    rr_mock.health_status_adjustment.assert_called_once()
    rr_mock.covid_adjustment.assert_called_once()
    rr_mock.expat_adjustment.assert_called_once()
    rr_mock.repat_adjustment.assert_called_once()
    rr_mock.waiting_list_adjustment.assert_called_once()
    rr_mock.baseline_adjustment.assert_called_once()
    rr_mock.non_demographic_adjustment.assert_called_once()
    rr_mock.activity_avoidance.assert_called_once()
    rr_mock.apply_resampling.assert_called_once()

    mr_mock.model.efficiencies.assert_called_once_with(mr_mock)


def test_run_baseline(mocker, mock_model_run):
    """test run calls the _run method correctly"""
    # arrange
    mr_mock = mock_model_run
    mr_mock.model_run = -1

    rr_mock = mocker.patch("model.model_run.ActivityAvoidance")

    # act
    mr_mock._run()

    # assert
    rr_mock.assert_not_called()
    mr_mock.model.efficiencies.assert_not_called()


# aggregate()


def test_get_aggregate_results(mock_model_run):
    """test the _run method"""

    # arrange
    def mock_agg(cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: name}

    mr_mock = mock_model_run

    mock_aggregates = {"results": "results"}
    mr_mock.model.aggregate.return_value = mock_agg, mock_aggregates

    mr_mock.get_step_counts = Mock(return_value={"step_counts": "step_counts"})

    # act
    results = mr_mock.get_aggregate_results()

    # assert
    assert results == {
        i: i for i in ["default", "sex+age_group", "age", "results", "step_counts"]
    }
    mr_mock.model.aggregate.assert_called_once_with(mr_mock)


def test_get_step_counts_empty(mock_model_run):
    # arrange
    step_counts = {}

    mr = mock_model_run
    mr.step_counts = step_counts
    mr.model = Mock()
    mr.step_counts = step_counts
    mr.model.data = pd.DataFrame(
        {"sitetret": ["a", "a", "b"], "group": ["a", "b", "a"]}
    )
    mr.model.measures = ["x", "y"]
    mr.model.model_type = "ip"

    # act
    actual = mock_model_run.get_step_counts()

    # assert
    assert not actual


def test_get_step_counts(mock_model_run):
    # arrange
    step_counts = {
        ("baseline", "-"): np.array([[1, 2, 3], [4, 5, 6]]),
        ("x", "-"): np.array([[3, 2, 1], [6, 5, 4]]),
    }

    mr = mock_model_run
    mr.step_counts = step_counts
    mr.model = Mock()
    mr.step_counts = step_counts
    mr.model.data = pd.DataFrame({"sitetret": ["a", "a", "b"], "pod": ["a", "b", "a"]})
    mr.model.measures = ["x", "y"]
    mr.model.model_type = "ip"

    expected = {
        "step_counts": {
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "a"),
                    ("pod", "a"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 1,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "a"),
                    ("pod", "a"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 3,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "a"),
                    ("pod", "b"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 2,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "a"),
                    ("pod", "b"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 2,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "b"),
                    ("pod", "a"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 3,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("sitetret", "b"),
                    ("pod", "a"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "x"),
                }
            ): 1,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "a"),
                    ("sitetret", "a"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 4,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "a"),
                    ("sitetret", "a"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 6,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "b"),
                    ("sitetret", "a"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 5,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "b"),
                    ("sitetret", "a"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 5,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "a"),
                    ("sitetret", "b"),
                    ("change_factor", "baseline"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 6,
            frozenset(
                {
                    ("activity_type", "ip"),
                    ("pod", "a"),
                    ("sitetret", "b"),
                    ("change_factor", "x"),
                    ("strategy", "-"),
                    ("measure", "y"),
                }
            ): 4,
        }
    }

    # act
    actual = mr.get_step_counts()

    # assert
    assert actual == expected


def test_get_model_results(mock_model_run):
    # arrange
    mr = mock_model_run
    mr.data = pd.DataFrame(
        {"x": [1, 2, 3], "hsagrp": ["h", "h", "h"]}, index=["a", "b", "c"]
    )

    # act
    actual = mr.get_model_results()

    # assert
    assert actual.to_dict(orient="list") == {"x": [1, 2, 3]}
