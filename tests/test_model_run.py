"""test run_model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, call, patch

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
    mr_mock.model_run = 1

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
    mr_mock.model_run = 0

    rr_mock = mocker.patch("model.model_run.ActivityAvoidance")

    # act
    mr_mock._run()

    # assert
    rr_mock.assert_not_called()
    mr_mock.model.efficiencies.assert_not_called()


# aggregate()


def test_get_aggregate_results(mock_model_run):
    """test the get_aggregate_results method"""

    # arrange
    mr_mock = mock_model_run

    mr_mock.model.aggregate.return_value = "aggregated_results", [["a"]]
    mr_mock.get_step_counts = Mock(return_value="step_counts")
    mr_mock.model.get_agg.return_value = "agg"

    # act
    actual = mr_mock.get_aggregate_results()

    # assert
    assert actual == (
        {"default": "agg", "sex+age_group": "agg", "age": "agg", "a": "agg"},
        "step_counts",
    )
    mr_mock.model.aggregate.assert_called_once_with(mr_mock)
    mr_mock.get_step_counts.assert_called_once_with()

    assert mr_mock.model.get_agg.call_args_list == [
        call("aggregated_results"),
        call("aggregated_results", "sex", "age_group"),
        call("aggregated_results", "age"),
        call("aggregated_results", "a"),
    ]


def test_get_step_counts_empty(mock_model_run):
    """test the get_step_counts method when step counts is empty"""

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
    """test the get_step_counts method when step counts is not empty"""

    # arrange
    step_counts = {
        ("x", "-"): np.array([[3, 2, 1, 1], [6, 5, 4, 1]]),
        ("y", "a"): np.array([[2, 1, 3, 1], [5, 4, 6, 1]]),
        ("z", "a"): np.array([[-2, -1, -3, -1], [-5, -4, -6, -1]]),
        ("efficiencies", "-"): np.array([[-1, -2, 0, -1], [-4, -5, 0, -1]]),
    }

    mr = mock_model_run
    mr.step_counts = step_counts
    mr.model = Mock()

    mr.model.measures = ["x", "y"]
    mr.model.model_type = "ip"
    mr.model.data = pd.DataFrame(
        {"sitetret": ["a", "a", "b", "b", "b"], "pod": ["a", "b", "a", "b", "b"]}
    )
    mr.model.baseline_counts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    mr._step_counts_get_type_changes = Mock(side_effect=lambda x: x)

    expected = {
        ("baseline", "-", "ip", "a", "a", "x"): 1.0,
        ("baseline", "-", "ip", "a", "a", "y"): 5.0,
        ("baseline", "-", "ip", "a", "b", "x"): 2.0,
        ("baseline", "-", "ip", "a", "b", "y"): 6.0,
        ("baseline", "-", "ip", "b", "a", "x"): 3.0,
        ("baseline", "-", "ip", "b", "a", "y"): 7.0,
        ("baseline", "-", "ip", "b", "b", "x"): 4.0,
        ("baseline", "-", "ip", "b", "b", "y"): 8.0,
        ("x", "-", "ip", "a", "a", "x"): 3.0,
        ("x", "-", "ip", "a", "a", "y"): 6.0,
        ("x", "-", "ip", "a", "b", "x"): 2.0,
        ("x", "-", "ip", "a", "b", "y"): 5.0,
        ("x", "-", "ip", "b", "a", "x"): 1.0,
        ("x", "-", "ip", "b", "a", "y"): 4.0,
        ("x", "-", "ip", "b", "b", "x"): 1.0,
        ("x", "-", "ip", "b", "b", "y"): 1.0,
        ("y", "a", "ip", "a", "a", "x"): 2.0,
        ("y", "a", "ip", "a", "a", "y"): 5.0,
        ("y", "a", "ip", "a", "b", "x"): 1.0,
        ("y", "a", "ip", "a", "b", "y"): 4.0,
        ("y", "a", "ip", "b", "a", "x"): 3.0,
        ("y", "a", "ip", "b", "a", "y"): 6.0,
        ("y", "a", "ip", "b", "b", "x"): 1.0,
        ("y", "a", "ip", "b", "b", "y"): 1.0,
        ("z", "a", "ip", "a", "a", "x"): -2.0,
        ("z", "a", "ip", "a", "a", "y"): -5.0,
        ("z", "a", "ip", "a", "b", "x"): -1.0,
        ("z", "a", "ip", "a", "b", "y"): -4.0,
        ("z", "a", "ip", "b", "a", "x"): -3.0,
        ("z", "a", "ip", "b", "a", "y"): -6.0,
        ("z", "a", "ip", "b", "b", "x"): -1.0,
        ("z", "a", "ip", "b", "b", "y"): -1.0,
        ("efficiencies", "-", "ip", "a", "a", "x"): -1.0,
        ("efficiencies", "-", "ip", "a", "a", "y"): -4.0,
        ("efficiencies", "-", "ip", "a", "b", "x"): -2.0,
        ("efficiencies", "-", "ip", "a", "b", "y"): -5.0,
        ("efficiencies", "-", "ip", "b", "a", "x"): 0.0,
        ("efficiencies", "-", "ip", "b", "a", "y"): 0.0,
        ("efficiencies", "-", "ip", "b", "b", "x"): -1.0,
        ("efficiencies", "-", "ip", "b", "b", "y"): -1.0,
    }

    # act
    actual = mr.get_step_counts()

    # assert
    mr._step_counts_get_type_changes.assert_called_once()

    assert actual.to_dict() == expected


def test_step_counts_get_changes(mock_model_run):
    # arrange
    mr = mock_model_run
    mr._step_counts_get_type_change_daycase = Mock(return_value=pd.Series([4]))
    mr._step_counts_get_type_change_outpatients = Mock(return_value=pd.Series([5]))

    sc = pd.Series([1, 2, 3], index=[3, 2, 1])

    # act
    actual = mr._step_counts_get_type_changes(sc)

    # assert
    assert actual.to_list() == [3, 2, 1, 4, 5]
    mr._step_counts_get_type_change_daycase.assert_called_once_with(sc)
    mr._step_counts_get_type_change_outpatients.assert_called_once_with(sc)


def test_step_counts_get_type_change_daycase(mock_model_run):
    # arrange
    mr = mock_model_run

    sc = pd.Series(
        [-i for i in range(10)],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", strategy, "ip", "c", "ip_elective_admission", measure)
                for strategy in [
                    "x",
                    "day_procedures_usually_dc",
                    "day_procedures_occasionally_dc",
                    "day_procedures_usually_op",
                    "day_procedures_occasionally_op",
                ]
                for measure in ["admissions", "beddays"]
            ]
        ),
        name="value",
    )

    expected = pd.Series(
        [4, 4, 2, 2],
        index=pd.MultiIndex.from_tuples(
            ("a", strategy, "ip", "c", "ip_elective_daycase", measure)
            for strategy in [
                "day_procedures_occasionally_dc",
                "day_procedures_usually_dc",
            ]
            for measure in ["admissions", "beddays"]
        ),
        name="value",
    )

    expected.index.names = sc.index.names = [
        "change_factor",
        "strategy",
        "activity_type",
        "sitetret",
        "pod",
        "measure",
    ]

    # act
    actual = mr._step_counts_get_type_change_daycase(sc)

    # assert
    assert actual.equals(expected)


def test_step_counts_get_type_change_outpatients(mock_model_run):
    # arrange
    mr = mock_model_run

    sc = pd.Series(
        [-i for i in range(20)],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", strategy, "ip", "c", pod, measure)
                for strategy in [
                    "x",
                    "day_procedures_usually_dc",
                    "day_procedures_occasionally_dc",
                    "day_procedures_usually_op",
                    "day_procedures_occasionally_op",
                ]
                for pod in ["ip_elective_admission", "ip_elective_daycase"]
                for measure in ["admissions", "beddays"]
            ]
        ),
        name="value",
    )

    expected = pd.Series(
        [34, 26],
        index=pd.MultiIndex.from_tuples(
            ("a", strategy, "op", "c", "op_procedure", "attendances")
            for strategy in [
                "day_procedures_occasionally_op",
                "day_procedures_usually_op",
            ]
        ),
        name="value",
    )

    expected.index.names = sc.index.names = [
        "change_factor",
        "strategy",
        "activity_type",
        "sitetret",
        "pod",
        "measure",
    ]

    # act
    actual = mr._step_counts_get_type_change_outpatients(sc)

    # assert
    assert actual.equals(expected)


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
