"""test run_model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from nhp.model.model_iteration import ModelIteration


# fixtures
@pytest.fixture
def mock_model_iteration():
    """create a mock Model instance"""
    with patch.object(ModelIteration, "__init__", lambda m, c, r: None):
        mr = ModelIteration(None, None)
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
    prp_mock = mocker.patch("nhp.model.model_iteration.ModelIteration._patch_run_params")

    mocker.patch("nhp.model.model_iteration.ModelIteration._run")

    # act
    actual = ModelIteration(model, run)

    # assert
    assert actual.model == model
    assert actual.params == "params"
    assert actual.run_params == {"seed": 1}
    assert actual.rng == "rng"
    assert actual.data == "data"
    assert actual.step_counts is None
    assert not actual.avoided_activity

    rng_mock.assert_called_once_with(1)
    prp_mock.assert_called_once_with()
    model._get_run_params.assert_called_once_with(rp_call)
    actual._run.assert_called_once()


def test_patch_run_params(mock_model_iteration):
    # arrange
    mr = mock_model_iteration
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


def test_run(mocker, mock_model_iteration):
    """test run calls the _run method correctly"""
    # arrange
    mr_mock = mock_model_iteration
    mr_mock.model_run = 1
    mr_mock.model.baseline_step_counts = "step_counts_baseline"

    ar_mock = mocker.patch("nhp.model.model_iteration.ActivityResampling")
    ar_mock.return_value = ar_mock
    ar_mock.demographic_adjustment.return_value = ar_mock
    ar_mock.birth_adjustment.return_value = ar_mock
    ar_mock.health_status_adjustment.return_value = ar_mock
    ar_mock.inequalities_adjustment.return_value = ar_mock
    ar_mock.covid_adjustment.return_value = ar_mock
    ar_mock.expat_adjustment.return_value = ar_mock
    ar_mock.repat_adjustment.return_value = ar_mock
    ar_mock.waiting_list_adjustment.return_value = ar_mock
    ar_mock.baseline_adjustment.return_value = ar_mock
    ar_mock.non_demographic_adjustment.return_value = ar_mock

    ar_mock.apply_resampling.return_value = (data_ar_mock := Mock()), "step_counts_ar"
    mr_mock.model.activity_avoidance.return_value = (
        (data_aa_mock := Mock()),
        "step_counts_aa",
    )
    mr_mock.model.efficiencies.return_value = (data_ef_mock := Mock()), "step_counts_ef"
    mr_mock.model.calculate_avoided_activity.return_value = "avoided_activity"

    pd_mock = mocker.patch("pandas.concat", return_value="pd.concat")

    # act
    mr_mock._run()

    # assert
    ar_mock.assert_called_once_with(mr_mock)
    ar_mock.demographic_adjustment.assert_called_once()
    ar_mock.birth_adjustment.assert_called_once()
    ar_mock.health_status_adjustment.assert_called_once()
    ar_mock.inequalities_adjustment.assert_called_once()
    ar_mock.covid_adjustment.assert_called_once()
    ar_mock.expat_adjustment.assert_called_once()
    ar_mock.repat_adjustment.assert_called_once()
    ar_mock.waiting_list_adjustment.assert_called_once()
    ar_mock.baseline_adjustment.assert_called_once()
    ar_mock.non_demographic_adjustment.assert_called_once()
    ar_mock.apply_resampling.assert_called_once()

    mr_mock.model.activity_avoidance.assert_called_once_with(data_ar_mock.copy(), mr_mock)
    mr_mock.model.efficiencies.assert_called_once_with(data_aa_mock.copy(), mr_mock)
    mr_mock.model.calculate_avoided_activity.assert_called_once_with(
        data_ar_mock, data_aa_mock
    )

    pd_mock.assert_called_once_with(
        ["step_counts_baseline", "step_counts_ar", "step_counts_aa", "step_counts_ef"]
    )

    assert mr_mock.data == data_ef_mock
    assert mr_mock.avoided_activity == "avoided_activity"
    assert mr_mock.step_counts == "pd.concat"


def test_run_baseline(mocker, mock_model_iteration):
    """test run calls the _run method correctly"""
    # arrange
    mr_mock = mock_model_iteration
    mr_mock.model_run = 0

    rr_mock = mocker.patch("nhp.model.model_iteration.ActivityResampling")

    # act
    mr_mock._run()

    # assert
    rr_mock.assert_not_called()
    mr_mock.model.efficiencies.assert_not_called()


# test_fix_step_counts


def test_fix_step_counts(mock_model_iteration):
    # arrange
    baseline = np.array([[1, 1, 1], [1, 2, 3]])
    future = np.array([[0, 1, 2], [0, 2, 6]])

    mr_mock = mock_model_iteration
    mr_mock.model.get_data_counts.return_value = baseline
    mr_mock.model.measures = ["x", "y"]

    factors = pd.DataFrame(
        {
            k: v * np.ones_like(baseline[0])
            for k, v in {"a": 1.5, "b": 2.0, "c": 0.5}.items()
        }
    )

    data = pd.DataFrame({"pod": ["a", "a", "b"], "sitetret": ["c", "d", "c"]})

    # act
    actual = mr_mock.fix_step_counts(data, future, factors, "term")

    # assert
    assert actual.to_dict("list") == {
        "pod": ["a", "a", "b"] * 4,
        "sitetret": ["c", "d", "c"] * 4,
        "x": [i for i in [0.5, 1.0, -0.5] for _ in range(3)] + [-2.0, -1.0, 0.0],
        "y": [0.5, 1.0, 1.5, 1.0, 2.0, 3.0, -0.5, -1.0, -1.5, -2.0, -2.0, 0.0],
        "change_factor": [i for i in ["a", "b", "c", "term"] for _ in range(3)],
    }
    assert mr_mock.model.get_data_counts.call_args[0][0].equals(data)


# aggregate()


def test_get_aggregate_results(mock_model_iteration):
    """test the get_aggregate_results method"""

    # arrange
    mr_mock = mock_model_iteration

    mr_mock.model.aggregate.return_value = "aggregated_results", [["a"]]
    mr_mock.get_step_counts = Mock(return_value="step_counts")
    mr_mock.model.get_agg.return_value = "agg"
    mr_mock.avoided_activity = "avoided_activity"
    mr_mock.model.process_results = Mock(return_value="avoided_activity_agg")

    # act
    actual = mr_mock.get_aggregate_results()

    # assert
    assert actual == (
        {
            "default": "agg",
            "sex+age_group": "agg",
            "age": "agg",
            "a": "agg",
            "avoided_activity": "agg",
        },
        "step_counts",
    )
    mr_mock.model.aggregate.assert_called_once_with(mr_mock)
    mr_mock.get_step_counts.assert_called_once_with()
    mr_mock.model.process_results.assert_called_once_with("avoided_activity")

    assert mr_mock.model.get_agg.call_args_list == [
        call("aggregated_results"),
        call("aggregated_results", "sex", "age_group"),
        call("aggregated_results", "age"),
        call("aggregated_results", "a"),
        call("avoided_activity_agg", "sex", "age_group"),
    ]


def test_get_aggregate_results_avoided_activity_none(mock_model_iteration):
    """test the get_aggregate_results method when avoided activity is None"""

    # arrange
    mr_mock = mock_model_iteration

    mr_mock.model.aggregate.return_value = "aggregated_results", [["a"]]
    mr_mock.get_step_counts = Mock(return_value="step_counts")
    mr_mock.model.get_agg.return_value = "agg"
    mr_mock.avoided_activity = None
    mr_mock.model.process_results = Mock(return_value="avoided_activity_agg")

    # act
    actual = mr_mock.get_aggregate_results()

    # assert
    mr_mock.model.process_results.assert_not_called()


def test_get_step_counts_empty(mock_model_iteration):
    """test the get_step_counts method when step counts is empty"""
    # arrange
    mr = mock_model_iteration
    mr.step_counts = None

    # act
    actual = mr.get_step_counts()

    # assert
    assert actual is None


def test_get_step_counts(mock_model_iteration):
    """test the get_step_counts method when step counts is not empty"""

    # arrange
    step_counts = pd.DataFrame(
        {
            "pod": ["a"] * 4 + ["b"] * 4,
            "sitetret": (["a"] * 2 + ["b"] * 2) * 2,
            "change_factor": ["efficiencies"] * 8,
            "strategy": ["a", "b"] * 4,
            "x": range(8),
            "y": range(8, 16),
        }
    )

    mr = mock_model_iteration
    mr.step_counts = step_counts
    mr.model = Mock()

    mr.model.measures = ["x", "y"]
    mr.model.model_type = "ip"

    mr._step_counts_get_type_changes = Mock(side_effect=lambda x: x)

    expected = {
        ("ip", "a", "a", "efficiencies", "a", "x"): 0,
        ("ip", "a", "a", "efficiencies", "a", "y"): 8,
        ("ip", "a", "a", "efficiencies", "b", "x"): 1,
        ("ip", "a", "a", "efficiencies", "b", "y"): 9,
        ("ip", "a", "b", "efficiencies", "a", "x"): 4,
        ("ip", "a", "b", "efficiencies", "a", "y"): 12,
        ("ip", "a", "b", "efficiencies", "b", "x"): 5,
        ("ip", "a", "b", "efficiencies", "b", "y"): 13,
        ("ip", "b", "a", "efficiencies", "a", "x"): 2,
        ("ip", "b", "a", "efficiencies", "a", "y"): 10,
        ("ip", "b", "a", "efficiencies", "b", "x"): 3,
        ("ip", "b", "a", "efficiencies", "b", "y"): 11,
        ("ip", "b", "b", "efficiencies", "a", "x"): 6,
        ("ip", "b", "b", "efficiencies", "a", "y"): 14,
        ("ip", "b", "b", "efficiencies", "b", "x"): 7,
        ("ip", "b", "b", "efficiencies", "b", "y"): 15,
    }

    # act
    actual = mr.get_step_counts()

    # assert
    mr._step_counts_get_type_changes.assert_called_once()

    assert actual.to_dict() == expected


def test_step_counts_get_type_changes(mock_model_iteration):
    # arrange
    mr = mock_model_iteration
    mr._step_counts_get_type_change_daycase = Mock(return_value=pd.Series([4]))
    mr._step_counts_get_type_change_outpatients = Mock(return_value=pd.Series([5]))
    mr._step_counts_get_type_change_sdec = Mock(return_value=pd.Series([6]))

    sc = pd.Series([1, 2, 3], index=[3, 2, 1])

    # act
    actual = mr._step_counts_get_type_changes(sc)

    # assert
    assert actual.to_list() == [1, 2, 3, 4, 5, 6]
    mr._step_counts_get_type_change_daycase.assert_called_once_with(sc)
    mr._step_counts_get_type_change_outpatients.assert_called_once_with(sc)
    mr._step_counts_get_type_change_sdec.assert_called_once_with(sc)


def test_step_counts_get_type_change_daycase(mock_model_iteration):
    # arrange
    mr = mock_model_iteration

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


def test_step_counts_get_type_change_outpatients(mock_model_iteration):
    # arrange
    mr = mock_model_iteration

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


def test_step_counts_get_type_change_sdec(mock_model_iteration):
    # arrange
    mr = mock_model_iteration

    sc = pd.Series(
        [-i for i in range(10)],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", strategy, "ip", "c", "ip_emergency_admission", measure)
                for strategy in [
                    "x",
                    "same_day_emergency_care_very_high",
                    "same_day_emergency_care_high",
                    "same_day_emergency_care_moderate",
                    "same_day_emergency_care_low",
                ]
                for measure in ["admissions", "beddays"]
            ]
        ),
        name="value",
    )

    expected = pd.Series(
        [2, 4, 6, 8],
        index=pd.MultiIndex.from_tuples(
            ("a", strategy, "aae", "c", "aae_type-05", "arrivals")
            for strategy in [
                "same_day_emergency_care_very_high",
                "same_day_emergency_care_high",
                "same_day_emergency_care_moderate",
                "same_day_emergency_care_low",
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
    actual = mr._step_counts_get_type_change_sdec(sc)

    # assert
    assert actual.equals(expected.sort_index())


def test_get_model_results(mock_model_iteration):
    # arrange
    mr = mock_model_iteration
    mr.data = pd.DataFrame(
        {"x": [1, 2, 3], "hsagrp": ["h", "h", "h"]}, index=["a", "b", "c"]
    )

    # act
    actual = mr.get_model_results()

    # assert
    assert actual.to_dict(orient="list") == {"x": [1, 2, 3]}
