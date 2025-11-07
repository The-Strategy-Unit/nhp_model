"""Test inpatients model."""

from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from nhp.model.inpatients import InpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """Create a mock Model instance."""
    with patch.object(InpatientsModel, "__init__", lambda s, p, d, h, r: None):
        mdl = InpatientsModel(None, None, None, None)  # type: ignore
    mdl.model_type = "ip"
    mdl.params = {
        "dataset": "synthetic",
        "nhp.model_runs": 3,
        "seed": 1,
        "demographic_factors": {
            "file": "demographics_file.csv",
            "variant_probabilities": {"a": 0.6, "b": 0.4},
        },
        "start_year": 2018,
        "end_year": 2020,
        "health_status_adjustment": [0.8, 1.0],
        "waiting_list_adjustment": "waiting_list_adjustment",
        "expat": {"ip": {"elective": {"Other": [0.7, 0.9]}}},
        "repat_local": {"ip": {"elective": {"Other": [1.0, 1.2]}}},
        "repat_nonlocal": {"ip": {"elective": {"Other": [1.3, 1.5]}}},
        "non-demographic_adjustment": {
            "a": {"a_a": [1, 1.2], "a_b": [1, 1.2]},
            "b": {"b_a": [1, 1.2], "b_b": [1, 1.2]},
        },
        "inpatient_factors": {
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
    }
    # create a minimal data object for testing
    mdl.data = pd.DataFrame(
        {
            "rn": list(range(1, 21)),
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
            "hsagrp": [x for _ in range(1, 11) for x in ["aae_a_a", "aae_b_b"]],
            "admigroup": ["elective", "non-elective"] * 10,
            "admidate": [pd.to_datetime("2022-01-01")] * 20,
        }
    )
    return mdl


# methods


def test_init_calls_super_init(mocker):
    """Test that the model calls the super method and loads the strategies."""
    # arrange
    super_mock = mocker.patch("nhp.model.inpatients.super")
    # act
    InpatientsModel("params", "nhp_data", "hsa", "run_params")  # type: ignore
    # assert
    super_mock.assert_called_once()


def test_get_data(mock_model):
    # arrange
    mdl = mock_model
    data_loader = Mock()
    data_loader.get_ip.return_value = "ip data"

    # act
    actual = mdl._get_data(data_loader)

    # assert
    assert actual == "ip data"
    data_loader.get_ip.assert_called_once_with()


@pytest.mark.parametrize(
    "gen_los_type, expected",
    [
        ("general_los_reduction_elective", [7, 9]),
        ("general_los_reduction_emergency", [8, 10]),
    ],
)
def test_load_strategies(mock_model, gen_los_type, expected):
    """Test that the method returns a dataframe."""
    # arrange
    mdl = mock_model
    mdl.data["speldur"] = np.repeat([1, 0], 10)
    mdl.data["admimeth"] = np.tile(["1", "2"], 10)
    mdl.data["classpat"] = "1"
    mdl.data.loc[[0, 1, 2, 3], "classpat"] = "2"
    mdl.params = {
        "activity_avoidance": {"ip": {"a": 1, "b": 2}},
        "efficiencies": {"ip": {"b": 3, "c": 4, gen_los_type: 5}},
    }

    data_loader = Mock()
    data_loader.get_ip_strategies.return_value = {
        "activity_avoidance": pd.DataFrame(
            {"rn": [1, 2, 3], "admission_avoidance_strategy": ["a", "b", "c"]}
        ),
        "efficiencies": pd.DataFrame({"rn": [4, 5, 6], "los_reduction_strategy": ["a", "b", "c"]}),
    }
    expected = {
        "activity_avoidance": {"strategy": {1: "a", 2: "b"}},
        "efficiencies": {"strategy": {5: "b", 6: "c", **{i: gen_los_type for i in expected}}},
    }

    # act
    mdl._load_strategies(data_loader)

    # assert
    assert {k: v.to_dict() for k, v in mdl.strategies.items()} == expected


def test_get_data_counts(mock_model):
    # arrange
    mdl = mock_model
    data = pd.DataFrame({"rn": [1, 2, 3], "speldur": [0, 1, 2]})

    # act
    actual = mdl.get_data_counts(data)

    # assert
    assert actual.tolist() == [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]


def test_apply_resampling(mock_model):
    # arrange
    row_samples = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    data = pd.DataFrame({"rn": [0, 1, 2, 3]})
    # act
    data = mock_model.apply_resampling(row_samples, data)
    # assert
    assert data["rn"].to_list() == [1, 2, 2, 3, 3, 3]


def test_efficiencies(mocker, mock_model):
    """Test that it runs the model steps."""
    mdl = mock_model

    mock = mocker.patch("nhp.model.inpatients.InpatientEfficiencies")
    mock.return_value = mock
    mock.data = "data_efficiencies"

    mock.losr_all.return_value = mock
    mock.losr_sdec.return_value = mock
    mock.losr_preop.return_value = mock
    mock.losr_day_procedures.return_value = mock
    mock.update_step_counts.return_value = mock

    mock.get_step_counts.return_value = "step_counts"

    mock_model_iteration = Mock()
    mock_model_iteration.empty = False
    mock_model_iteration.model.strategies = {"efficiencies": mock_model_iteration}

    # act
    actual = mdl.efficiencies("data", mock_model_iteration)

    # assert
    assert actual == ("data_efficiencies", "step_counts")

    mock.assert_called_once_with("data", mock_model_iteration)
    mock.losr_all.assert_called_once()
    mock.losr_sdec.assert_called_once()
    mock.losr_preop.assert_called_once()
    assert mock.losr_day_procedures.call_count == 2
    assert mock.losr_day_procedures.call_args_list == [
        call("day_procedures_daycase"),
        call("day_procedures_outpatients"),
    ]
    mock.get_step_counts.assert_called_once()


def test_efficiencies_no_params(mocker, mock_model):
    """Test that it runs the model steps."""
    mdl = mock_model

    mock = mocker.patch("nhp.model.inpatients.InpatientEfficiencies")
    mock.return_value = mock

    mock_model_iteration = Mock()
    mock_model_iteration.empty = True
    mock_model_iteration.model.strategies = {"efficiencies": mock_model_iteration}

    # act
    actual = mdl.efficiencies("data", mock_model_iteration)

    # assert
    assert actual == ("data", None)
    mock.assert_not_called()


def test_calculate_avoided_activity(mock_model):
    # arrange
    data = pd.DataFrame({"rn": [0, 0, 1, 2], "a": [0, 0, 1, 1]})
    data_resampled = pd.DataFrame({"rn": [0, 1], "a": [0, 1]})
    # act
    actual = mock_model.calculate_avoided_activity(data, data_resampled)
    # assert
    assert actual.to_dict(orient="list") == {"rn": [0, 2], "a": [0, 1]}


def test_process_results(mock_model):
    # arrange
    xs = list(range(6)) * 2
    df = pd.DataFrame(
        {
            "sitetret": ["trust"] * 12,
            "age": list(range(12)),
            "age_group": xs,
            "sex": xs,
            "group": ["elective", "non-elective", "maternity"] * 4,
            "classpat": ["1", "-2", "-3", "4", "5", "-1"] * 2,
            "tretspef": [str(i) for i in range(12)],
            "tretspef_grouped": [str(i) for i in xs],
            "rn": [1] * 12,
            "has_procedure": [0, 1] * 6,
            "speldur": list(range(12)),
            "maternity_delivery_in_spell": [True, False] * 6,
        }
    )
    df["pod"] = "ip_" + df["group"] + "_admission"

    expected = pd.DataFrame(
        {
            "sitetret": [
                "trust",
            ]
            * 24,
            "age": [
                0,
                0,
                1,
                1,
                1,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
                6,
                6,
                7,
                7,
                7,
                8,
                9,
                9,
                9,
                10,
                10,
                11,
            ],
            "age_group": [
                0,
                0,
                1,
                1,
                1,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
                0,
                0,
                1,
                1,
                1,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
            ],
            "sex": [
                0,
                0,
                1,
                1,
                1,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
                0,
                0,
                1,
                1,
                1,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
            ],
            "pod": [
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_elective_daycase",
                "ip_elective_daycase",
                "ip_elective_daycase",
                "aae_type-05",
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_non-elective_admission",
                "ip_non-elective_admission",
                "op_procedure",
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_elective_daycase",
                "ip_elective_daycase",
                "ip_elective_daycase",
                "aae_type-05",
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_elective_admission",
                "ip_non-elective_admission",
                "ip_non-elective_admission",
                "op_procedure",
            ],
            "tretspef": [
                "0",
                "0",
                "1",
                "1",
                "1",
                "Other",
                "3",
                "3",
                "3",
                "4",
                "4",
                "5",
                "6",
                "6",
                "7",
                "7",
                "7",
                "Other",
                "9",
                "9",
                "9",
                "10",
                "10",
                "11",
            ],
            "tretspef_grouped": [
                "0",
                "0",
                "1",
                "1",
                "1",
                "Other",
                "3",
                "3",
                "3",
                "4",
                "4",
                "5",
                "0",
                "0",
                "1",
                "1",
                "1",
                "Other",
                "3",
                "3",
                "3",
                "4",
                "4",
                "5",
            ],
            "los_group": [
                "0 days",
                "0 days",
                "1 day",
                "1 day",
                "1 day",
                np.nan,
                "3 days",
                "3 days",
                "3 days",
                "4-7 days",
                "4-7 days",
                np.nan,
                "4-7 days",
                "4-7 days",
                "4-7 days",
                "4-7 days",
                "4-7 days",
                np.nan,
                "8-14 days",
                "8-14 days",
                "8-14 days",
                "8-14 days",
                "8-14 days",
                np.nan,
            ],
            "maternity_delivery_in_spell": [
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
            ]
            * 2,
            "measure": [
                "admissions",
                "beddays",
                "admissions",
                "beddays",
                "procedures",
                "walk-in",
                "admissions",
                "beddays",
                "procedures",
                "admissions",
                "beddays",
                "attendances",
                "admissions",
                "beddays",
                "admissions",
                "beddays",
                "procedures",
                "walk-in",
                "admissions",
                "beddays",
                "procedures",
                "admissions",
                "beddays",
                "attendances",
            ],
            "value": [
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                4,
                1,
                1,
                5,
                1,
                1,
                7,
                1,
                8,
                1,
                1,
                1,
                10,
                1,
                1,
                11,
                1,
            ],
        }
    )
    # act
    actual = mock_model.process_results(df)
    # assert
    pd.testing.assert_frame_equal(actual, expected)


def test_specific_aggregations(mocker, mock_model):
    """Test that it aggregates the results correctly."""
    # arrange
    m = mocker.patch("nhp.model.InpatientsModel.get_agg", return_value="agg_data")

    mdl = mock_model

    mock_data = pd.DataFrame({"maternity_delivery_in_spell": [True, False], "value": [1, 2]})

    # act
    actual = mdl.specific_aggregations(mock_data)

    # assert
    assert actual == {
        "sex+tretspef_grouped": "agg_data",
        "tretspef": "agg_data",
        "tretspef+los_group": "agg_data",
        "delivery_episode_in_spell": "agg_data",
    }

    assert [(len(i[0][0]), *i[0][1:]) for i in m.call_args_list] == [
        (2, "sex", "tretspef_grouped"),
        (2, "tretspef"),
        (2, "tretspef", "los_group"),
        (1,),
    ]


def test_save_results(mocker, mock_model):
    """Test that it correctly saves the results."""

    # arrange
    def path_fn(x):
        return x

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = "data"
    mr_mock.avoided_activity = pd.DataFrame(
        {
            "rn": [0],
            "classpat": [1],
            "speldur": [5],
        }
    )

    mock_model._save_results_get_ip_rows = Mock()
    mock_model._save_results_get_op_converted = Mock()
    mock_model._save_results_get_sdec_converted = Mock()

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")

    # act
    mock_model.save_results(mr_mock, path_fn)

    # assert
    mock_model._save_results_get_ip_rows.assert_called_once_with("data")
    mock_model._save_results_get_ip_rows().to_parquet.assert_called_once_with("ip/0.parquet")

    mock_model._save_results_get_op_converted.assert_called_once_with("data")
    mock_model._save_results_get_op_converted().to_parquet.assert_called_once_with(
        "op_conversion/0.parquet"
    )

    mock_model._save_results_get_sdec_converted.assert_called_once_with("data")
    mock_model._save_results_get_sdec_converted().to_parquet.assert_called_once_with(
        "sdec_conversion/0.parquet"
    )

    to_parquet_mock.assert_called_once_with("ip_avoided/0.parquet")


def test_save_results_get_op_converted(mock_model):
    """Test that _save_results_get_op_converted processes and returns the correct data."""
    # arrange
    data = pd.DataFrame(
        {
            "rn": [1, 2, 3, 4, 5, 6],
            "group": ["elective"] * 6,
            "classpat": ["-1", "1"] * 3,
            "speldur": [0, 1] * 3,
            "age": [1, 2] * 3,
            "age_group": ["a", "b"] * 3,
            "sex": [1, 2] * 3,
            "tretspef": ["a", "b"] * 3,
            "tretspef_grouped": ["a", "b"] * 3,
            "sitetret": ["1", "1", "1", "2", "2", "2"],
        }
    )
    expected = pd.DataFrame(
        {
            "age": [1, 1],
            "age_group": ["a", "a"],
            "sex": [1, 1],
            "tretspef": ["a", "a"],
            "tretspef_grouped": ["a", "a"],
            "sitetret": ["1", "2"],
            "attendances": [2, 1],
            "tele_attendances": [0, 0],
        }
    )

    # act
    actual = mock_model._save_results_get_op_converted(data)

    # assert
    pd.testing.assert_frame_equal(actual, expected)


def test_save_results_get_sdec_converted(mock_model):
    """Test that _save_results_get_op_converted processes and returns the correct data."""
    # arrange
    data = pd.DataFrame(
        {
            "rn": [1, 2, 3, 4, 5, 6],
            "group": ["elective"] * 6,
            "classpat": ["-3", "1"] * 3,
            "speldur": [0, 1] * 3,
            "age": [1, 2] * 3,
            "age_group": ["a", "b"] * 3,
            "sex": [1, 2] * 3,
            "tretspef": ["a", "b"] * 3,
            "tretspef_grouped": ["a", "b"] * 3,
            "sitetret": ["1", "1", "1", "2", "2", "2"],
        }
    )
    expected = pd.DataFrame(
        {
            "age": [1, 1],
            "age_group": ["a", "a"],
            "sex": [1, 1],
            "sitetret": ["1", "2"],
            "arrivals": [2, 1],
            "aedepttype": ["05", "05"],
            "attendance_category": ["1", "1"],
            "acuity": ["standard", "standard"],
            "group": ["walk-in", "walk-in"],
        }
    )

    # act
    actual = mock_model._save_results_get_sdec_converted(data)

    # assert
    pd.testing.assert_frame_equal(actual, expected)


def test_save_results_get_ip_rows(mock_model):
    """Test that _save_results_get_op_converted processes and returns the correct data."""
    # arrange
    data = pd.DataFrame(
        {
            "rn": [1, 2, 3, 4, 5, 6],
            "classpat": ["-3", "-2", "-1", "1", "2", "3"],
            "speldur": [0, 1, 2, 3, 4, 5],
            "sitetret": ["1", "1", "1", "2", "2", "2"],
        }
    )
    expected = pd.DataFrame(
        {
            "rn": [2, 4, 5, 6],
            "speldur": [1, 3, 4, 5],
            "classpat": ["2", "1", "2", "3"],
        }
    )

    # act
    actual = mock_model._save_results_get_ip_rows(data)

    # assert
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
