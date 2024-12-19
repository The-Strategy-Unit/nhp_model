"""test inpatients model"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring,unnecessary-lambda-assignment

from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.inpatients import InpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(InpatientsModel, "__init__", lambda s, p, d, h, r: None):
        mdl = InpatientsModel(None, None, None, None)
    mdl._data_loader = Mock()
    mdl.model_type = "ip"
    mdl.params = {
        "dataset": "synthetic",
        "model_runs": 3,
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
    mdl._data_path = "data/synthetic"
    # create a mock object for the hsa gams
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    mdl.hsa_gams = {(i, j): hsa_mock for i in ["aae_a_a", "aae_b_b"] for j in [1, 2]}
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
    """test that the model calls the super method and loads the strategies"""
    # arrange
    super_mock = mocker.patch("model.inpatients.super")
    # act
    InpatientsModel("params", "nhp_data", "hsa", "run_params")
    # assert
    super_mock.assert_called_once()


def test_add_ndggrp_to_data(mock_model):
    # arrange
    mdl = mock_model
    mdl.data = pd.DataFrame({"group": ["a", "b", "c"], "admimeth": ["80", "82", "83"]})

    # act
    mock_model._add_ndggrp_to_data()

    # assert
    assert mdl.data["ndggrp"].to_list() == ["a", "maternity", "maternity"]


def test_add_pod_to_data(mock_model):
    # arrange
    mock_model.data = pd.DataFrame(
        {
            "group": [
                "elective",
                "elective",
                "elective",
                "non-elective",
                "maternity",
                "elective",
                "maternity",
            ],
            "classpat": ["1", "2", "3", "1", "1", "4", "5"],
        }
    )
    # act
    mock_model._add_pod_to_data()

    # assert
    assert mock_model.data["pod"].to_list() == [
        "ip_elective_admission",
        "ip_elective_daycase",
        "ip_regular_day_attender",
        "ip_non-elective_admission",
        "ip_maternity_admission",
        "ip_regular_night_attender",
        "ip_maternity_admission",
    ]


@pytest.mark.parametrize(
    "test, expected",
    [
        (True, ["ip_regular_day_attender", "ip_regular_night_attender"]),
        (False, ["ip_elective_daycase", "ip_elective_admission"]),
    ],
)
def test_add_pod_to_data_separate_regular_day_attenders_param(
    mock_model, test, expected
):
    # arrange
    mock_model.data = pd.DataFrame(
        {
            "group": ["elective", "elective"],
            "classpat": ["3", "4"],
        }
    )
    mock_model.params["separate_regular_attenders"] = test

    # act
    mock_model._add_pod_to_data()

    # assert
    assert mock_model.data["pod"].to_list() == expected


def test_add_pod_to_data_no_regular_attenders(mock_model):
    # arrange
    mock_model.data = pd.DataFrame(
        {
            "group": ["elective", "elective"],
            "classpat": ["1", "2"],
        }
    )

    # act
    mock_model._add_pod_to_data()

    # assert
    assert mock_model.data["pod"].to_list() == [
        "ip_elective_admission",
        "ip_elective_daycase",
    ]


def test_get_data(mock_model):
    # arrange
    mdl = mock_model
    mdl._data_loader.get_ip.return_value = "ip data"

    # act
    actual = mdl._get_data()

    # assert
    assert actual == "ip data"
    mdl._data_loader.get_ip.assert_called_once_with()


@pytest.mark.parametrize(
    "gen_los_type, expected",
    [
        ("general_los_reduction_elective", [7, 9]),
        ("general_los_reduction_emergency", [8, 10]),
    ],
)
def test_load_strategies(mock_model, gen_los_type, expected):
    """test that the method returns a dataframe"""
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
    mdl._data_loader.get_ip_strategies.return_value = {
        "activity_avoidance": pd.DataFrame(
            {"rn": [1, 2, 3], "admission_avoidance_strategy": ["a", "b", "c"]}
        ),
        "efficiencies": pd.DataFrame(
            {"rn": [4, 5, 6], "los_reduction_strategy": ["a", "b", "c"]}
        ),
    }
    expected = {
        "activity_avoidance": {"strategy": {1: "a", 2: "b"}},
        "efficiencies": {
            "strategy": {5: "b", 6: "c", **{i: gen_los_type for i in expected}}
        },
    }

    # act
    mdl._load_strategies()

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
    """test that it runs the model steps"""

    mdl = mock_model

    mock = mocker.patch("model.inpatients.InpatientEfficiencies")
    mock.return_value = mock
    mock.data = "data_efficiencies"

    mock.losr_all.return_value = mock
    mock.losr_aec.return_value = mock
    mock.losr_preop.return_value = mock
    mock.losr_day_procedures.return_value = mock
    mock.update_step_counts.return_value = mock

    mock.get_step_counts.return_value = "step_counts"

    mock_model_run = Mock()
    mock_model_run.empty = False
    mock_model_run.model.strategies = {"efficiencies": mock_model_run}

    # act
    actual = mdl.efficiencies("data", mock_model_run)

    # assert
    assert actual == ("data_efficiencies", "step_counts")

    mock.assert_called_once_with("data", mock_model_run)
    mock.losr_all.assert_called_once()
    mock.losr_aec.assert_called_once()
    mock.losr_preop.assert_called_once()
    assert mock.losr_day_procedures.call_count == 2
    assert mock.losr_day_procedures.call_args_list == [
        call("day_procedures_daycase"),
        call("day_procedures_outpatients"),
    ]
    mock.get_step_counts.assert_called_once()


def test_efficiencies_no_params(mocker, mock_model):
    """test that it runs the model steps"""

    mdl = mock_model

    mock = mocker.patch("model.inpatients.InpatientEfficiencies")
    mock.return_value = mock

    mock_model_run = Mock()
    mock_model_run.empty = True
    mock_model_run.model.strategies = {"efficiencies": mock_model_run}

    # act
    actual = mdl.efficiencies("data", mock_model_run)

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
            "classpat": ["1", "-2", "3", "4", "5", "-1"] * 2,
            "tretspef": xs,
            "tretspef_raw": list(range(12)),
            "rn": [1] * 12,
            "has_procedure": [0, 1] * 6,
            "speldur": list(range(12)),
        }
    )
    df["pod"] = "ip_" + df["group"] + "_admission"

    expected = {
        "sitetret": ["trust"] * 26,
        "age": list(range(12)) + [i for i in range(11) if i != 5] + [1, 3, 7, 9],
        "age_group": list(range(6)) * 2 + list(range(5)) * 2 + [1, 3] * 2,
        "sex": list(range(6)) * 2 + list(range(5)) * 2 + [1, 3] * 2,
        "tretspef": list(range(6)) * 2 + list(range(5)) * 2 + [1, 3] * 2,
        "tretspef_raw": list(range(12))
        + [i for i in range(11) if i != 5]
        + [1, 3, 7, 9],
        "measure": (["admissions"] * 5 + ["attendances"]) * 2
        + ["beddays"] * 10
        + ["procedures"] * 4,
        "value": [1] * 12 + [i for i in range(1, 12) if i != 6] + [1] * 4,
        "pod": [
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_maternity_admission",
            "ip_elective_admission",
            "ip_non-elective_admission",
            "op_procedure",
        ]
        * 2
        + [
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_maternity_admission",
            "ip_elective_admission",
            "ip_non-elective_admission",
        ]
        * 2
        + [
            "ip_elective_daycase",
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_elective_admission",
        ],
        "los_group": ["0 days", "1 day", "2 days", "3 days"]
        + ["4-7 days"] * 4
        + ["8-14 days"] * 4
        + ["0 days", "1 day", "2 days", "3 days"]
        + ["4-7 days"] * 3
        + ["8-14 days"] * 3
        + ["1 day", "3 days", "4-7 days", "8-14 days"],
    }
    # act
    actual = mock_model.process_results(df)
    # assert
    assert actual.to_dict("list") == expected


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: model_results.to_dict(orient="list")}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl.process_results = Mock(return_value="processed_data")

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = "model_data"

    # act
    actual_mr, actual_aggs = mdl.aggregate(mr_mock)

    # assert

    mdl.process_results.assert_called_once_with("model_data")
    assert actual_mr == "processed_data"
    assert actual_aggs == [
        ["sex", "tretspef"],
        ["tretspef_raw"],
        ["tretspef_raw", "los_group"],
    ]


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    # arrange
    path_fn = lambda x: x

    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame(
        {
            "rn": [0],
            "classpat": [1],
            "age": [2],
            "sex": [3],
            "tretspef": [4],
            "speldur": [5],
            "tretspef_raw": [6],
            "sitetret": [7],
        }
    )
    mr_mock.avoided_activity = pd.DataFrame(
        {
            "rn": [0],
            "classpat": [1],
            "speldur": [5],
        }
    )

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")

    # act
    mock_model.save_results(mr_mock, path_fn)

    # assert
    assert to_parquet_mock.call_count == 3
    assert to_parquet_mock.call_args_list[0] == call("op_conversion/0.parquet")
    assert to_parquet_mock.call_args_list[1] == call("ip/0.parquet")
    assert to_parquet_mock.call_args_list[2] == call("ip_avoided/0.parquet")
