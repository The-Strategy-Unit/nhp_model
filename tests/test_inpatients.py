"""test inpatients model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from datetime import datetime, timedelta
from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.inpatients import InpatientsModel


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(InpatientsModel, "__init__", lambda s, p, d: None):
        mdl = InpatientsModel(None, None)
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
        "bed_occupancy": {
            "a": {"a": [0.4, 0.6], "b": 0.7},
            "b": {"a": [0.4, 0.6], "b": 0.8},
        },
        "theatres": {
            "change_utilisation": {
                "100": {
                    "baseline": 1.01,
                    "interval": [1.02, 1.04],
                },
                "110": {"baseline": 1.11, "interval": [1.12, 1.14]},
                "Other (Surgical)": {"baseline": 1.21, "interval": [1.22, 1.24]},
            },
            "change_availability": [1.03, 1.05],
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
    mocker.patch("model.inpatients.InpatientsModel._load_theatres_data")
    mocker.patch("model.inpatients.InpatientsModel._load_kh03_data")
    # act
    mdl = InpatientsModel("params", "data_path")
    # assert
    super_mock.assert_called_once()
    mdl._load_theatres_data.assert_called_once()
    mdl._load_kh03_data.assert_called_once()


def test_get_data_mask(mock_model):
    # arrange
    expected = [True] * 3 + [False] * 2
    mock_model.data = pd.DataFrame({"bedday_rows": [not i for i in expected]})

    # act
    actual = mock_model._get_data_mask()

    # assert
    assert actual.tolist() == expected


def test_load_kh03_data(mocker, mock_model):
    """test that the kh03 data is loaded correctly"""
    # arrange
    mdl = mock_model
    mdl.params["bed_occupancy"] = {
        "specialty_mapping": {
            "General and Acute": {"100": "a", "200": "b", "300": "b", "500": "_"},
            "Maternity": {"501": "c"},
        }
    }
    mdl._bedday_summary = Mock(return_value="beds_baseline")
    mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "quarter": ["q1"] * 4 + ["q2"],
                "specialty_code": ["100", "200", "300", "501", "100"],
                "available": [1, 2, 3, 4, 5],
                "occupied": [4, 5, 6, 7, 8],
            }
        ),
    )
    mdl.data = "data"
    # act
    mdl._load_kh03_data()
    # assert
    assert mdl._ward_groups.to_dict("index") == {
        "100": {"ward_type": "General and Acute", "ward_group": "a"},
        "200": {"ward_type": "General and Acute", "ward_group": "b"},
        "300": {"ward_type": "General and Acute", "ward_group": "b"},
        "500": {"ward_type": "General and Acute", "ward_group": "_"},
        "501": {"ward_type": "Maternity", "ward_group": "c"},
    }
    assert mdl._kh03_data.to_dict("index") == {
        ("q1", "a"): {"available": 1, "occupied": 4},
        ("q1", "b"): {"available": 5, "occupied": 11},
        ("q1", "c"): {"available": 4, "occupied": 7},
        ("q2", "a"): {"available": 5, "occupied": 8},
    }
    mdl._bedday_summary.assert_called_once_with("data", 2018)
    assert mdl._beds_baseline == "beds_baseline"


def test_load_theatres_data(mocker, mock_model):
    """test that it loads the json file correctly"""
    json_mock = mocker.patch(
        "json.load",
        return_value={
            "theatres": 10,
            "four_hour_sessions": {"100": 1, "200": 2, "Other (Surgical)": 3},
        },
    )
    mock_model.data["has_procedure"] = [i for i in [0, 1] for _j in range(10)]
    mock_model.data["tretspef"] = ["100", "200"] * 10
    with patch("builtins.open", mock_open()) as mock_file:
        mock_model._load_theatres_data()
        json_mock.assert_called_once()
        mock_file.assert_called_with(
            "data/synthetic/theatres.json", "r", encoding="UTF-8"
        )
        assert mock_model._theatres_data["theatres"] == 10
        assert mock_model._theatres_data["four_hour_sessions"].to_dict() == {
            "100": 1,
            "200": 2,
            "Other (Surgical)": 3,
        }
        assert mock_model._procedures_baseline.to_dict() == {"100": 5, "200": 5}
        assert mock_model._theatre_spells_baseline == 3.469437852876197


def test_load_strategies(mock_model):
    """test that the method returns a dataframe"""
    # arrange
    mdl = mock_model
    mdl.params = {
        "activity_avoidance": {"ip": {"a": 1, "b": 2}},
        "efficiencies": {"ip": {"b": 3, "c": 4}},
    }

    mdl._load_parquet = Mock()
    mdl._load_parquet.side_effect = [
        pd.DataFrame(
            {"rn": [1, 2, 3], "admission_avoidance_strategy": ["a", "b", "c"]}
        ),
        pd.DataFrame({"rn": [4, 5, 6], "los_reduction_strategy": ["a", "b", "c"]}),
    ]
    expected = {
        "activity_avoidance": {"strategy": {1: "a", 2: "b"}},
        "efficiencies": {"strategy": {5: "b", 6: "c"}},
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
    actual = mdl._get_data_counts(data)

    # assert
    assert actual.tolist() == [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]


def test_apply_resampling(mocker, mock_model):
    # arrange
    row_samples = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    gdc_mock = mocker.patch(
        "model.inpatients.InpatientsModel._get_data_counts",
        return_value=np.array([1, 1, 1, 1, 1, 1]),
    )
    data = pd.DataFrame(
        {"rn": [0, 1, 2, 3], "bedday_rows": [False, False, False, True]}
    )
    # act
    data, counts = mock_model.apply_resampling(row_samples, data)
    # assert
    assert data["rn"].to_list() == [1, 2, 2, 3, 3, 3]
    assert counts.sum() == 3.0
    gdc_mock.assert_called_once()


def test_get_step_counts_dataframe(mock_model):
    # arrange
    step_counts = {("a", "-"): [1, 2], ("b", "-"): [3, 4]}
    expected = {
        "change_factor": ["a", "b", "a", "b"],
        "strategy": ["-", "-", "-", "-"],
        "measure": ["admissions", "admissions", "beddays", "beddays"],
        "value": [1, 3, 2, 4],
    }

    # act
    actual = mock_model.get_step_counts_dataframe(step_counts)

    # assert
    assert actual.to_dict("list") == expected


def test_run(mocker, mock_model):
    """test that it runs the model steps"""

    mdl = mock_model

    aa_mock = mocker.patch("model.inpatients.InpatientEfficiencies")
    aa_mock.return_value = aa_mock
    aa_mock.losr_all.return_value = aa_mock
    aa_mock.losr_aec.return_value = aa_mock
    aa_mock.losr_preop.return_value = aa_mock
    aa_mock.losr_bads.return_value = aa_mock

    # act
    mdl._run("model_run")

    # assert
    aa_mock.assert_called_once_with("model_run")
    aa_mock.losr_all.assert_called_once()
    aa_mock.losr_aec.assert_called_once()
    aa_mock.losr_preop.assert_called_once()
    aa_mock.losr_bads.assert_called_once()


def test_bedday_summary(mock_model):
    """test that it aggregates the data to the maximum beds occupied in a day per quarter"""
    # arrange
    mdl = mock_model
    mdl._ward_groups = pd.DataFrame(
        {
            "ward_type": ["general_and_acute"] * 3,
            "ward_group": ["A", "A", "B"],
        },
        index=["a", "b", "c"],
    )
    data = pd.DataFrame(
        {
            "admidate": [f"2022-04-0{d}" for d in range(6)] * 2,
            "speldur": list(range(6)) * 2,
            "mainspef": ["a", "b", "c"] * 4,
            "classpat": (["1"] * 5 + ["2"]) * 2,
        }
    )
    # act
    actual = mdl._bedday_summary(data, 2022)
    # assert
    assert actual.to_dict("list") == {
        "quarter": ["q1", "q1"],
        "ward_type": ["general_and_acute", "general_and_acute"],
        "ward_group": ["A", "B"],
        "size": [2.6666666666666665, 2.0],
    }


def test_bed_occupancy(mock_model):
    """test that it aggregates the bed occupancy data"""
    # arrange
    mdl = mock_model

    mr_mock = Mock()
    mr_mock.run_params = {
        "bed_occupancy": {"day+night": {"A": 0.75, "B": 0.875, "C": 0.5}}
    }
    mr_mock.model_run = 0

    mdl._bedday_summary = Mock(
        return_value=pd.DataFrame(
            {
                "quarter": ["q1"] * 3 + ["q2"] * 3,
                "ward_type": ["general_and_acute"] * 6,
                "ward_group": ["A", "B", "C"] * 2,
                "size": [4, 2, 1, 8, 4, 2],
            }
        )
    )
    mdl._kh03_data = pd.DataFrame(
        {
            "available": [10, 50, 20, 100],
            "occupied": [5, 40, 10, 80],
        },
        index=pd.MultiIndex.from_tuples(
            [(q, k) for q in ["q1", "q2"] for k in ["A", "B"]]
        ),
    )
    mdl._beds_baseline = pd.DataFrame(
        {
            "quarter": ["q1"] * 3 + ["q2"] * 3,
            "ward_type": ["general_and_acute"] * 6,
            "ward_group": ["A", "B", "C"] * 2,
            "size": [2, 1, 1, 4, 2, 2],
        }
    )

    # act
    actual = mdl._bed_occupancy("model_results", mr_mock)

    # assert
    assert {tuple(k): v for k, v in actual.items()} == {
        ("ip", "day+night", "q1", "A"): 13,
        ("ip", "day+night", "q1", "B"): 91,
        ("ip", "day+night", "q1", "C"): 0,
        ("ip", "day+night", "q2", "A"): 27,
        ("ip", "day+night", "q2", "B"): 183,
        ("ip", "day+night", "q2", "C"): 0,
    }
    mdl._bedday_summary.assert_called_once_with("model_results", 2018)


def test_bed_occupancy_baseline(mock_model):
    """test that it just uses the kh03 available data"""
    # arrange
    mdl = mock_model

    mr_mock = Mock()
    mr_mock.run_params = {
        "bed_occupancy": {"day+night": {"A": 0.75, "B": 0.875, "C": 0.5}}
    }
    mr_mock.model_run = -1

    mdl._kh03_data = pd.DataFrame(
        {
            "available": [10, 50, 20, 100],
            "occupied": [5, 40, 10, 80],
        },
        index=pd.MultiIndex.from_tuples(
            [(q, k) for q in ["q1", "q2"] for k in ["A", "B"]]
        ),
    )
    # act
    actual = mock_model._bed_occupancy("model_results", mr_mock)

    # assert
    assert {tuple(k): v for k, v in actual.items()} == {
        ("ip", "day+night", "q1", "A"): 10,
        ("ip", "day+night", "q1", "B"): 50,
        ("ip", "day+night", "q2", "A"): 20,
        ("ip", "day+night", "q2", "B"): 100,
    }


def test_theatres_available(mock_model):
    """test that it aggregates the theatres data"""
    # arrange
    mdl = mock_model

    mr_mock = Mock()
    mr_mock.run_params = {
        "theatres": {
            "change_utilisation": {"100": 2, "110": 2.5, "Other (Surgical)": 3},
            "change_availability": 5,
        }
    }
    mr_mock.model_run = 0

    mdl._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other (Surgical)": 300}, name="four_hour_sessions"
        ),
    }
    mdl._procedures_baseline = pd.Series(
        {"100": 2, "110": 3, "200": 4, "Other (Surgical)": 5}
    )

    mdl._theatre_spells_baseline = 1000

    model_results = pd.DataFrame(
        [
            {"tretspef": t, "measure": p}
            for t in ["100", "110", "200", "Other (Surgical)"]
            for p in ["x"] + ["procedures"] * 2
        ]
    )
    model_results["value"] = list(range(len(model_results)))
    # this makes the results nicer numbers
    model_results["value"] *= 3

    # act
    actual = mdl._theatres_available(model_results, mr_mock)

    # assert
    assert {tuple(k): v for k, v in actual.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 450.0,
        ("ip_theatres", "four_hour_sessions", "110"): 1800.0,
        ("ip_theatres", "four_hour_sessions", "Other (Surgical)"): 3780.0,
        ("ip_theatres", "theatres"): 4.41,
    }


def test_theatres_available_baseline(mock_model):
    """test that it returns the baseline theatres data"""
    # arrange

    mr_mock = Mock()
    mr_mock.run_params = {
        "theatres": {
            "change_utilisation": {"100": 2, "110": 2.5, "Other (Surgical)": 3},
            "change_availability": 5,
        }
    }
    mr_mock.model_run = -1

    mock_model._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other (Surgical)": 300}, name="four_hour_sessions"
        ),
    }

    # act
    theatres_available = mock_model._theatres_available(None, mr_mock)

    # assert
    assert {tuple(k): v for k, v in theatres_available.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 100,
        ("ip_theatres", "four_hour_sessions", "110"): 200,
        ("ip_theatres", "four_hour_sessions", "Other (Surgical)"): 300,
        ("ip_theatres", "theatres"): 10,
    }


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""
    # arrange
    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: model_results.to_dict(orient="list")}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl._get_run_params = Mock(
        return_value={"bed_occupancy": "run_params", "theatres": "theatres"}
    )
    mdl._bed_occupancy = Mock(return_value=2)
    mdl._theatres_available = Mock(return_value=3)
    xs = list(range(6)) * 2
    mr_mock = Mock()
    mr_mock.get_model_results.return_value = pd.DataFrame(
        {
            "sitetret": ["trust"] * 24,
            "age_group": xs * 2,
            "sex": xs * 2,
            "group": ["elective", "non-elective", "maternity"] * 8,
            "classpat": ["1", "2", "3", "4", "5", "-1"] * 4,
            "tretspef": xs * 2,
            "rn": [1] * 24,
            "has_procedure": [0, 1] * 12,
            "speldur": list(range(12)) * 2,
            "bedday_rows": [False] * 12 + [True] * 12,
        }
    )

    expected_mr = {
        "sitetret": [
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
            "trust",
        ],
        "age_group": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "sex": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "group": [
            "elective",
            "non-elective",
            "maternity",
            "elective",
            "non-elective",
            "maternity",
            "elective",
            "non-elective",
            "maternity",
            "elective",
            "non-elective",
            "elective",
            "non-elective",
            "maternity",
            "elective",
            "non-elective",
        ],
        "classpat": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "-1",
            "1",
            "2",
            "3",
            "4",
            "5",
            "1",
            "2",
            "3",
            "4",
            "5",
        ],
        "tretspef": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "measure": [
            "admissions",
            "admissions",
            "admissions",
            "admissions",
            "admissions",
            "attendances",
            "procedures",
            "procedures",
            "procedures",
            "procedures",
            "procedures",
            "beddays",
            "beddays",
            "beddays",
            "beddays",
            "beddays",
        ],
        "value": [2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 8, 10, 12, 14, 16],
        "pod": [
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_elective_daycase",
            "ip_elective_admission",
            "ip_non-elective_admission",
            "op_procedure",
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_elective_daycase",
            "ip_elective_admission",
            "ip_non-elective_admission",
            "ip_elective_admission",
            "ip_elective_daycase",
            "ip_elective_daycase",
            "ip_elective_admission",
            "ip_non-elective_admission",
        ],
    }

    # act
    (agg, results) = mdl._aggregate(mr_mock)

    # assert
    assert agg() == {"default": expected_mr}
    assert results == {
        "sex+tretspef": expected_mr,
        "bed_occupancy": 2,
        "theatres_available": 3,
    }


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
        }
    )

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")

    # act
    mock_model.save_results(mr_mock, path_fn)

    # assert
    assert to_parquet_mock.call_count == 2
    assert to_parquet_mock.call_args_list[0] == call("op_conversion/0.parquet")
    assert to_parquet_mock.call_args_list[1] == call("ip/0.parquet")
