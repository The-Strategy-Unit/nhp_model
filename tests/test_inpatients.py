"""test inpatients model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

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
            "change_utilisation": {"a": [1.01, 1.03], "b": [1.02, 1.04]},
            "change_availability": [1.03, 1.05],
        },
    }
    mdl._data_path = "data/synthetic"
    # create a mock object for the hsa gams
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    mdl._hsa_gams = {(i, j): hsa_mock for i in ["aae_a_a", "aae_b_b"] for j in [1, 2]}
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


@pytest.fixture
def mock_losr():
    return pd.DataFrame(
        {
            "type": [x for x in ["all", "aec", "pre-op"] for _ in [0, 1]]
            + ["bads"] * 3,
            "baseline_target_rate": [pd.NA] * 6 + [0.25, 0.50, 0.75],
            "op_dc_split": [pd.NA] * 6 + [0, 1, 0.5],
            "pre-op_days": [pd.NA] * 4 + [1, 2] + [pd.NA] * 3,
            "losr_f": [1 - 1 / (2**x) for x in range(9)],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    )


# methods


def test_init_calls_super_init(mocker):
    """test that the model calls the super method and loads the strategies"""
    mocker.patch("model.inpatients.super")
    mocker.patch(
        "model.inpatients.InpatientsModel._load_parquet", wraps=lambda _: Mock()
    )
    mocker.patch("model.inpatients.InpatientsModel._load_theatres_data")
    mocker.patch("model.inpatients.InpatientsModel._load_kh03_data")
    mocker.patch("model.inpatients.InpatientsModel._union_bedday_rows")
    mdl = InpatientsModel("params", "data_path")
    # no asserts to perform, so long as this method doesn't fail
    assert mdl._load_parquet.call_count == 2
    assert (
        mdl._load_parquet.call_args_list[0][0][0] == "ip_admission_avoidance_strategies"
    )
    assert mdl._load_parquet.call_args_list[1][0][0] == "ip_los_reduction_strategies"
    mdl._load_theatres_data.assert_called_once()
    mdl._load_kh03_data.assert_called_once()
    mdl._union_bedday_rows.assert_called_once()


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
            "four_hour_sessions": {"100": 1, "200": 2, "Other": 3},
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
        assert mock_model._theatres_data["four_hour_sessions"].equals(
            pd.Series({"100": 1, "200": 2, "Other": 3}, name="four_hour_sessions")
        )
        assert mock_model._theatres_baseline.equals(
            pd.DataFrame(
                {"tretspef": ["100", "200"], "is_elective": [5, 0], "n": [5, 5]}
            )
        )


def test_los_reduction(mock_model):
    """test that the method returns a dataframe"""
    mdl = mock_model
    actual = mdl._los_reduction(
        {"inpatient_factors": {"los_reduction": {"b_a": 0.45, "b_b": 0.55}}}
    )
    assert actual.equals(
        pd.DataFrame(
            {"interval": [[0.4, 0.6], [0.4, 0.6]], "losr_f": [0.45, 0.55]},
            index=["b_a", "b_b"],
        )
    )


def test_random_strategy(mocker, mock_model):
    """test that it selects one random strategy"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = np.asarray([1, 1, 0, 1, 1, 0, 1, 1])
    rng.bit_generator = np.random.default_rng(1).bit_generator
    mocker.patch("pandas.DataFrame.sample", wraps=lambda x, *_: x)
    mdl = mock_model
    mdl._strategies = {
        "a": pd.DataFrame(
            {
                "strategy": ["a", "b", "c", "NULL", "b", "c", "d", "NULL"],
                "sample_rate": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            index=[1, 1, 1, 1, 2, 2, 2, 2],
        )
    }
    mdl.params = {"inpatient_factors": {"a": {"a": 1, "b": 2, "c": 3}}}
    # act
    actual = mdl._random_strategy(rng, "a")
    # assert
    assert actual.equals(pd.Series(["NULL", "a"], index=[2, 1]))


def test_waiting_list_adjustment(mock_model):
    """test that it returns the wla numpy array"""
    data = pd.DataFrame(
        {
            "tretspef": [1] * 2 + [2] * 8 + [3] * 4 + [4] * 1,
            "admimeth": ["11", "21"] * 7 + ["11"],
        }
    )
    mdl = mock_model
    mdl.params["waiting_list_adjustment"] = {"ip": {1: 1, 2: 2, 3: 3, 5: 1}}
    actual = mdl._waiting_list_adjustment(data)
    expected = [1.5, 1] + [1.25, 1] * 4 + [1.75, 1] * 2 + [1]
    assert np.array_equal(actual, expected)


def test_non_demographic_adjustment(mock_model):
    """test that is returns the nda numpy array"""
    run_params = {
        "non-demographic_adjustment": {
            "elective": {
                0: 1,
                1: 2,
            },
            "maternity": {
                0: 3,
                1: 4,
            },
            "non-elective": {
                0: 5,
                1: 6,
            },
        }
    }
    data = pd.DataFrame(
        {
            "admigroup": [
                i
                for i in ["elective", "non-elective", "maternity", "non-elective"]
                for _ in [0, 1]
            ],
            "age_group": [0, 1] * 4,
        }
    )
    actual = mock_model._non_demographic_adjustment(data, run_params)
    expected = [1, 2, 5, 6, 3, 4, 5, 6]
    assert list(actual) == expected


def test_admission_avoidance(mock_model):
    """test admission avoidance returns the correct factors"""
    # arrange
    x = list(range(9))
    admission_avoidance = pd.Series(
        ["a_a", "a_b", "NULL"] * 3, index=x, name="admission_avoidance_strategy"
    )
    run_params = {
        "inpatient_factors": {"admission_avoidance": {"a_a": 0.25, "a_b": 0.75}}
    }
    # act
    actual = mock_model._admission_avoidance(admission_avoidance, run_params)
    # assert
    assert actual == [0.25, 0.75, 1.0] * 3


def test_losr_all(mock_model, mock_losr):
    """test that it reduces the speldur column for 'all' types"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = list(range(6))
    losr = mock_losr
    data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "a", "b"] * 3)
    step_counts = {}
    # act
    mock_model._losr_all(data, losr, rng, step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 0, 3, 3, 1, 4, 6, 2, 5]
    assert step_counts == {
        ("los_reduction", "a"): {"admissions": 0, "beddays": -9},
        ("los_reduction", "b"): {"admissions": 0, "beddays": -3},
    }
    assert rng.binomial.call_args_list[0][0][0].to_list() == [1, 4, 7, 2, 5, 8]
    assert rng.binomial.call_args_list[0][0][1].to_list() == [0, 0, 0, 0.5, 0.5, 0.5]


def test_losr_bads(mock_model, mock_losr):
    """test that it reduces the speldur column for 'bads' types"""
    rng = Mock()
    # 1: rvr < ur0
    # 2: rvr < ur0 + ur1
    # 3: rvr >= ur0 + ur1
    rng.uniform.return_value = (
        [0.02, 0.02, 0.01] + [0.04, 0.03, 1.0] + [0.04, 0.03, 0.55]
    )
    losr = mock_losr
    data = pd.DataFrame(
        {"speldur": list(range(12)), "classpat": ["1"] * 6 + ["2"] * 3 + ["1"] * 3},
        index=[x for x in ["x", "g", "h", "i"] for _ in range(3)],
    )
    step_counts = {}
    # act
    mock_model._losr_bads(data, losr, rng, step_counts)
    # assert

    assert data["speldur"].to_list() == [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]
    assert step_counts == {
        ("los_reduction", "g"): {"admissions": 0, "beddays": 0},
        ("los_reduction", "h"): {"admissions": -1, "beddays": -21},
        ("los_reduction", "i"): {"admissions": -1, "beddays": -30},
    }
    assert rng.uniform.call_args_list[0][1] == {"size": 9}


def test_losr_aec(mock_model, mock_losr):
    """test that it reduces the speldur column for 'aec' types"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = [0, 0, 0, 1, 1, 1]
    losr = mock_losr
    data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "c", "d"] * 3)
    step_counts = {}
    # act
    mock_model._losr_aec(data, losr, rng, step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 0, 2, 3, 0, 5, 6, 0, 8]
    assert step_counts == {
        ("los_reduction", "c"): {"admissions": 0, "beddays": -12},
        ("los_reduction", "d"): {"admissions": 0, "beddays": 0},
    }
    assert rng.binomial.call_args_list[0][0][0] == 1
    assert rng.binomial.call_args_list[0][0][1].equals(
        losr.loc[["c"] * 3 + ["d"] * 3, "losr_f"]
    )


def test_losr_preop(mock_model, mock_losr):
    """test that is reduces the speldur column for 'pre-op' types"""
    # arrange
    rng = Mock()
    rng.binomial.return_value = [0, 1, 0, 1, 0, 1]
    losr = mock_losr
    data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "e", "f"] * 3)
    step_counts = {}
    # act
    mock_model._losr_preop(data, losr, rng, step_counts)
    # assert
    assert data["speldur"].to_list() == [0, 1, 0, 3, 3, 5, 6, 7, 6]
    assert step_counts == {
        ("los_reduction", "e"): {"admissions": 0, "beddays": -1},
        ("los_reduction", "f"): {"admissions": 0, "beddays": -4},
    }
    assert rng.binomial.call_args_list[0][0][0] == 1
    assert rng.binomial.call_args_list[0][0][1].equals(
        1 - losr.loc[["e"] * 3 + ["f"] * 3, "losr_f"]
    )


def test_expat_adjustment():
    """ "test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame(
        {
            "tretspef": ["100", "120", "Other"] * 2,
            "admigroup": [i for i in ["elective", "non-elective"] for _ in range(3)],
        }
    )
    run_params = {
        "expat": {
            "ip": {
                "elective": {"100": 0.6, "120": 0.7},
                "non-elective": {"100": 0.8, "120": 0.9},
            }
        }
    }
    # act
    actual = InpatientsModel._expat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [0.6, 0.7, 1.0, 0.8, 0.9, 1.0]


def test_repat_adjustment():
    """test that it returns the right parameters"""
    # arrange
    data = pd.DataFrame(
        {
            "tretspef": ["100", "120", "Other"] * 4,
            "admigroup": [i for i in ["elective", "non-elective"] for _ in range(3)]
            * 2,
            "is_main_icb": [i for i in [True, False] for _ in range(6)],
        }
    )
    run_params = {
        "repat_local": {
            "ip": {
                "elective": {"100": 1.1, "120": 1.2},
                "non-elective": {"100": 1.3, "120": 1.4},
            }
        },
        "repat_nonlocal": {
            "ip": {
                "elective": {"100": 1.5, "120": 1.6},
                "non-elective": {"100": 1.7, "120": 1.8},
            }
        },
    }
    # act
    actual = InpatientsModel._repat_adjustment(data, run_params)
    # assert
    assert actual.tolist() == [
        1.1,
        1.2,
        1.0,
        1.3,
        1.4,
        1.0,
        1.5,
        1.6,
        1.0,
        1.7,
        1.8,
        1.0,
    ]


def test_step_counts(mock_model):
    """test that it estimates the step counts correctly"""
    # arrange
    mdl = mock_model
    admissions = pd.Series([0, 1, 2, 3, 4, 5])
    beddays_before = admissions + 1
    beddays_after = (admissions * beddays_before).sum()
    admission_avoidance = pd.Series(
        [0.5 + x / 16 for x in admissions], index=["a", "a", "b", "b", "c", "c"]
    )
    factors = {
        "x": [1.0 + x / 16 for x in admissions],
        "y": [1.5 + x / 16 for x in admissions],
    }
    # act
    actual = mdl._step_counts(
        admissions, beddays_before, beddays_after, admission_avoidance, factors
    )
    # assert
    assert actual == {
        ("baseline", "-"): {"admissions": 6, "beddays": 21},
        ("admission_avoidance", "a"): {
            "admissions": 4.746600741656366,
            "beddays": 6.5056105610561055,
        },
        ("admission_avoidance", "b"): {
            "admissions": 3.480840543881335,
            "beddays": 11.236963696369637,
        },
        ("admission_avoidance", "c"): {
            "admissions": 2.215080346106304,
            "beddays": 11.236963696369637,
        },
        ("x", "-"): {"admissions": 3.4610630407910996, "beddays": 15.524752475247524},
        ("y", "-"): {"admissions": 15.981458590852904, "beddays": 62.454785478547855},
    }


def test_run(mock_model):
    """test that it runs the model steps"""
    # arrange
    data = pd.DataFrame({"rn": [1, 2], "speldur": [0, 1], "hsagrp": ["a", "b"]})
    mdl = mock_model
    mdl._random_strategy = Mock(
        return_value=pd.Series(["a", "b"], index=data["rn"]).rename_axis("rn")
    )
    mdl._los_reduction = Mock()
    mdl._expat_adjustment = Mock(return_value=pd.Series([5, 6]))
    mdl._repat_adjustment = Mock(return_value=pd.Series([7, 8]))
    mdl._waiting_list_adjustment = Mock(return_value=np.array([9, 10]))
    mdl._non_demographic_adjustment = Mock(return_value=np.array([11, 12]))
    mdl._admission_avoidance = Mock(return_value=pd.Series([13, 14], index=data["rn"]))
    mdl._step_counts = Mock(
        return_value={("baseline", "-"): {"admissions": 2, "beddays": 3}}
    )
    mdl._losr_all = Mock()
    mdl._losr_aec = Mock()
    mdl._losr_preop = Mock()
    mdl._losr_bads = Mock()
    rng = Mock()
    rng.poisson.return_value = [1, 2]
    # act
    change_factors, model_results = mdl._run(
        rng,
        data,
        {"inpatient_factors": {"admission_avoidance": {"a": 1, "b": 2}}},
        pd.Series([1, 2], index=data["rn"]),
        np.array([3, 4]),
    )
    # assert
    assert change_factors.to_dict() == {
        "change_factor": {0: "baseline", 1: "baseline"},
        "strategy": {0: "-", 1: "-"},
        "measure": {0: "admissions", 1: "beddays"},
        "value": {0: 2, 1: 3},
    }
    assert model_results.equals(
        data.loc[[0, 1, 1]].drop("hsagrp", axis="columns").reset_index(drop=True)
    )
    assert mdl._random_strategy.call_count == 2
    rng.poisson.assert_called_once()
    assert rng.poisson.call_args_list[0][0][0].to_list() == [135135, 645120]
    mdl._los_reduction.assert_called_once()
    mdl._expat_adjustment.assert_called_once()
    mdl._repat_adjustment.assert_called_once()
    mdl._waiting_list_adjustment.assert_called_once()
    mdl._non_demographic_adjustment.assert_called_once()
    mdl._admission_avoidance.assert_called_once()
    mdl._step_counts.assert_called_once()
    assert mdl._step_counts.call_args_list[0][0][0] == [1, 2]
    assert mdl._step_counts.call_args_list[0][0][1].to_list() == [1, 2]
    assert mdl._step_counts.call_args_list[0][0][2] == 5
    assert mdl._step_counts.call_args_list[0][0][3].to_dict() == {"a": 1, "b": 2}
    mdl._losr_all.assert_called_once()
    mdl._losr_aec.assert_called_once()
    mdl._losr_preop.assert_called_once()
    mdl._losr_bads.assert_called_once()


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
        "size": [4, 2],
    }


def test_bed_occupancy(mock_model):
    """test that it aggregates the bed occupancy data"""
    # arrange
    mdl = mock_model
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
    actual = mdl._bed_occupancy(
        "model_results", {"day+night": {"A": 0.75, "B": 0.875, "C": 0.5}}, 0
    )
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
    mdl.params["bed_occupancy"] = {
        "specialty_mapping": {
            "General and Acute": {"100": "a", "200": "b", "300": "b", "500": "_"},
            "Maternity": {"501": "c"},
        }
    }
    mdl._ward_groups = pd.concat(
        [
            pd.DataFrame(
                {"ward_type": k, "ward_group": list(v.values())},
                index=list(v.keys()),
            )
            for k, v in mdl.params["bed_occupancy"]["specialty_mapping"].items()
        ]
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
    mdl.data = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "2", "4", "1", "5"],
            "mainspef": ["100", "200", "300", "100", "200", "501", "501"],
            "speldur": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    model_results = pd.DataFrame(
        {
            "classpat": ["1", "1", "1", "1", "1", "2", "4"] * 3 + ["1", "5"],
            "mainspef": ["100", "100", "200", "200", "300", "100", "200"] * 3
            + ["501"] * 2,
            "pod": [
                x
                for x in [
                    "ip_non-elective_admission",
                    "ip_elective_admission",
                    "ip_daycase",
                ]
                for _ in range(7)
            ]
            + ["ip_non-elective_admission", "ip_non-elective_birth-episode"],
            "measure": ["beddays"] * 23,
            "value": [1, 2, 3, 4, 5, 6, 7] * 3 + [8, 9],
        }
    )
    # act
    actual = mock_model._bed_occupancy(
        model_results, {"day+night": {"a": 0.75, "b": 0.875, "c": 0.5}}, -1
    )
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
    model_results = pd.DataFrame(
        [
            {"tretspef": t, "admigroup": a, "measure": p}
            for t in ["100", "110", "200", "Other"]
            for a in ["elective", "non-elective"]
            for p in ["x", "procedures"]
        ]
    )
    model_results["value"] = list(range(len(model_results)))
    mock_model._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other": 300}, name="four_hour_sessions"
        ),
    }
    mock_model._theatres_baseline = pd.DataFrame(
        {
            "tretspef": ["100", "110", "200", "Other"],
            "is_elective": [1, 2, 3, 4],
            "n": [2, 4, 6, 8],
        },
    )
    # act
    theatres_available = mock_model._theatres_available(
        model_results,
        {
            "change_utilisation": {"100": 2, "110": 2.5, "Other": 3},
            "change_availability": 5,
        },
        0,
    )
    # assert
    assert {tuple(k): v for k, v in theatres_available.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 50,
        ("ip_theatres", "four_hour_sessions", "110"): 200,
        ("ip_theatres", "four_hour_sessions", "Other"): 314.2857142857143,
        ("ip_theatres", "theatres"): 2.276190476190476,
    }


def test_theatres_available_baseline(mock_model):
    """test that it returns the baseline theatres data"""
    # arrange
    mock_model.data = pd.DataFrame(
        [
            {"tretspef": t, "admimeth": a, "has_procedure": p}
            for t in ["100", "110", "200", "Other"]
            for a in ["11", "21"]
            for p in [1, 0]
        ]
    )
    mock_model._theatres_data = {
        "theatres": 10,
        "four_hour_sessions": pd.Series(
            {"100": 100, "110": 200, "Other": 300}, name="four_hour_sessions"
        ),
    }
    model_results = pd.concat([mock_model.data] * 3)
    # act
    theatres_available = mock_model._theatres_available(
        model_results,
        {
            "change_utilisation": {"100": 2, "110": 2.5, "Other": 3},
            "change_availability": 5,
        },
        -1,
    )
    # assert
    assert {tuple(k): v for k, v in theatres_available.items()} == {
        ("ip_theatres", "four_hour_sessions", "100"): 100,
        ("ip_theatres", "four_hour_sessions", "110"): 200,
        ("ip_theatres", "four_hour_sessions", "Other"): 300,
        ("ip_theatres", "theatres"): 10,
    }


def test_aggregate(mock_model):
    """test that it aggregates the results correctly"""

    def create_agg_stub(model_results, cols=None):
        name = "+".join(cols) if cols else "default"
        return {name: 1}

    mdl = mock_model
    mdl._create_agg = Mock(wraps=create_agg_stub)
    mdl._get_run_params = Mock(
        return_value={"bed_occupancy": "run_params", "theatres": "theatres"}
    )
    mdl._bed_occupancy = Mock(return_value=2)
    mdl._theatres_available = Mock(return_value=3)
    xs = list(range(6)) * 2
    model_results = pd.DataFrame(
        {
            "age_group": xs * 2,
            "sex": xs * 2,
            "admimeth": xs * 2,
            "admigroup": ["elective", "non-elective", "maternity"] * 8,
            "classpat": ["1", "2", "3", "4", "5", "-1"] * 4,
            "mainspef": xs * 2,
            "tretspef": xs * 2,
            "rn": [1] * 24,
            "has_procedure": [0, 1] * 12,
            "speldur": list(range(12)) * 2,
            "bedday_rows": [False] * 12 + [True] * 12,
        }
    )
    results = mdl.aggregate(model_results, 1)
    assert mdl._create_agg.call_count == 3
    assert results == {
        "default": 1,
        "sex+age_group": 1,
        "sex+tretspef": 1,
        "bed_occupancy": 2,
        "theatres_available": 3,
    }
    #

    mdl._get_run_params.assert_called_once_with(1)
    mdl._bed_occupancy.assert_called_once()
    mdl._theatres_available.assert_called_once()
    assert mdl._bed_occupancy.call_args_list[0][0][1] == "run_params"


def test_save_results(mocker, mock_model):
    """test that it correctly saves the results"""
    path_fn = lambda x: x

    to_parquet_mock = mocker.patch("pandas.DataFrame.to_parquet")
    results = pd.DataFrame(
        {
            "rn": [0],
            "classpat": [1],
            "age": [2],
            "sex": [3],
            "tretspef": [4],
            "speldur": [5],
        }
    )
    mock_model.save_results(results, path_fn)
    assert to_parquet_mock.call_count == 2
    assert to_parquet_mock.call_args_list[0] == call("op_conversion/0.parquet")
    assert to_parquet_mock.call_args_list[1] == call("ip/0.parquet")
