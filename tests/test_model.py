"""test model"""
# pylint: disable=protected-access,redefined-outer-name,no-member

import re
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
from model.model import Model


# fixtures
@pytest.fixture
def mock_model():
    """create a mock Model instance"""
    with patch.object(Model, "__init__", lambda s, m, p, d, c: None):
        mdl = Model(None, None, None, None)
    mdl.model_type = "aae"
    mdl.params = {
        "input_data": "synthetic",
        "model_runs": 3,
        "demographic_factors": {
            "file": "demographics_file.csv",
            "variant_probabilities": {"a": 0.6, "b": 0.4},
        },
        "start_year": 2018,
        "end_year": 2020,
    }
    mdl._data_path = "data/synthetic"
    mdl._hsa_gams = "gams"
    mdl.data = pd.DataFrame(
        {
            "rn": list(range(1, 21)),
            "age": list(range(1, 6)) * 4,
            "sex": ([1] * 5 + [2] * 5) * 2,
        }
    )
    return mdl


# __init__()


@pytest.mark.init
@pytest.mark.parametrize("model_type", ["aae", "ip", "op"])
def test_model_init_sets_values(mocker, model_type):
    """test model constructor works as expected"""
    params = {"input_data": "synthetic", "create_datetime": "20220101_012345"}

    mocker.patch("model.model.Model._load_parquet", return_value={"age": 1})
    mocker.patch("model.model.Model._load_demog_factors", return_value=None)
    mocker.patch("model.model.Model._generate_run_params", return_value=None)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("pickle.load", return_value="load_hsa")
    mocker.patch("os.path.join", lambda *args: "/".join(args))

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        mdl = Model(model_type, params, "data")
        assert mdl.model_type == model_type
        assert mdl.params == params
        assert mdl._data_path == "data/synthetic"
        mock_file.assert_called_with("data/synthetic/hsa_gams.pkl", "rb")
        assert mdl._hsa_gams == "load_hsa"
        assert mdl.data == {"age": 1, "age_group": "age_groups"}
        mdl._load_demog_factors.assert_called_once()
        mdl._generate_run_params.assert_called_once()


@pytest.mark.init
def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None)


@pytest.mark.init
def test_model_init_sets_create_datetime(mocker):
    """it sets the create_datetime item in params if not already set"""
    params = {"input_data": "synthetic"}

    mocker.patch("model.model.Model._load_parquet", return_value={"age": 1})
    mocker.patch("model.model.Model._load_demog_factors", return_value=None)
    mocker.patch("model.model.Model._generate_run_params", return_value=None)
    mocker.patch("model.model.age_groups", return_value="age_groups")
    mocker.patch("pickle.load", return_value="load_hsa")
    mocker.patch("os.path.join", lambda *args: "/".join(args))

    with patch("builtins.open", mock_open(read_data="data")):
        mdl = Model("aae", params, "data")
        assert re.match("^\\d{8}_\\d{6}$", mdl.params["create_datetime"])


# _load_demog_factors()


@pytest.mark.load_demog_factors
@pytest.mark.parametrize(
    "start_year, end_year", [("2018", "2019"), ("2018", "2020"), ("2019", "2020")]
)
def test_demog_factors_loads_correctly(mocker, mock_model, start_year, end_year):
    """test that the demographic factors are loaded correctly"""
    mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "variant": ["a"] * 10 + ["b"] * 10,
                "age": list(range(1, 6)) * 4,
                "sex": ([1] * 5 + [2] * 5) * 2,
                "2018": list(range(1, 21)),
                "2019": list(range(11, 31)),
                "2020": list(range(21, 41)),
            }
        ),
    )
    mdl = mock_model
    mdl.params["start_year"] = start_year
    mdl.params["end_year"] = end_year
    mdl._load_demog_factors()

    # choose the values for the ranges to test
    if start_year == "2018":
        a_range = range(1, 11)
        b_range = range(11, 21)
        diff = 10 if end_year == "2019" else 20
    else:
        a_range = range(11, 21)
        b_range = range(21, 31)
        diff = 10

    assert mdl._demog_factors.equals(
        pd.DataFrame(
            {
                "a": [(i + diff) / i for i in a_range for _ in [0, 1]],
                "b": [(i + diff) / i for i in b_range for _ in [0, 1]],
                "rn": [i + j for i in range(1, 11) for j in [0, 10]],
            }
        ).set_index("rn")
    )

    assert mdl._variants == ["a", "b"]
    assert mdl._probabilities == [0.6, 0.4]
