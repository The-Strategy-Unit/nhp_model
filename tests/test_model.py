"""test model"""

import re
from unittest.mock import Mock, mock_open, patch

import pytest
from model.model import Model


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


def test_model_init_validates_model_type():
    """it raises an exception if an invalid model_type is passed"""
    with pytest.raises(AssertionError):
        Model("", None, None)


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
