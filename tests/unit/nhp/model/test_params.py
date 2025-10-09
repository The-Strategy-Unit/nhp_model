"""Test params-sample."""

from unittest.mock import mock_open, patch

import jsonschema
import pytest

from nhp.model.params import load_params, load_sample_params, validate_params


def test_validate_params(mocker):
    # arrange
    m_validate = mocker.patch("jsonschema.validate")
    m_json_load = mocker.patch("json.load", return_value="schema")

    # act
    validate_params("params")  # ty: ignore[invalid-argument-type]

    # assert
    m_validate.assert_called_once_with(instance="params", schema="schema")
    assert m_json_load.call_args[0][0].name.endswith("params-schema.json")


def test_load_params(mocker):
    """Test that load_params opens the params file."""
    # arrange
    m_vp = mocker.patch("nhp.model.params.validate_params")

    # act
    with patch("builtins.open", mock_open(read_data='{"params": 0}')) as mock_file:
        assert load_params("filename.json") == {"params": 0}

        # assert
        mock_file.assert_called_with("filename.json", "r", encoding="UTF-8")
        m_vp.assert_called_once_with({"params": 0})


def test_load_sample_params():
    # arrange
    # act
    actual = load_sample_params(dataset="dev", scenario="unit-test")

    # assert
    assert actual["dataset"] == "dev"
    assert actual["scenario"] == "unit-test"


def test_load_sample_params_validation_fails():
    # arrange
    # act
    with pytest.raises(jsonschema.ValidationError):
        load_sample_params(demographic_factors="invalid-factor")
