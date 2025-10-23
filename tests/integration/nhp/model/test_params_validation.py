"""Test params-sample."""

import pytest

from nhp.model.params import load_sample_params


def test_sample_params_are_valid():
    load_sample_params(dataset="dev", scenario="unit-test")
    # assert: no exception raised


def test_load_sample_params_validation_fails():
    from jsonschema.exceptions import ValidationError

    with pytest.raises(ValidationError):
        load_sample_params(demographic_factors="invalid-factor")
