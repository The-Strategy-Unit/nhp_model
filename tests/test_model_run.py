"""test run_model"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from unittest.mock import Mock, patch

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
    mr._model = Mock()
    return mr


@pytest.mark.parametrize("run, rp_call", [(-1, 0), (0, 0), (1, 1)])
def test_init(mocker, run, rp_call):
    # arrange
    model = Mock()
    model.params = "params"
    model.data = "data"
    model._get_run_params.return_value = {"seed": 1}

    rng_mock = mocker.patch("numpy.random.default_rng", return_value="rng")
    prp_mock = mocker.patch("model.model_run.ModelRun._patch_run_params")

    # act
    actual = ModelRun(model, run)

    # assert
    assert actual.model == model
    assert actual.params == "params"
    assert actual.run_params == {"seed": 1}
    assert actual.rng == "rng"
    assert actual.data == "data"
    assert actual.step_counts == {}

    rng_mock.assert_called_once_with(1)
    prp_mock.assert_called_once_with()
    model._get_run_params.assert_called_once_with(rp_call)


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


def test_get_step_counts(mock_model_run):
    # arrange
    mr = mock_model_run
    mr.model = Mock()
    mr.step_counts = "step_counts"
    mr.model.get_step_counts_dataframe.return_value = "step_counts_df"

    # act
    actual = mr.get_step_counts()

    # assert
    assert actual == "step_counts_df"
    mr.model.get_step_counts_dataframe.assert_called_once_with("step_counts")


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
