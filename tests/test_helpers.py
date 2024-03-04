"""test helper methods"""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.helpers import age_groups, create_time_profiles, inrange, load_params, rnorm


@pytest.mark.parametrize(
    "value, expected", [(-1.1, 0), (1.1, 1), (0, 0), (1, 1), (0.5, 0.5)]
)
def test_inrange(value, expected):
    """test that the inrange function returns expected values"""
    assert inrange(value) == expected


@pytest.mark.parametrize(
    "value, low, high, expected", [(0, 0.25, 0.75, 0.25), (1, 0.25, 0.75, 0.75)]
)
def test_inrange_lo_hi(value, low, high, expected):
    """test that the inrange function returns expected values"""
    assert inrange(value, low, high) == expected


def test_rnorm():
    """test that the rnorm function returns random values"""
    rng = Mock()
    rng.normal.return_value = 1.5
    assert rnorm(rng, 1, 2) == 1.5
    rng.normal.assert_called_once_with(1.5, 0.3901520929904105)


def test_age_groups():
    """test that the age_groups function returns expected values"""
    ages = pd.Series([np.nan] + list(range(0, 90)))
    expected = list(
        np.concatenate(
            [
                ["Unknown"],
                [" 0- 4"] * 5,
                [" 5-14"] * 10,
                ["15-34"] * 20,
                ["35-49"] * 15,
                ["50-64"] * 15,
                ["65-84"] * 20,
                ["85+"] * 5,
            ]
        ).flat
    )
    assert np.array_equal(age_groups(ages), expected)


def test_load_params():
    """test that load_params opens the params file"""
    with patch("builtins.open", mock_open(read_data='{"params": 0}')) as mock_file:
        assert load_params("filename.json") == {"params": 0}
        mock_file.assert_called_with("filename.json", "r", encoding="UTF-8")


def test_create_time_profiles():
    # act
    time_profiles = create_time_profiles(5, 1)

    # assert
    assert time_profiles["none"] == 1
    assert time_profiles["linear"] == 0.2
    assert time_profiles["front_loaded"] == 0.6
    assert time_profiles["back_loaded"] == 1 - 24**0.5 / 5
    assert time_profiles["step"](0) == 1
    assert time_profiles["step"](1) == 1
    assert time_profiles["step"](2) == 0
