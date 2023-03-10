"""test health status adjustmente"""
# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.health_status_adjustment import (HealthStatusAdjustment,
                                            HealthStatusAdjustmentGAM,
                                            HealthStatusAdjustmentInterpolated)

# Health Status Adjustment


@pytest.fixture
def mock_hsa():
    """create a mock Model instance"""
    with patch.object(HealthStatusAdjustment, "__init__", lambda s, d, l: None):
        hsa = HealthStatusAdjustment(None, None)

    hsa._all_ages = np.arange(0, 101)
    hsa._activity_table_path = "data/synthetic/hsa_activity_table.csv"
    hsa._cache = dict()

    return hsa


def test_hsa_init(mocker):
    # arrange
    mocker.patch("pickle.load", return_value="pkl_load")
    lle_mock = mocker.patch(
        "model.health_status_adjustment.HealthStatusAdjustment._load_life_expectancy_series",
    )
    laa_mock = mocker.patch(
        "model.health_status_adjustment.HealthStatusAdjustment._load_activity_ages",
    )

    # act
    hsa = HealthStatusAdjustment("data/synthetic", "life_expectancy")

    # assert
    assert hsa._all_ages.tolist() == list(range(0, 101))
    assert hsa._activity_table_path == "data/synthetic/hsa_activity_table.csv"
    assert hsa._cache == dict()

    lle_mock.assert_called_once_with("life_expectancy")
    laa_mock.assert_called_once_with()


def test_hsa_load_life_expectancy_series(mock_hsa):
    # arrange
    life_expectancy = {
        "m": [1, 2, 3],
        "f": [4, 5, 6],
        "min_age": 50,
        "max_age": 52,
    }

    # act
    mock_hsa._load_life_expectancy_series(life_expectancy)

    # assert
    assert mock_hsa._ages.tolist() == [50, 51, 52]
    assert mock_hsa._life_expectancy.to_dict() == {
        (1, 50): 1,
        (1, 51): 2,
        (1, 52): 3,
        (2, 50): 4,
        (2, 51): 5,
        (2, 52): 6,
    }


def test_load_activity_ages(mocker, mock_hsa):
    # arrange
    df = pd.DataFrame(
        {
            "hsagrp": ["a"] * 4 + ["b"] * 4,
            "sex": [1, 2] * 4,
            "age": [1, 2, 3, 4] * 2,
            "activity": list(range(8)),
        }
    )
    mocker.patch("pandas.read_csv", return_value=df)

    # act
    mock_hsa._load_activity_ages()

    # assert
    assert mock_hsa._activity_ages.to_dict() == {
        ("a", 1, 1): 0,
        ("a", 2, 2): 1,
        ("a", 1, 3): 2,
        ("a", 2, 4): 3,
        ("b", 1, 1): 4,
        ("b", 2, 2): 5,
        ("b", 1, 3): 6,
        ("b", 2, 4): 7,
    }


def test_hsa_run_not_cached(mock_hsa):
    # arrange
    mock_hsa._ages = [1, 2]
    mock_hsa._life_expectancy = pd.Series(
        {
            ("a", 1, 1): 0,
            ("a", 1, 2): 1,
            ("a", 2, 1): 2,
            ("a", 2, 2): 3,
            ("b", 1, 1): 4,
            ("b", 1, 2): 5,
            ("b", 2, 1): 6,
            ("b", 2, 2): 7,
        }
    ).rename_axis(["hsagrp", "sex", "age"])
    activity = pd.Series(
        {
            ("a", 1, 1): 2,  # 2 / 2 = 1
            ("a", 1, 2): 6,  # 6 / 3 = 2
            ("a", 2, 1): 12,  # 12 / 4 = 3
            ("a", 2, 2): 20,  # 20 / 5 = 4
            ("b", 1, 1): 30,  # 30 / 6 = 5
            ("b", 1, 2): 42,  # 42 / 7 = 6
            ("b", 2, 1): 56,  # 56 / 8 = 7
            ("b", 2, 2): 72,  # 72 / 9 = 8
        }
    )
    mock_hsa._activity_ages = pd.Series(
        {
            ("a", 1, 0): 0,  #
            ("a", 1, 1): 2,
            ("a", 1, 2): 3,
            ("a", 1, 3): 0,  #
            ("a", 2, 0): 0,  #
            ("a", 2, 1): 4,
            ("a", 2, 2): 5,
            ("a", 3, 3): 0,  #
            ("b", 1, 0): 0,  #
            ("b", 1, 1): 6,
            ("b", 1, 2): 7,
            ("b", 1, 3): 0,  #
            ("b", 2, 0): 0,  #
            ("b", 2, 1): 8,
            ("b", 2, 2): 9,
            ("b", 2, 3): 0,  #
        }
    ).rename_axis(["hsagrp", "sex", "age"])
    mock_hsa._predict_activity = Mock(return_value=activity)

    # act
    actual = mock_hsa.run(2)

    # assert
    assert actual.to_dict() == {
        ("a", 1, 1): 1.0,
        ("a", 1, 2): 2.0,
        ("a", 2, 1): 3.0,
        ("a", 2, 2): 4.0,
        ("b", 1, 1): 5.0,
        ("b", 1, 2): 6.0,
        ("b", 2, 1): 7.0,
        ("b", 2, 2): 8.0,
    }
    assert mock_hsa._cache[2].equals(actual)


def test_hsa_run_cached(mock_hsa):
    # arrange
    mock_hsa._cache[1] = "a"

    # act
    actual = mock_hsa.run(1)

    # assert
    assert actual == "a"


def test_hsa_predict_activity():
    pass


# GAM


@pytest.fixture
def mock_hsa_gam():
    """create a mock Model instance"""
    with patch.object(HealthStatusAdjustmentGAM, "__init__", lambda s, d, l: None):
        hsa = HealthStatusAdjustmentGAM(None, None)

    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    hsa._gams = {(h, s): hsa_mock for h in ["a", "b"] for s in [1, 2]}

    hsa._all_ages = np.arange(0, 3)
    hsa._ages = [1, 2]

    return hsa


def test_hsa_gam_init(mocker):
    # arrange
    mocker.patch("pickle.load", return_value="pkl_load")
    super_mock = mocker.patch(
        "model.health_status_adjustment.HealthStatusAdjustment.__init__"
    )

    # act
    with patch("builtins.open", mock_open(read_data="hsa_gams")) as mock_file:
        hsa = HealthStatusAdjustmentGAM("data/synthetic", "life_expectancy")

    # assert
    assert hsa._gams == "pkl_load"
    mock_file.assert_called_with("data/synthetic/hsa_gams.pkl", "rb")
    super_mock.assert_called_once_with("data/synthetic", "life_expectancy")


def test_hsa_gam_predict_activity(mock_hsa_gam):
    # arrange
    adjusted_ages = pd.Series({1: [0.5, 1.5], 2: [1.5, 2.5]})

    # act
    actual = mock_hsa_gam._predict_activity(adjusted_ages)

    # assert
    assert actual.to_dict() == {
        ("a", 1, 1): 0.5,
        ("a", 1, 2): 1.5,
        ("a", 2, 1): 1.5,
        ("a", 2, 2): 2.5,
        ("b", 1, 1): 0.5,
        ("b", 1, 2): 1.5,
        ("b", 2, 1): 1.5,
        ("b", 2, 2): 2.5,
    }


# Interpolated


@pytest.fixture
def mock_hsa_interpolated():
    """create a mock Model instance"""
    with patch.object(
        HealthStatusAdjustmentInterpolated, "__init__", lambda s, d, l: None
    ):
        hsa = HealthStatusAdjustmentInterpolated(None, None)

    hsa._activity_ages = pd.Series(
        {
            ("a", 1, 0): 0,
            ("a", 1, 1): 1,
            ("a", 1, 2): 2,
            ("a", 1, 3): 3,
            ("a", 2, 0): 4,
            ("a", 2, 1): 5,
            ("a", 2, 2): 6,
            ("a", 2, 3): 7,
            ("b", 1, 0): 8,
            ("b", 1, 1): 9,
            ("b", 1, 2): 10,
            ("b", 1, 3): 11,
            ("b", 2, 0): 12,
            ("b", 2, 1): 13,
            ("b", 2, 2): 14,
            ("b", 2, 3): 15,
        }
    )

    hsa._all_ages = [0, 1, 2, 3]
    hsa._ages = [1, 2]

    return hsa


def test_hsa_interpolated_init(mocker):
    # arrange
    super_mock = mocker.patch(
        "model.health_status_adjustment.HealthStatusAdjustment.__init__"
    )
    lal_mock = mocker.patch(
        # pylint: disable=line-too-long
        "model.health_status_adjustment.HealthStatusAdjustmentInterpolated._load_activity_ages_lists"
    )

    # act
    HealthStatusAdjustmentInterpolated("data/synthetic", "life_expectancy")

    # assert
    super_mock.assert_called_once_with("data/synthetic", "life_expectancy")
    lal_mock.assert_called_once_with()


def test_hsa_interpolated_load_activity_ages_lists(mock_hsa_interpolated):
    # arrange

    # act
    mock_hsa_interpolated._load_activity_ages_lists()

    # assert
    assert mock_hsa_interpolated._activity_ages_lists.to_dict() == {
        ("a", 1): [0, 1, 2, 3],
        ("a", 2): [4, 5, 6, 7],
        ("b", 1): [8, 9, 10, 11],
        ("b", 2): [12, 13, 14, 15],
    }


def test_hsa_interpolated_predict_activity(mock_hsa_interpolated):
    # arrange
    adjusted_ages = pd.Series({1: [0.5, 1.5], 2: [1.5, 2.5]})
    mock_hsa_interpolated._activity_ages_lists = pd.Series(
        {
            ("a", 1): [0, 1, 2, 3],
            ("a", 2): [4, 5, 6, 7],
            ("b", 1): [8, 9, 10, 11],
            ("b", 2): [12, 13, 14, 15],
        }
    )

    # act
    actual = mock_hsa_interpolated._predict_activity(adjusted_ages)

    # assert
    assert actual.to_dict() == {
        ("a", 1, 1): 0.5,
        ("a", 1, 2): 1.5,
        ("a", 2, 1): 5.5,
        ("a", 2, 2): 6.5,
        ("b", 1, 1): 8.5,
        ("b", 1, 2): 9.5,
        ("b", 2, 1): 13.5,
        ("b", 2, 2): 14.5,
    }
