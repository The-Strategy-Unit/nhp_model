"""Test health status adjustmente."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nhp.model.health_status_adjustment import (
    HealthStatusAdjustment,
    HealthStatusAdjustmentGAM,
    HealthStatusAdjustmentInterpolated,
)

# Health Status Adjustment


@pytest.fixture
def mock_hsa():
    """Create a mock Model instance."""
    with patch.object(HealthStatusAdjustment, "__init__", lambda *args: None):
        hsa = HealthStatusAdjustment(None, None)  # type: ignore

    hsa._all_ages = np.arange(0, 101)
    hsa._cache = dict()

    return hsa


@pytest.mark.unit
def test_hsa_init(mocker):
    # arrange
    ages = np.arange(55, 91)
    laa_mock = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustment._load_activity_ages",
    )
    gha_m = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustment._generate_hsa_adjusted_ages",
    )
    gha_m.side_effect = [
        {1: np.tile(ages, (2, 1)), 2: np.tile(ages, (2, 1))},
        {1: np.tile(ages, (5, 1)), 2: np.tile(ages, (5, 1))},
    ]

    # act
    hsa = HealthStatusAdjustment("nhp_data", 2020, 2022, 42, 3)  # type: ignore

    # assert
    assert hsa._all_ages.tolist() == list(range(0, 101))
    assert hsa._cache == {}
    assert hsa._start_year == 2020
    assert hsa._end_year == 2022
    assert hsa._model_runs == 3
    assert hsa._ages.tolist() == np.arange(55, 91).tolist()

    laa_mock.assert_called_once_with("nhp_data")


@pytest.mark.unit
def test_load_activity_ages(mock_hsa):
    # arrange
    data_loader = Mock()
    data_loader.get_hsa_activity_table.return_value = pd.DataFrame(
        {
            "hsagrp": ["a"] * 4 + ["b"] * 4,
            "sex": [1, 2] * 4,
            "age": [1, 2, 3, 4] * 2,
            "activity": list(range(8)),
        }
    )

    # act
    mock_hsa._load_activity_ages(data_loader)

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


@pytest.mark.unit
def test_generate_params(mocker, mock_hsa):
    # arrange
    ex_dat = pd.Series(
        {
            (2020, 1, 65): 20.0,
            (2021, 1, 65): 21.0,
            (2020, 2, 65): 22.0,
            (2021, 2, 65): 23.5,
        }
    )
    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.life_expectancy.return_value = ex_dat

    dist_mock_male = Mock()
    dist_mock_female = Mock()
    dist_mock_male.rvs.return_value = np.array([50.0, 60.0])
    dist_mock_female.rvs.return_value = np.array([55.0, 65.0])
    ref_mock.hsa_metalog_parameters.return_value = {1: dist_mock_male, 2: dist_mock_female}

    mock_hsa._rv_params = Mock()

    # act
    result = mock_hsa._generate_params(2020, 2021)

    # assert
    # sex=1 (male): DFLE=10.45, target_ex=21.0, base_ex=20.0, denominator=1.0
    # numerator = [50/100*21 - 10.45, 60/100*21 - 10.45] = [0.05, 2.15]
    np.testing.assert_almost_equal(result[1], np.array([0.05, 2.15]))
    # sex=2 (female): DFLE=10.66, target_ex=23.5, base_ex=22.0, denominator=1.5
    # numerator = [55/100*23.5 - 10.66, 65/100*23.5 - 10.66] = [2.265, 4.615]
    np.testing.assert_almost_equal(result[2], np.array([2.265 / 1.5, 4.615 / 1.5]))

    ref_mock.life_expectancy.assert_called_once_with(2020, 2021)
    ref_mock.hsa_metalog_parameters.assert_called_once_with(2021)
    dist_mock_male.rvs.assert_called_once_with(mock_hsa._rv_params)
    dist_mock_female.rvs.assert_called_once_with(mock_hsa._rv_params)


@pytest.mark.unit
def test_generate_hsa_adjusted_ages(mocker, mock_hsa):
    # arrange
    ages = np.arange(55, 91)
    mock_hsa._ages = ages
    mock_hsa._start_year = 2020

    # ex_dat indexed by (year, sex, age)
    ex_dat = pd.Series(
        {
            (yr, sx, ag): float(ag + (yr - 2020) + sx)
            for yr in [2020, 2022]
            for sx in [1, 2]
            for ag in ages
        }
    )
    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.life_expectancy.return_value = ex_dat

    mock_hsa._generate_params = Mock(return_value={1: np.array([0.5]), 2: np.array([0.0])})

    # act
    result = mock_hsa._generate_hsa_adjusted_ages(2020, 2022)

    # assert
    ref_mock.life_expectancy.assert_called_once_with(2020, 2022)

    # For sex=1: ex_diff = ex(2022,1,age) - ex(2020,1,age) = 2 for all ages
    # adjusted = ages - 0.5 * 2 = ages - 1
    for i, ag in enumerate(ages):
        assert result[1][0][i] == pytest.approx(float(ag) - 1)
    # For sex=2: hsa_params=0.0 so adjusted = ages - 0 = ages
    for i, ag in enumerate(ages):
        assert result[2][0][i] == pytest.approx(float(ag))


@pytest.mark.unit
def test_hsa_run_not_cached(mocker, mock_hsa):
    # arrange
    mock_hsa._ages = [1, 2]
    activity = pd.Series(
        {
            ("a", 1, 1): 2,  # 2 / 2 = 1
            ("a", 1, 2): 6,  # 6 / 3 = 2
            ("a", 2, 1): 12,  # 12 / 4 = 3
            ("a", 2, 2): -20,  # clipped to 0
            ("b", 1, 1): 30,  # 30 / 6 = 5
            ("b", 1, 2): 42,  # 42 / 7 = 6
            ("b", 2, 1): 56,  # 56 / 8 = 7
            ("b", 2, 2): 72,  # 72 / 9 = 8
        }
    )
    mock_hsa._activity_ages = pd.Series(
        {
            ("a", 1, 0): 0,
            ("a", 1, 1): 2,
            ("a", 1, 2): 3,
            ("a", 1, 3): 0,
            ("a", 2, 0): 0,
            ("a", 2, 1): 4,
            ("a", 2, 2): 5,
            ("a", 3, 3): 0,
            ("b", 1, 0): 0,
            ("b", 1, 1): 6,
            ("b", 1, 2): 7,
            ("b", 1, 3): 0,
            ("b", 2, 0): 0,
            ("b", 2, 1): 8,
            ("b", 2, 2): 9,
            ("b", 2, 3): 0,
        }
    ).rename_axis(["hsagrp", "sex", "age"])
    mock_hsa._predict_activity = Mock(return_value=activity)
    mock_hsa.hsa_params = {
        1: np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
        2: np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
    }

    # act
    actual = mock_hsa.run({"model_run": 5})

    # assert
    assert actual.to_dict() == {
        ("a", 1, 1): 1.0,
        ("a", 1, 2): 2.0,
        ("a", 2, 1): 3.0,
        ("a", 2, 2): 0.0,
        ("b", 1, 1): 5.0,
        ("b", 1, 2): 6.0,
        ("b", 2, 1): 7.0,
        ("b", 2, 2): 8.0,
    }
    mock_hsa._predict_activity.assert_called_once()
    assert mock_hsa._cache[4].equals(actual)


@pytest.mark.unit
def test_hsa_run_cached(mocker, mock_hsa):
    # arrange
    mock_hsa._cache[2] = "a"

    # act
    actual = mock_hsa.run({"model_run": 3})

    # assert
    assert actual == "a"


@pytest.mark.unit
def test_hsa_predict_activity(mock_hsa):
    # arrange
    # act & assert
    with pytest.raises(NotImplementedError):
        mock_hsa._predict_activity(None)


# GAM


@pytest.fixture
def mock_hsa_gam():
    """Create a mock Model instance."""
    with patch.object(HealthStatusAdjustmentGAM, "__init__", lambda *args: None):
        hsa = HealthStatusAdjustmentGAM(None, None)  # type: ignore

    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    hsa._gams = {(h, s): hsa_mock for h in ["a", "b"] for s in [1, 2]}

    hsa._all_ages = np.arange(0, 3)
    hsa._ages = np.array([1, 2])

    return hsa


@pytest.mark.unit
def test_hsa_gam_init(mocker):
    # arrange
    super_mock = mocker.patch("nhp.model.health_status_adjustment.HealthStatusAdjustment.__init__")
    nhp_data_mock = Mock()
    nhp_data_mock.get_hsa_gams.return_value = "hsa_gams"

    # act
    hsa = HealthStatusAdjustmentGAM(nhp_data_mock, 2020, 2022, 42, 3)

    # assert
    assert hsa._gams == "hsa_gams"
    nhp_data_mock.get_hsa_gams.assert_called_once()
    super_mock.assert_called_once_with(nhp_data_mock, 2020, 2022, 42, 3)


@pytest.mark.unit
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
    """Create a mock Model instance."""
    with patch.object(HealthStatusAdjustmentInterpolated, "__init__", lambda *args: None):
        hsa = HealthStatusAdjustmentInterpolated(None, None)  # type: ignore

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

    hsa._all_ages = np.array([0, 1, 2, 3])
    hsa._ages = np.array([1, 2])

    return hsa


@pytest.mark.unit
def test_hsa_interpolated_init(mocker):
    # arrange
    super_mock = mocker.patch("nhp.model.health_status_adjustment.HealthStatusAdjustment.__init__")
    lal_mock = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustmentInterpolated._load_activity_ages_lists"
    )

    # act
    HealthStatusAdjustmentInterpolated("data/synthetic", 2020, 2022, 42, 3)  # type: ignore

    # assert
    super_mock.assert_called_once_with("data/synthetic", 2020, 2022, 42, 3)
    lal_mock.assert_called_once_with()


@pytest.mark.unit
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


@pytest.mark.unit
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
