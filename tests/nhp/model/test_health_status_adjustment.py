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


def test_hsa_init(mocker):
    # arrange
    mocker.patch("pickle.load", return_value="pkl_load")
    lle_mock = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustment._load_life_expectancy_series",
    )
    laa_mock = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustment._load_activity_ages",
    )

    # act
    hsa = HealthStatusAdjustment("nhp_data", 2020)  # type: ignore

    # assert
    assert hsa._all_ages.tolist() == list(range(0, 101))
    assert hsa._cache == {}

    lle_mock.assert_called_once_with(2020)
    laa_mock.assert_called_once_with("nhp_data")


@pytest.mark.parametrize(
    "year, expectation",
    [
        (
            2020,
            {
                "2020": {("a", 1, 55): 0, ("b", 2, 55): 0},
                "2021": {("a", 1, 55): 1, ("b", 2, 55): 5},
                "2022": {("a", 1, 55): 3, ("b", 2, 55): 13},
            },
        ),
        (
            2021,
            {
                "2020": {("a", 1, 55): -1, ("b", 2, 55): -5},
                "2021": {("a", 1, 55): 0, ("b", 2, 55): 0},
                "2022": {("a", 1, 55): 2, ("b", 2, 55): 8},
            },
        ),
    ],
)
def test_hsa_load_life_expectancy_series(mocker, mock_hsa, year, expectation):
    # arrange
    life_expectancy = pd.DataFrame(
        {
            "var": ["a", "b"],
            "sex": [1, 2],
            "age": [55, 55],
            "2020": [2, 8],
            "2021": [3, 13],
            "2022": [5, 21],
        }
    )
    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.life_expectancy.return_value = life_expectancy

    # act
    mock_hsa._load_life_expectancy_series(year)

    # assert
    assert mock_hsa._ages.tolist() == np.arange(55, 91).tolist()
    assert mock_hsa._life_expectancy.to_dict() == expectation
    ref_mock.life_expectancy.assert_called_once_with()


def test_hsa_load_life_expectancy_series_filters_ages(mocker, mock_hsa):
    # arrange
    life_expectancy = pd.DataFrame(
        {
            "var": ["a"] * 100,
            "sex": [1] * 100,
            "age": list(range(100)),
            "2020": list(range(100)),
            "2022": [i * 2 for i in range(100)],
        }
    )
    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.life_expectancy.return_value = life_expectancy

    # act
    mock_hsa._load_life_expectancy_series(2020)

    # assert
    assert list(mock_hsa._life_expectancy.to_dict()["2020"].keys()) == [
        ("a", 1, i) for i in range(55, 91)
    ]


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


@pytest.mark.parametrize(
    "year, expected_results, expected_call",
    [
        (2021, [2] + list(range(1, 11)), (2, 5, 8)),
        (2022, [3] + list(range(1, 11)) + [2], (3, 6, 9)),
    ],
)
def test_generate_params(mocker, year, expected_results, expected_call):
    # arrange
    m = mocker.patch("nhp.model.health_status_adjustment.HealthStatusAdjustment.random_splitnorm")
    m.return_value = list(range(1, 11))

    split_normal_params = pd.DataFrame(
        {
            "var": ["ppp"] * 8,
            "sex": ["m"] * 4 + ["f"] * 4,
            "year": [2019, 2020, 2021, 2022] * 2,
            "mode": [0, 1, 2, 3] * 2,
            "sd1": [0, 4, 5, 6] * 2,
            "sd2": [0, 7, 8, 9] * 2,
        }
    )
    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.split_normal_params.return_value = split_normal_params
    ref_mock.variant_lookup.return_value = {"principal_proj": "ppp"}

    # act
    actual = HealthStatusAdjustment.generate_params(
        2020,
        year,
        ["principal_proj"] * 11,
        "rng",  # type: ignore
        10,
    )

    # assert
    assert [i[0] for i in actual] == expected_results
    assert [i[1] for i in actual] == expected_results
    m.assert_called_with("rng", 10, *expected_call)
    assert m.call_count == 2

    ref_mock.split_normal_params.assert_called_once_with()
    ref_mock.variant_lookup.assert_called_once_with()


def test_random_splitnorm():
    # arrange
    rng = Mock()
    rng.uniform.return_value = np.arange(0.1, 1.0, 0.1)
    # checking against R:
    # > fanplot::qsplitnorm(seq(0.1, 0.9, 0.1), 3, sd1 = 1, sd2 = 2)
    expected = [
        1.963567,
        2.475599,
        2.874339,
        3.251323,
        3.637279,
        4.048801,
        4.510830,
        5.072867,
        5.879063,
    ]

    # act
    actual = HealthStatusAdjustment.random_splitnorm(rng, 9, 3, 1, 2)

    # assert
    rng.uniform.assert_called_once_with(size=9)
    np.testing.assert_almost_equal(actual, expected, 6)


def test_hsa_run_not_cached(mocker, mock_hsa):
    # arrange
    mock_hsa._ages = [1, 2]
    mock_hsa._life_expectancy = pd.DataFrame(
        {
            "2020": {
                ("ppp", 1, 1): 0,
                ("ppp", 1, 2): 1,
                ("ppp", 2, 1): 2,
                ("ppp", 2, 2): 3,
                ("hle", 1, 1): 4,
                ("hle", 1, 2): 5,
                ("hle", 2, 1): 6,
                ("hle", 2, 2): 7,
            }
        }
    ).rename_axis(["var", "sex", "age"])
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

    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.variant_lookup.return_value = {"principal_proj": "ppp"}

    # act
    actual = mock_hsa.run(
        {"year": 2020, "health_status_adjustment": [2, 3], "variant": "principal_proj"}
    )

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
    assert mock_hsa._cache[(2, 3, "ppp")].equals(actual)


def test_hsa_run_cached(mocker, mock_hsa):
    # arrange
    mock_hsa._cache[(1, 2, "ppp")] = "a"

    ref_mock = mocker.patch("nhp.model.health_status_adjustment.reference")
    ref_mock.variant_lookup.return_value = {"principal_proj": "ppp"}

    # act
    actual = mock_hsa.run(
        {"year": 2020, "health_status_adjustment": [1, 2], "variant": "principal_proj"}
    )

    # assert
    assert actual == "a"


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
    hsa._ages = [1, 2]  # type: ignore

    return hsa


def test_hsa_gam_init(mocker):
    # arrange
    super_mock = mocker.patch("nhp.model.health_status_adjustment.HealthStatusAdjustment.__init__")
    nhp_data_mock = Mock()
    nhp_data_mock.get_hsa_gams.return_value = "hsa_gams"

    # act
    hsa = HealthStatusAdjustmentGAM(nhp_data_mock, 2020)  # type: ignore

    # assert
    assert hsa._gams == "hsa_gams"
    nhp_data_mock.get_hsa_gams.assert_called_once()
    super_mock.assert_called_once_with(nhp_data_mock, 2020)


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

    hsa._all_ages = [0, 1, 2, 3]  # type: ignore
    hsa._ages = [1, 2]  # type: ignore

    return hsa


def test_hsa_interpolated_init(mocker):
    # arrange
    super_mock = mocker.patch("nhp.model.health_status_adjustment.HealthStatusAdjustment.__init__")
    lal_mock = mocker.patch(
        "nhp.model.health_status_adjustment.HealthStatusAdjustmentInterpolated._load_activity_ages_lists"
    )

    # act
    HealthStatusAdjustmentInterpolated("data/synthetic", 2020)  # type: ignore

    # assert
    super_mock.assert_called_once_with("data/synthetic", 2020)
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
