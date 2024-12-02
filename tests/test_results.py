"""test results.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from model.results import (
    _combine_model_results,
    _combine_step_counts,
    _generate_results_json,
    combine_results,
)

results = [
    [
        (
            pd.DataFrame(
                {
                    "pod": ["a1", "a2"] * 4,
                    "measure": [i for i in ["a", "b"] for _ in [1, 2]] * 2,
                    "value": range(0, 8),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["a1", "a2", "a1", "a2"],
                    "change_factor": ["a", "a", "b", "b"],
                    "value": range(0, 4),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
        (
            pd.DataFrame(
                {
                    "pod": ["a1", "a2"] * 4,
                    "measure": [i for i in ["a", "b"] for _ in [1, 2]] * 2,
                    "value": range(8, 16),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["a1", "a2", "a1", "a2"],
                    "change_factor": ["a", "a", "b", "b"],
                    "value": range(4, 8),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
        (
            pd.DataFrame(
                {
                    "pod": ["a1"] * 4,
                    "measure": [i for i in ["a", "b"] for _ in [1, 2]],
                    "value": range(16, 20),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["a1", "a1"],
                    "change_factor": ["a", "b"],
                    "value": range(8, 10),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
    ],
    [
        (
            pd.DataFrame(
                {
                    "pod": ["b1", "b2"] * 4,
                    "measure": [i for i in ["c", "d"] for _ in [1, 2]] * 2,
                    "value": range(8),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["b1", "b2", "b1", "b2"],
                    "change_factor": ["a", "a", "b", "b"],
                    "value": range(0, 4),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
        (
            pd.DataFrame(
                {
                    "pod": ["b1", "b2"] * 4,
                    "measure": [i for i in ["c", "d"] for _ in [1, 2]] * 2,
                    "value": range(8, 16),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["b1", "b2", "b1", "b2"],
                    "change_factor": ["a", "a", "b", "b"],
                    "value": range(4, 8),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
        (
            pd.DataFrame(
                {
                    "pod": ["b1"] * 4,
                    "measure": [i for i in ["c", "d"] for _ in [1, 2]],
                    "value": range(16, 20),
                }
            ),
            pd.DataFrame(
                {
                    "pod": ["b1", "b1"],
                    "change_factor": ["a", "b"],
                    "value": range(8, 10),
                }
            ).set_index(["pod", "change_factor"])["value"],
        ),
    ],
]


def test_combine_model_results():
    # arrange

    expected = {
        "pod": ["a1"] * 6
        + ["a2"] * 5
        + ["a1"] * 6
        + ["a2"] * 5
        + ["b1"] * 6
        + ["b2"] * 5
        + ["b1"] * 6
        + ["b2"] * 5,
        "measure": ["a"] * 11 + ["b"] * 11 + ["c"] * 11 + ["d"] * 11,
        "value": [
            0,
            4,
            8,
            12,
            16,
            17,
            1,
            5,
            9,
            13,
            0,
            2,
            6,
            10,
            14,
            18,
            19,
            3,
            7,
            11,
            15,
            0,
        ]
        * 2,
        "model_run": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2] * 4,
    }

    # act
    actual = _combine_model_results(results)

    # assert
    assert actual.to_dict("list") == expected


def test_combine_step_counts():
    # arrange

    expected = {
        "pod": ["a1"] * 4 + ["a2"] * 4 + ["b1"] * 4 + ["b2"] * 4,
        "change_factor": [i for i in ["a", "b"] for _ in [1, 2]] * 4,
        "value": [4, 8, 6, 9, 5, 0, 7, 0, 4, 8, 6, 9, 5, 0, 7, 0],
        "model_run": [1, 2] * 8,
    }

    # act
    actual = _combine_step_counts(results)

    # assert
    assert actual.to_dict("list") == expected

def test_generate_results_json():
    # arrange

    # act

    # assert

def test_combine_results(mocker):
    # arrange
    ma = mocker.patch(
        "model.results._combine_model_results", return_value="combined_results"
    )
    mb = mocker.patch(
        "model.results._combine_step_counts", return_value="combined_step_counts"
    )
    mc = mocker.patch(
        "model.results._generate_results_json", return_value="dict_results"
    )

    # act
    actual = combine_results("results")

    # assert
    assert actual == "dict_results"
    ma.assert_called_once_with("results")
    mb.assert_called_once_with("results")
    mc.assert_called_once_with("combined_results", "combined_step_counts")
