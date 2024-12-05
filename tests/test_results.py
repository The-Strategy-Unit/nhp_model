"""test results.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

import pandas as pd
import pytest

from model.results import (
    _combine_model_results,
    _combine_step_counts,
    _complete_model_runs,
    combine_results,
    generate_results_json,
)


def test_complete_model_runs_include_baseline():
    # arrange
    df = pd.DataFrame(
        [
            {"b": b, "model_run": mr, "value": b * 10 + mr}
            for b in range(2)
            for mr in range(4)
            if not (b == 1 and mr == 2)
        ]
    )

    results = [df.assign(a=0), df.assign(a=1, value=lambda r: r["value"] + 100)]

    # act
    actual = _complete_model_runs(results, 3)

    # assert
    # we should have added in two (missing) rows
    assert len(actual) == sum(len(i) + 1 for i in results)
    # but the sum of the value column should be unchanged
    assert actual["value"].sum() == sum(i["value"].sum() for i in results)
    # each group should have 3 model runs + baseline
    assert all(actual.value_counts(["a", "b"]) == 4)


def test_complete_model_runs_exclude_baseline():
    # arrange
    df = pd.DataFrame(
        [
            {"b": b, "model_run": mr, "value": b * 10 + mr}
            for b in range(2)
            for mr in range(1, 4)
            if not (b == 1 and mr == 2)
        ]
    )

    results = [df.assign(a=0), df.assign(a=1, value=lambda r: r["value"] + 100)]

    # act
    actual = _complete_model_runs(results, 3, False)

    # assert
    # we should have added in two (missing) rows
    assert len(actual) == sum(len(i) + 1 for i in results)
    # but the sum of the value column should be unchanged
    assert actual["value"].sum() == sum(i["value"].sum() for i in results)
    # each group should have 3 model runs
    assert all(actual.value_counts(["a", "b"]) == 3)


def test_combine_model_results(mocker):
    # arrange
    def r(*args):
        return pd.Series(args, name="value")

    results = [
        [
            ({"default": r(1, 2, 3), "other": r(4, 5, 6)}, None),
            ({"default": r(4, 5, 6), "other": r(1, 2, 3)}, None),
        ],
        [
            (
                {
                    "default": r(7),
                },
                None,
            ),
            (
                {
                    "default": r(8),
                },
                None,
            ),
        ],
    ]

    cmr_mock = mocker.patch("model.results._complete_model_runs", return_value="cmr")

    # act
    actual = _combine_model_results(results)

    # assert
    assert actual == {"default": "cmr", "other": "cmr"}

    assert [i["value"].sum() for i in cmr_mock.call_args_list[0][0][0]] == [6, 15, 7, 8]
    assert cmr_mock.call_args_list[0][0][1] == 1
    assert [i["value"].sum() for i in cmr_mock.call_args_list[1][0][0]] == [15, 6]
    assert cmr_mock.call_args_list[0][0][1] == 1


def test_combine_step_counts(mocker):
    # arrange
    df = pd.DataFrame(
        [
            {"a": 1, "b": 1, "value": 1},
            {"a": 1, "b": 2, "value": 2},
            {"a": 2, "b": 1, "value": 3},
            {"a": 2, "b": 2, "value": 4},
        ]
    ).set_index(["a", "b"])

    results = [
        [
            (None, df),
            (None, df),
        ],
        [
            (None, df),
            (None, df),
        ],
    ]

    expected = {
        "a": [1, 1, 2, 2],
        "b": [1, 2, 1, 2],
        "value": [1, 2, 3, 4],
        "model_run": [1, 1, 1, 1],
    }

    cmr_mock = mocker.patch("model.results._complete_model_runs", return_value="cmr")

    # act
    actual = _combine_step_counts(results)

    # assert
    assert actual == "cmr"
    assert all(i.to_dict("list") == expected for i in cmr_mock.call_args[0][0])
    assert cmr_mock.call_args[0][1] == 1
    assert cmr_mock.call_args[1] == {"include_baseline": False}


def test_generate_results_json():
    # arrange
    combined_results = {
        "default": pd.DataFrame(
            {
                "a": [i for i in [0, 1] for _ in range(5)],
                "model_run": list(range(5)) * 2,
                "value": range(10),
            }
        ),
        "a": pd.DataFrame(
            {
                "a": [i for i in [0, 1] for _ in list(range(5)) * 2],
                "b": [i for i in [0, 1] for _ in list(range(5))] * 2,
                "model_run": list(range(5)) * 4,
                "value": list(range(20)),
            }
        ),
    }

    combined_step_counts = pd.DataFrame(
        {
            "pod": ["a1"] * 4 * 5,
            "change_factor": ["baseline", "a", "b", "c", "c"] * 4,
            "strategy": ["-", "-", "-", "a", "b"] * 4,
            "sitetret": ["s"] * 4 * 5,
            "activity_type": ["a"] * 4 * 5,
            "measure": ["x"] * 4 * 5,
            "value": range(20),
        }
    )

    expected = {
        "default": [
            {"a": 0, "baseline": 0, "model_runs": [1, 2, 3, 4]},
            {"a": 1, "baseline": 5, "model_runs": [6, 7, 8, 9]},
        ],
        "a": [
            {"a": 0, "b": 0, "baseline": 0, "model_runs": [1, 2, 3, 4]},
            {"a": 0, "b": 1, "baseline": 5, "model_runs": [6, 7, 8, 9]},
            {"a": 1, "b": 0, "baseline": 10, "model_runs": [11, 12, 13, 14]},
            {"a": 1, "b": 1, "baseline": 15, "model_runs": [16, 17, 18, 19]},
        ],
        "step_counts": [
            {
                "pod": "a1",
                "change_factor": "a",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [1, 6, 11, 16],
            },
            {
                "pod": "a1",
                "change_factor": "b",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [2, 7, 12, 17],
            },
            {
                "pod": "a1",
                "change_factor": "baseline",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [0],
            },
            {
                "pod": "a1",
                "change_factor": "c",
                "strategy": "a",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [3, 8, 13, 18],
            },
            {
                "pod": "a1",
                "change_factor": "c",
                "strategy": "b",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [4, 9, 14, 19],
            },
        ],
    }

    # act
    actual = generate_results_json(combined_results, combined_step_counts)

    # assert
    assert actual == expected


def test_combine_results(mocker):
    # arrange
    ma = mocker.patch(
        "model.results._combine_model_results", return_value="combined_results"
    )
    mb = mocker.patch(
        "model.results._combine_step_counts", return_value="combined_step_counts"
    )

    # act
    actual = combine_results("results")

    # assert
    assert actual == ("combined_results", "combined_step_counts")
    ma.assert_called_once_with("results")
    mb.assert_called_once_with("results")
