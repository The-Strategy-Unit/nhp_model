"""test results.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

import pandas as pd

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
    combined_results = pd.DataFrame(
        {
            "model_run": list(range(5)) * 5,
            "pod": [i for i in ["a1", "a2", "b1", "b2", "c1"] for _ in range(5)],
            "sitetret": ["s1"] * 5 * 5,
            "measure": ["x"] * 5 * 2 + ["y"] * 5 * 2 + ["z"] * 5,
            "sex": [0] * 5 * 5,
            "age_group": [0] * 5 * 5,
            "age": [0] * 5 * 5,
            "acuity": [None] * 5 * 4 + [0] * 5,
            "attendance_category": [None] * 5 * 4 + [0] * 5,
            "tretspef": ["a"] * 5 * 4 + [None] * 5,
            "tretspef_raw": ["a"] * 5 * 4 + [None] * 5,
            "los_group": ["a"] * 5 * 2 + [None] * 5 * 3,
            "value": range(25),
        }
    )

    combined_step_counts = pd.DataFrame(
        {
            "pod": ["a1"] * 4 * 4,
            "change_factor": ["a", "b", "c", "c"] * 4,
            "strategy": ["-", "-", "a", "b"] * 4,
            "sitetret": ["s"] * 4 * 4,
            "activity_type": ["a"] * 4 * 4,
            "measure": ["x"] * 4 * 4,
            "value": [i for i in range(4) for _ in range(4)],
        }
    )

    expected = {
        "default": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
            {
                "pod": "b1",
                "sitetret": "s1",
                "measure": "y",
                "baseline": 10,
                "model_runs": [11, 12, 13, 14],
            },
            {
                "pod": "b2",
                "sitetret": "s1",
                "measure": "y",
                "baseline": 15,
                "model_runs": [16, 17, 18, 19],
            },
            {
                "pod": "c1",
                "sitetret": "s1",
                "measure": "z",
                "baseline": 20,
                "model_runs": [21, 22, 23, 24],
            },
        ],
        "sex+age_group": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "sex": 0,
                "age_group": 0,
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "sex": 0,
                "age_group": 0,
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
            {
                "pod": "b1",
                "sitetret": "s1",
                "sex": 0,
                "age_group": 0,
                "measure": "y",
                "baseline": 10,
                "model_runs": [11, 12, 13, 14],
            },
            {
                "pod": "b2",
                "sitetret": "s1",
                "sex": 0,
                "age_group": 0,
                "measure": "y",
                "baseline": 15,
                "model_runs": [16, 17, 18, 19],
            },
            {
                "pod": "c1",
                "sitetret": "s1",
                "sex": 0,
                "age_group": 0,
                "measure": "z",
                "baseline": 20,
                "model_runs": [21, 22, 23, 24],
            },
        ],
        "age": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "age": 0,
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "age": 0,
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
            {
                "pod": "b1",
                "sitetret": "s1",
                "age": 0,
                "measure": "y",
                "baseline": 10,
                "model_runs": [11, 12, 13, 14],
            },
            {
                "pod": "b2",
                "sitetret": "s1",
                "age": 0,
                "measure": "y",
                "baseline": 15,
                "model_runs": [16, 17, 18, 19],
            },
            {
                "pod": "c1",
                "sitetret": "s1",
                "age": 0,
                "measure": "z",
                "baseline": 20,
                "model_runs": [21, 22, 23, 24],
            },
        ],
        "acuity": [
            {
                "pod": "c1",
                "sitetret": "s1",
                "acuity": 0.0,
                "measure": "z",
                "baseline": 20,
                "model_runs": [21, 22, 23, 24],
            }
        ],
        "attendance_category": [
            {
                "pod": "c1",
                "sitetret": "s1",
                "attendance_category": 0.0,
                "measure": "z",
                "baseline": 20,
                "model_runs": [21, 22, 23, 24],
            }
        ],
        "sex+tretspef": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "sex": 0,
                "tretspef": "a",
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "sex": 0,
                "tretspef": "a",
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
            {
                "pod": "b1",
                "sitetret": "s1",
                "sex": 0,
                "tretspef": "a",
                "measure": "y",
                "baseline": 10,
                "model_runs": [11, 12, 13, 14],
            },
            {
                "pod": "b2",
                "sitetret": "s1",
                "sex": 0,
                "tretspef": "a",
                "measure": "y",
                "baseline": 15,
                "model_runs": [16, 17, 18, 19],
            },
        ],
        "tretspef_raw": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
            {
                "pod": "b1",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "measure": "y",
                "baseline": 10,
                "model_runs": [11, 12, 13, 14],
            },
            {
                "pod": "b2",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "measure": "y",
                "baseline": 15,
                "model_runs": [16, 17, 18, 19],
            },
        ],
        "tretspef_raw+los_group": [
            {
                "pod": "a1",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "los_group": "a",
                "measure": "x",
                "baseline": 0,
                "model_runs": [1, 2, 3, 4],
            },
            {
                "pod": "a2",
                "sitetret": "s1",
                "tretspef_raw": "a",
                "los_group": "a",
                "measure": "x",
                "baseline": 5,
                "model_runs": [6, 7, 8, 9],
            },
        ],
        "step_counts": [
            {
                "pod": "a1",
                "change_factor": "a",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [0, 1, 2, 3],
            },
            {
                "pod": "a1",
                "change_factor": "b",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [0, 1, 2, 3],
            },
            {
                "pod": "a1",
                "change_factor": "c",
                "strategy": "a",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [0, 1, 2, 3],
            },
            {
                "pod": "a1",
                "change_factor": "c",
                "strategy": "b",
                "sitetret": "s",
                "activity_type": "a",
                "measure": "x",
                "model_runs": [0, 1, 2, 3],
            },
        ],
    }

    # act
    actual = _generate_results_json(combined_results, combined_step_counts)

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
