"""test results.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from model.results import (
    _add_metadata_to_dataframe,
    _combine_model_results,
    _combine_step_counts,
    _complete_model_runs,
    _patch_converted_sdec_activity,
    _save_params_file,
    _save_parquet_file,
    combine_results,
    generate_results_json,
    save_results_files,
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

    # duplicate the dataframe to test it properly reaggregates
    df = pd.concat([df, df])

    results = [df.assign(a=0), df.assign(a=1, value=lambda r: r["value"] + 100)]

    # act
    actual = _complete_model_runs(results, 3)

    # assert
    # we should have added in two (missing) rows
    assert len(actual) * 2 == sum(len(i) + 2 for i in results)
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


def test_generate_results_json(mocker):
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

    os_m = mocker.patch("os.makedirs")
    jd_m = mocker.patch("json.dump")

    json_content = {
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

    params = {
        "dataset": "synthetic",
        "scenario": "test",
        "create_datetime": "create_datetime",
    }

    run_params = {"variant": [1, 2, 3]}

    expected = "synthetic/test-create_datetime"

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        actual = generate_results_json(
            combined_results, combined_step_counts, params, run_params
        )

    # assert
    assert actual == expected
    mock_file.assert_called_once_with(
        "results/synthetic/test-create_datetime.json", "w", encoding="utf-8"
    )
    os_m.assert_called_once_with("results/synthetic", exist_ok=True)
    jd_m.assert_called_once_with(
        {
            "params": params,
            "population_variants": [1, 2, 3],
            "results": json_content,
        },
        mock_file(),
    )


def test_combine_results(mocker):
    # arrange
    ma = mocker.patch(
        "model.results._combine_model_results", return_value="combined_results"
    )
    mb = mocker.patch(
        "model.results._combine_step_counts", return_value="combined_step_counts"
    )
    ms = mocker.patch("model.results._patch_converted_sdec_activity")

    # act
    actual = combine_results("results")

    # assert
    assert actual == ("combined_results", "combined_step_counts")
    ma.assert_called_once_with("results")
    mb.assert_called_once_with("results")

    assert ms.call_count == 2
    assert list(ms.call_args_list) == [
        call("combined_results", "acuity", "standard"),
        call("combined_results", "attendance_category", "1"),
    ]


def test_save_results_files(mocker):
    # arrange
    results = {"default": "default_df", "step_counts": "step_counts_df"}
    params = {
        "dataset": "synthetic",
        "scenario": "test",
        "create_datetime": "create_datetime",
    }

    save_parquet_mock = mocker.patch(
        "model.results._save_parquet_file",
        side_effect=["default.parquet", "step_counts.parquet"],
    )
    save_params_mock = mocker.patch(
        "model.results._save_params_file", return_value="params.json"
    )

    os_m = mocker.patch("os.makedirs")

    path = "results/synthetic/test/create_datetime"

    expected = [
        "default.parquet",
        "step_counts.parquet",
        "params.json",
    ]

    # act
    actual = save_results_files(results, params)

    # assert
    assert actual == expected
    os_m.assert_called_once_with(path, exist_ok=True)

    assert save_parquet_mock.call_args_list == [
        call(path, "default", "default_df", params),
        call(path, "step_counts", "step_counts_df", params),
    ]

    assert save_params_mock.called_once_with(path, params)


def test_save_parquet_file(mocker):
    # arrange
    df = Mock()
    params = Mock()
    add_metadata_to_dataframe_mock = mocker.patch(
        "model.results._add_metadata_to_dataframe", return_value=df
    )

    # act
    actual = _save_parquet_file("path", "file", df, params)

    # assert
    assert actual == "path/file.parquet"
    add_metadata_to_dataframe_mock.assert_called_once_with(df, params)
    df.to_parquet.assert_called_once_with(actual)


def test_add_metadata_to_dataframe(mocker):
    # arrange
    df = pd.DataFrame({"one": [1], "two": [2]})
    params = {
        "dataset": "dataset",
        "scenario": "scenario",
        "app_version": "app_version",
        "create_datetime": "create_datetime",
    }
    expected = {
        "one": [1],
        "two": [2],
        "dataset": ["dataset"],
        "app_version": ["app_version"],
        "scenario": ["scenario"],
        "create_datetime": ["create_datetime"],
    }

    # act
    actual = _add_metadata_to_dataframe(df, params)

    # assert
    assert actual.to_dict("list") == expected


def test_save_params_file(mocker):
    # arrange
    j_mock = mocker.patch("json.dump")

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        actual = _save_params_file("path", "params")

    # assert
    assert actual == "path/params.json"
    mock_file.assert_called_once_with(actual, "w", encoding="utf-8")
    j_mock.assert_called_once_with("params", mock_file())


def test_patch_converted_sdec_activity():
    # arrange
    results = {
        "default": pd.DataFrame(
            {
                "pod": ["aae_type-01"] * 8 + ["aae_type-05"] * 8,
                "sitetret": (["a"] * 4 + ["b"] * 4) * 2,
                "measure": ["arrivals"] * 16,
                "model_run": [1, 2, 3, 4] * 4,
                "value": range(16),
            }
        ),
        "acuity": pd.DataFrame(
            {
                "pod": ["aae_type-01"] * 8 + ["aae_type-05"] * 4,
                "sitetret": ["a"] * 4 + ["b"] * 4 + ["a"] * 4,
                "measure": ["arrivals"] * 12,
                "acuity": ["urgent"] * 8 + ["standard"] * 4,
                "model_run": [1, 2, 3, 4] * 3,
                "value": range(12),
            }
        ),
    }

    expected = pd.DataFrame(
        {
            "pod": ["aae_type-01"] * 8 + ["aae_type-05"] * 8,
            "sitetret": (["a"] * 4 + ["b"] * 4) * 2,
            "measure": ["arrivals"] * 16,
            "acuity": ["urgent"] * 8 + ["standard"] * 8,
            "model_run": [1, 2, 3, 4] * 4,
            "value": range(16),
        }
    )
    # act
    _patch_converted_sdec_activity(results, "acuity", "standard")
    actual = results["acuity"].sort_values(
        ["pod", "sitetret", "measure", "acuity", "model_run"]
    )
    # assert
    pd.testing.assert_frame_equal(actual, expected)
