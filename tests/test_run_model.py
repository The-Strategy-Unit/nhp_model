"""test run_model.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pytest

from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.outpatients import OutpatientsModel
from run_model import (
    _combine_results,
    _run_model,
    _split_model_runs_out,
    main,
    run_all,
    run_single_model_run,
    timeit,
    tqdm,
)


def test_tqdm():
    tqdm.progress_callback = Mock()
    t = tqdm()
    t.update(5)
    tqdm.progress_callback.assert_called_once_with(5)


def test_timeit(mocker, capsys):
    """it should evaluate a function and print how long it took to run it"""
    # arrange
    m = Mock(return_value="function")
    mocker.patch("time.time", return_value=0)
    # act
    actual = timeit(m, 1, 2, 3)
    # assert
    assert actual == "function"
    assert capsys.readouterr().out == "elapsed: 0.000s\n"


def test_combine_results(mocker):
    # arrange
    m = mocker.patch("run_model._split_model_runs_out")
    results = [
        [
            {
                **{
                    k: {
                        frozenset({("measure", "a"), ("pod", "a")}): 1 + i * 4 + j * 20,
                        frozenset({("measure", "a"), ("pod", "b")}): 2 + i * 4 + j * 20,
                        frozenset({("measure", "b"), ("pod", "a")}): 3 + i * 4 + j * 20,
                        frozenset({("measure", "b"), ("pod", "b")}): 4 + i * 4 + j * 20,
                    }
                    for k, i in [
                        ("a", 0),
                        ("b", 1),
                        ("c", 2),
                        ("d", 3),
                    ]
                },
                "step_counts": {
                    frozenset(
                        {
                            ("change_factor", "baseline"),
                            ("strategy", "-"),
                        }
                    ): [0, 1],
                    frozenset(
                        {
                            ("change_factor", "a"),
                            ("strategy", "a"),
                        }
                    ): [2 + j, 3 + j],
                },
            }
            for j in range(3)
        ],
        [{"a": {frozenset({("measure", "a"), ("pod", "a")}): 100}}],
    ]
    expected = {
        "a": [
            {
                "measure": "a",
                "pod": "a",
                "baseline": 101,
                "model_runs": [21, 41],
            },
            {
                "measure": "a",
                "pod": "b",
                "baseline": 2,
                "model_runs": [22, 42],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 3,
                "model_runs": [23, 43],
            },
            {
                "measure": "b",
                "pod": "b",
                "baseline": 4,
                "model_runs": [24, 44],
            },
        ],
        "b": [
            {
                "measure": "a",
                "pod": "a",
                "baseline": 5,
                "model_runs": [25, 45],
            },
            {
                "measure": "a",
                "pod": "b",
                "baseline": 6,
                "model_runs": [26, 46],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 7,
                "model_runs": [27, 47],
            },
            {
                "measure": "b",
                "pod": "b",
                "baseline": 8,
                "model_runs": [28, 48],
            },
        ],
        "c": [
            {
                "measure": "a",
                "pod": "a",
                "baseline": 9,
                "model_runs": [29, 49],
            },
            {
                "measure": "a",
                "pod": "b",
                "baseline": 10,
                "model_runs": [30, 50],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 11,
                "model_runs": [31, 51],
            },
            {
                "measure": "b",
                "pod": "b",
                "baseline": 12,
                "model_runs": [32, 52],
            },
        ],
        "d": [
            {
                "measure": "a",
                "pod": "a",
                "baseline": 13,
                "model_runs": [33, 53],
            },
            {
                "measure": "a",
                "pod": "b",
                "baseline": 14,
                "model_runs": [34, 54],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 15,
                "model_runs": [35, 55],
            },
            {
                "measure": "b",
                "pod": "b",
                "baseline": 16,
                "model_runs": [36, 56],
            },
        ],
        "step_counts": [
            {
                "strategy": "-",
                "change_factor": "baseline",
                "baseline": [0, 1],
                "model_runs": [[0, 1], [0, 1]],
            },
            {
                "strategy": "a",
                "change_factor": "a",
                "baseline": [2, 3],
                "model_runs": [[3, 4], [4, 5]],
            },
        ],
    }

    # act
    actual = _combine_results(results, 2)

    # assert
    assert actual == expected
    assert m.call_args_list == [call(k, v) for k, v in expected.items()]


def test_split_model_runs_out():
    # arrange
    results = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "model_runs": list(range(101)),
        }
    ]
    expected = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "model_runs": list(range(101)),
        }
    ]

    # act
    _split_model_runs_out("default", results)

    # assert
    assert results == expected


def test_split_model_runs_out_step_counts():
    # arrange
    results = [
        {
            "change_factor": "baseline",
            "strategy": "-",
            "baseline": 0,
            "model_runs": [1, 2],
            "time_profiles": [3, 4],
        },
        {
            "change_factor": "a",
            "strategy": "a",
            "baseline": 0,
            "model_runs": [5, 6],
            "time_profiles": [7, 8],
        },
    ]
    expected = [
        {"change_factor": "baseline", "model_runs": [1]},
        {
            "change_factor": "a",
            "strategy": "a",
            "model_runs": [5, 6],
            "time_profiles": [7, 8],
        },
    ]

    # act
    _split_model_runs_out("step_counts", results)

    # assert
    assert results == expected


def test_run_model(mocker):
    # arrange
    model_m = Mock()
    model_m.__name__ = "InpatientsModel"

    params = {"start_year": 2020, "end_year": 2022, "model_runs": 2}
    mocker.patch("os.cpu_count", return_value=2)

    pool_mock = mocker.patch("run_model.Pool")
    pool_ctm = pool_mock.return_value.__enter__.return_value
    pool_ctm.name = "pool"
    pool_ctm.imap = Mock(wraps=lambda f, i, **kwargs: map(f, i))

    pc_m = Mock()

    # act
    actual = _run_model(model_m, params, "data", "hsa", "run_params", pc_m, False)

    # assert
    pool_ctm.imap.assert_called_once_with(model_m().go, [0, 1, 2, 3], chunksize=1)
    assert actual == [model_m().go()] * 4
    pc_m.assert_not_called()


def test_run_all(mocker):
    # arrange
    grp_m = mocker.patch(
        "run_model.Model.generate_run_params", return_value={"variant": "variants"}
    )
    hsa_m = mocker.patch(
        "run_model.HealthStatusAdjustmentInterpolated", return_value="hsa"
    )

    rm_m = mocker.patch("run_model._run_model", side_effect=["ip", "op", "aae"])
    cr_m = mocker.patch("run_model._combine_results", return_value="combined_results")
    nd_m = mocker.patch("run_model.Local")

    os_m = mocker.patch("os.makedirs")

    pc_m = Mock()
    pc_m().return_value = "progress callback"
    pc_m.reset_mock()

    params = {
        "id": "1",
        "dataset": "synthetic",
        "scenario": "test",
        "start_year": 2020,
        "end_year": 2025,
        "model_runs": 10,
        "create_datetime": "20230123_012345",
    }

    jd_m = mocker.patch("json.dump")

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        actual = run_all(params, "data_path", pc_m, False)

    # assert
    assert actual == "synthetic/test-20230123_012345"

    nd_m.create.assert_called_once_with("data_path")
    nd_c = nd_m.create()
    nd_c.assert_called_once_with(2020, "synthetic")

    pc_m.assert_called_once_with()
    assert pc_m().call_args_list == [
        call("Inpatients"),
        call("Outpatients"),
        call("AaE"),
    ]

    grp_m.assert_called_once_with(params)
    hsa_m.assert_called_once_with(nd_c(2020, "synthetic"), 2020)

    assert rm_m.call_args_list == [
        call(
            m,
            params,
            nd_c,
            "hsa",
            {"variant": "variants"},
            "progress callback",
            False,
        )
        for m in [InpatientsModel, OutpatientsModel, AaEModel]
    ]

    cr_m.assert_called_once_with(["ip", "op", "aae"], 10)
    os_m.assert_called_once_with("results/synthetic", exist_ok=True)

    mock_file.assert_called_once_with(
        "results/synthetic/test-20230123_012345.json", "w", encoding="utf-8"
    )
    jd_m.assert_called_once_with(
        {
            "params": params,
            "population_variants": "variants",
            "results": "combined_results",
        },
        mock_file(),
    )


def test_run_single_model_run(mocker, capsys):
    """it should run the model and display outputs"""
    # arrange
    mr_mock = Mock()
    ndl_mock = mocker.patch("run_model.Local")
    ndl_mock.create.return_value = "nhp_data"

    timeit_mock = mocker.patch(
        "run_model.timeit",
        side_effect=[
            None,
            mr_mock,
            {
                "default": {
                    frozenset(
                        {
                            ("measure", "admissions"),
                            ("pod", "ip_elective_admission"),
                            ("sitetret", "RXX01"),
                        }
                    ): 100.0,
                    frozenset(
                        {
                            ("measure", "beddays"),
                            ("pod", "ip_elective_admission"),
                            ("sitetret", "RXX01"),
                        }
                    ): 200.0,
                    frozenset(
                        {
                            ("measure", "procedures"),
                            ("pod", "ip_elective_admission"),
                            ("sitetret", "RXX01"),
                        }
                    ): 300.0,
                    frozenset(
                        {
                            ("pod", "ip_elective_daycase"),
                            ("measure", "admissions"),
                            ("sitetret", "RXX01"),
                        }
                    ): 400.0,
                },
                "step_counts": {
                    frozenset(
                        {
                            ("activity_type", "ip"),
                            ("change_factor", "baseline"),
                            ("measure", "admissions"),
                            ("strategy", "-"),
                        }
                    ): 100.0,
                    frozenset(
                        {
                            ("activity_type", "ip"),
                            ("change_factor", "baseline"),
                            ("measure", "beddays"),
                            ("strategy", "-"),
                        }
                    ): 200.0,
                    frozenset(
                        {
                            ("activity_type", "ip"),
                            ("change_factor", "demographic_adjustment"),
                            ("measure", "admissions"),
                            ("strategy", "-"),
                        }
                    ): 50.0,
                    frozenset(
                        {
                            ("activity_type", "ip"),
                            ("change_factor", "demographic_adjustment"),
                            ("measure", "beddays"),
                            ("strategy", "-"),
                        }
                    ): 100.0,
                },
            },
        ],
    )
    params = {"dataset": "synthetic", "start_year": 2020, "end_year": 2025}

    # act
    run_single_model_run(params, "data", "model_type", 0)

    # assert
    ndl_mock.create.assert_called_once_with("data")

    assert timeit_mock.call_count == 3
    assert timeit_mock.call_args_list[0] == call("model_type", params, "nhp_data")
    assert timeit_mock.call_args_list[2] == call(mr_mock.get_aggregate_results)

    assert capsys.readouterr().out == "\n".join(
        [
            "initialising model...  running model...       aggregating results... ",
            "change factors:",
            "                            value        ",
            "measure                admissions beddays",
            "change_factor                            ",
            "baseline                      100     200",
            "demographic_adjustment         50     100",
            "total                         150     300",
            "",
            "aggregated (default) results:",
            "                           value                   ",
            "measure               admissions beddays procedures",
            "pod                                                ",
            "ip_elective_admission      100.0   200.0      300.0",
            "ip_elective_daycase        400.0     0.0        0.0",
            "total                      500.0   200.0      300.0",
            "",
        ]
    )


@pytest.mark.parametrize(
    "activity_type, model_class",
    [("aae", AaEModel), ("ip", InpatientsModel), ("op", OutpatientsModel)],
)
def test_main_debug_runs_model(mocker, activity_type, model_class):
    # arrange
    args = Mock
    args.type = activity_type
    args.data_path = "data"
    args.model_run = 0
    args.params_file = "queue/params.json"
    mocker.patch("run_model._run_model_argparser", return_value=args)
    ldp_mock = mocker.patch("run_model.load_params", return_value="params")

    run_all_mock = mocker.patch("run_model.run_all")
    run_single_mock = mocker.patch("run_model.run_single_model_run")

    # act
    main()

    # assert
    run_all_mock.assert_not_called()
    run_single_mock.assert_called_once_with("params", "data", model_class, 0)
    ldp_mock.assert_called_once_with("queue/params.json")


def test_main_all_runs(mocker):
    # arrange
    args = Mock
    args.type = "all"
    args.data_path = "data"
    args.params_file = "queue/params.json"
    args.save_full_model_results = False
    mocker.patch("run_model._run_model_argparser", return_value=args)
    ldp_mock = mocker.patch("run_model.load_params", return_value="params")

    run_all_mock = mocker.patch("run_model.run_all")
    run_single_mock = mocker.patch("run_model.run_single_model_run")

    # act
    main()

    # assert
    run_all_mock.assert_called_once()
    assert run_all_mock.call_args[0][0] == "params"
    assert run_all_mock.call_args[0][1] == "data"
    assert run_all_mock.call_args[0][2]()(0) is None

    run_single_mock.assert_not_called()
    ldp_mock.assert_called_once_with("queue/params.json")


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import run_model as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("run_model.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
