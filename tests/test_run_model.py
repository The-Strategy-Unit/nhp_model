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
)


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
            for j in range(4)
        ],
        [{"a": {frozenset({("measure", "a"), ("pod", "a")}): 100}}],
    ]
    expected = {
        "a": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 101,
                "principal": 21,
                "model_runs": [41, 61],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 2,
                "principal": 22,
                "model_runs": [42, 62],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 3,
                "principal": 23,
                "model_runs": [43, 63],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 4,
                "principal": 24,
                "model_runs": [44, 64],
            },
        ],
        "b": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 5,
                "principal": 25,
                "model_runs": [45, 65],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 6,
                "principal": 26,
                "model_runs": [46, 66],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 7,
                "principal": 27,
                "model_runs": [47, 67],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 8,
                "principal": 28,
                "model_runs": [48, 68],
            },
        ],
        "c": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 9,
                "principal": 29,
                "model_runs": [49, 69],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 10,
                "principal": 30,
                "model_runs": [50, 70],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 11,
                "principal": 31,
                "model_runs": [51, 71],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 12,
                "principal": 32,
                "model_runs": [52, 72],
            },
        ],
        "d": [
            {
                "pod": "a",
                "measure": "a",
                "baseline": 13,
                "principal": 33,
                "model_runs": [53, 73],
            },
            {
                "pod": "b",
                "measure": "a",
                "baseline": 14,
                "principal": 34,
                "model_runs": [54, 74],
            },
            {
                "pod": "a",
                "measure": "b",
                "baseline": 15,
                "principal": 35,
                "model_runs": [55, 75],
            },
            {
                "pod": "b",
                "measure": "b",
                "baseline": 16,
                "principal": 36,
                "model_runs": [56, 76],
            },
        ],
        "step_counts": [
            {
                "change_factor": "baseline",
                "strategy": "-",
                "baseline": [0, 1],
                "principal": [0, 1],
                "model_runs": [[0, 1], [0, 1]],
            },
            {
                "change_factor": "a",
                "strategy": "a",
                "baseline": [2, 3],
                "principal": [3, 4],
                "model_runs": [[4, 5], [5, 6]],
            },
        ],
    }

    # act
    actual = _combine_results(results)

    # assert
    assert actual == expected
    assert m.call_args_list == [call(k, v) for k, v in expected.items()]


@pytest.mark.parametrize(
    "agg_type",
    [
        "default",
        "bed_occupancy",
        "theatres_available",
    ],
)
def test_split_model_runs_out_default(agg_type):
    # arrange
    results = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
        }
    ]
    expected = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
            "lwr_ci": 5.0,
            "median": 50.0,
            "upr_ci": 95.0,
        }
    ]

    # act
    _split_model_runs_out(agg_type, results)

    # assert
    assert results == expected


def test_split_model_runs_out_other():
    # arrange
    results = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "model_runs": list(range(101)),
        }
    ]
    expected = [
        {
            "pod": "a",
            "measure": "a",
            "baseline": 1,
            "principal": 2,
            "lwr_ci": 5.0,
            "median": 50.0,
            "upr_ci": 95.0,
        }
    ]

    # act
    _split_model_runs_out("other", results)

    # assert
    assert results == expected


def test_split_model_runs_out_step_counts():
    # arrange
    results = [
        {
            "change_factor": "baseline",
            "strategy": "-",
            "baseline": [0, 1],
            "principal": [0, 1],
            "model_runs": [[0, 1], [0, 1]],
        },
        {
            "change_factor": "a",
            "strategy": "a",
            "baseline": [2, 3],
            "principal": [3, 4],
            "model_runs": [[4, 5], [5, 6]],
        },
    ]
    expected = [
        {"change_factor": "baseline", "value": [0, 1]},
        {
            "change_factor": "a",
            "strategy": "a",
            "value": [3, 4],
            "model_runs": [[4, 5], [5, 6]],
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

    params = {"model_runs": 2}
    mocker.patch("os.cpu_count", return_value=2)

    pool_mock = mocker.patch("run_model.Pool")
    pool_ctm = pool_mock.return_value.__enter__.return_value
    pool_ctm.name = "pool"
    pool_ctm.imap = Mock(wraps=lambda f, i, **kwargs: map(f, i))

    # act
    actual = _run_model(model_m, params, "data", "hsa", "run_params")

    # assert
    pool_ctm.imap.assert_called_once_with(model_m().go, [-1, 0, 1, 2], chunksize=16)
    assert actual == [model_m().go()] * 4


def test_run_all(mocker):
    # arrange
    grp_m = mocker.patch(
        "run_model.Model.generate_run_params", return_value={"variant": "variants"}
    )
    hsa_m = mocker.patch(
        "run_model.HealthStatusAdjustmentInterpolated", return_value="hsa"
    )

    rm_m = mocker.patch("run_model._run_model", side_effect=["ip", "op", "aae"])
    cr_m = mocker.patch(
        "run_model._combine_results", return_value={"results": "results"}
    )

    params = {"id": 1, "dataset": "synthetic", "life_expectancy": "le"}

    jd_m = mocker.patch("json.dump")

    # act
    with patch("builtins.open", mock_open()) as mock_file:
        actual = run_all(params, "data")

    # assert
    assert actual == "1.json"

    grp_m.assert_called_once_with(params)
    hsa_m.assert_called_once_with("data/synthetic", "le")

    assert rm_m.call_args_list == [
        call(m, params, "data", "hsa", {"variant": "variants"})
        for m in [InpatientsModel, OutpatientsModel, AaEModel]
    ]

    cr_m.assert_called_once_with(["ip", "op", "aae"])

    mock_file.assert_called_once_with("results/1.json", "w", encoding="utf-8")
    jd_m.assert_called_once_with(
        {"results": "results", "population_variants": "variants"}, mock_file()
    )


def test_run_single_model_run(mocker, capsys):
    """it should run the model and display outputs"""
    # arrange
    m = Mock()

    mr_mock = Mock()
    grp_mock = mocker.patch(
        "run_model.Model.generate_run_params", return_value="run_params"
    )
    hsa_mock = mocker.patch(
        "run_model.HealthStatusAdjustmentInterpolated",
        return_value="hsa",
    )
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
    params = {"dataset": "synthetic", "life_expectancy": "life_expectancy"}

    # act
    run_single_model_run(params, "data", "model_type", 0)

    # assert
    grp_mock.assert_called_once_with(params)
    hsa_mock.assert_called_once_with("data/synthetic", "life_expectancy")
    assert timeit_mock.call_count == 3
    assert timeit_mock.call_args_list[0] == call(
        "model_type", params, "data", "hsa", "run_params"
    )
    assert timeit_mock.call_args_list[2] == call(mr_mock.get_aggregate_results)

    assert capsys.readouterr().out == "\n".join(
        [
            "initialising model...  running model...       aggregating results... ",
            "change factors:",
            "                            value        ",
            "measure                admissions beddays",
            "change_factor                            ",
            "baseline                    100.0   200.0",
            "demographic_adjustment       50.0   100.0",
            "total                       150.0   300.0",
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
    mocker.patch("run_model._run_model_argparser", return_value=args)
    ldp_mock = mocker.patch("run_model.load_params", return_value="params")

    run_all_mock = mocker.patch("run_model.run_all")
    run_single_mock = mocker.patch("run_model.run_single_model_run")

    # act
    main()

    # assert
    run_all_mock.assert_called_once_with("params", "data")
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
