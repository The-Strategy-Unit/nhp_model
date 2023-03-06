"""test run_model.py"""

from collections import namedtuple
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.outpatients import OutpatientsModel
from run_model import debug_run, main, timeit


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


def test_debug_run(mocker, capsys):
    """it should run the model and display outputs"""

    m = Mock()

    mr_mock = Mock()
    result = namedtuple("result", ["pod", "measure"])
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
    debug_run(m, "params", "data", 0)

    assert timeit_mock.call_count == 3
    assert timeit_mock.call_args_list[0] == call(m, "params", "data")
    # assert timeit_mock.call_args_list[1] == call(m.aggregate, mr_mock)
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
    args.params_file = "params.json"
    mocker.patch("run_model._run_model_argparser", return_value=args)
    mocker.patch("run_model.load_params", return_value="params")

    debug_mock = mocker.patch("run_model.debug_run")

    # act
    main()

    # assert
    debug_mock.assert_called_once_with(model_class, "params", "data", 0)


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import run_model as r

    main_mock = mocker.patch("run_model.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
