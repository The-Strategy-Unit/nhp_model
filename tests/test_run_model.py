"""test run_model.py"""

from collections import namedtuple
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from run_model import debug_run, main, run_model, timeit


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

    result = namedtuple("result", ["pod", "measure"])
    timeit_mock = mocker.patch(
        "run_model.timeit",
        side_effect=[
            (
                pd.DataFrame(
                    {
                        "change_factor": ["a"] * 4 + ["b"] * 4,
                        "strategy": ["-"] * 8,
                        "measure": ["a", "b"] * 4,
                        "value": list(range(8)),
                    }
                ),
                "results",
            ),
            {
                "default": {
                    result("a", "a"): 1,
                    result("b", "a"): 2,
                    result("c", "a"): 3,
                }
            },
        ],
    )
    debug_run(m, 0)

    assert timeit_mock.call_count == 2
    assert timeit_mock.call_args_list[0] == call(m.run, 0)
    assert timeit_mock.call_args_list[1] == call(m.aggregate, "results", 0)

    assert capsys.readouterr().out == "\n".join(
        [
            "running model... aggregating results... ",
            "change factors:",
            "measure         a   b",
            "change_factor        ",
            "a               2   4",
            "b              10  12",
            "results        12  16",
            "",
            "aggregated (default) results:",
            "        value",
            "measure     a",
            "pod          ",
            "a           1",
            "b           2",
            "c           3",
            "total       6",
            "",
        ]
    )


def test_run_model_serial():
    """it runs in serial when cpus == 1"""
    # arrange
    save_model_mock = Mock()
    model_mock = Mock()
    # act
    run_model(save_model_mock, 0, 10, 1)(model_mock)
    # assert
    save_model_mock.set_model.assert_called_once_with(model_mock)
    assert save_model_mock.run_model.call_count == 10
    for i in range(10):
        assert save_model_mock.run_model.call_args_list[i] == call(i)


def test_run_model_parallel(mocker):
    """it runs in parallel when cpus > 1"""
    # arrange
    save_model_mock = Mock()
    model_mock = Mock()
    pool_mock = mocker.patch("run_model.Pool")
    pool_ctm = pool_mock.return_value.__enter__.return_value
    pool_ctm.name = "pool"
    pool_ctm.imap = Mock(wraps=lambda f, i, **kwargs: map(f, i))
    # act
    run_model(save_model_mock, 0, 10, 2)(model_mock)
    # assert
    save_model_mock.set_model.assert_called_once_with(model_mock)
    assert save_model_mock.run_model.call_count == 10
    for i in range(10):
        assert save_model_mock.run_model.call_args_list[i] == call(i)
    pool_mock.assert_called_once_with(2)
    pool_ctm.imap.assert_called_once_with(
        save_model_mock.run_model, range(0, 10), chunksize=8
    )


def test_run_model_dataset_file_not_found(capsys):
    """ "it skips missing dataset files"""
    save_model_mock = Mock()
    # cheat: raise the exception earlier than would normally be expected
    save_model_mock.set_model.side_effect = FileNotFoundError(0, "filename.parquet")
    model_mock = Mock()
    # act
    run_model(save_model_mock, 0, 10, 2)(model_mock)
    # assert
    assert (
        capsys.readouterr().out
        == "file [Errno 0] filename.parquet not found: skipping\n"
    )


def test_run_model_other_file_not_found():
    """it raises file not found for other files that are missing"""
    save_model_mock = Mock()
    # cheat: raise the exception earlier than would normally be expected
    save_model_mock.set_model.side_effect = FileNotFoundError(0, "filename.csv")
    model_mock = Mock()
    # act
    with pytest.raises(FileNotFoundError):
        run_model(save_model_mock, 0, 10, 2)(model_mock)


def test_main_debug_cant_use_all(mocker):
    """it raises an error if debug is set and type == all"""
    args = Mock()
    args.debug = True
    args.type = "all"
    args.params_file = "params.json"
    mocker.patch("run_model._run_model_argparser", return_value=args)
    mocker.patch("run_model.load_params")
    with pytest.raises(Exception):
        main()


@pytest.mark.parametrize(
    "activity_type, model_class",
    [("aae", "AaEModel"), ("ip", "InpatientsModel"), ("op", "OutpatientsModel")],
)
def test_main_debug_runs_model(mocker, activity_type, model_class):
    """it raises an error if debug is set and type == all"""
    # arrange
    args = Mock
    args.type = activity_type
    args.debug = True
    args.data_path = "data"
    args.run_start = 0
    args.params_file = "params.json"
    mocker.patch("run_model._run_model_argparser", return_value=args)
    mocker.patch("run_model.load_params", return_value="params")
    model_mock = mocker.patch(f"run_model.{model_class}", return_value="model")
    debug_mock = mocker.patch("run_model.debug_run")
    # act
    main()
    # assert
    model_mock.assert_called_once_with("params", "data")
    debug_mock.assert_called_once_with("model", 0)


@pytest.mark.parametrize(
    "model_save_type, model_save_class, run_post_runs",
    [
        (i, j, k)
        for i, j in [("local", "LocalSave"), ("cosmos", "CosmosDBSave")]
        for k in [True, False]
    ],
)
def test_main_non_debug(mocker, model_save_type, model_save_class, run_post_runs):
    """it should call the run model function"""
    # arrange
    args = Mock
    args.type = "all"
    args.debug = False
    args.save_type = model_save_type
    args.data_path = "data"
    args.results_path = "results"
    args.temp_results_path = "temp_results"
    args.save_results = True
    args.run_start = 0
    args.model_runs = 10
    args.cpus = 2
    args.batch_size = 8
    args.params_file = "params.json"
    args.run_postruns = run_post_runs
    mocker.patch("run_model._run_model_argparser", return_value=args)
    mocker.patch("run_model.load_params", return_value="params")
    model_save_mock = mocker.patch(f"run_model.{model_save_class}")
    runner_mock = Mock()
    run_model_mock = mocker.patch("run_model.run_model", runner_mock)
    aae_mock = mocker.patch("run_model.AaEModel")
    ip_mock = mocker.patch("run_model.InpatientsModel")
    op_mock = mocker.patch("run_model.OutpatientsModel")
    # act
    main()
    # assert
    model_save_mock.assert_called_once_with("params", "results", "temp_results", True)
    run_model_mock.assert_called_once_with(model_save_mock(), 0, 10, 2, 8)
    assert runner_mock().call_count == 3
    assert aae_mock.called_once_with("params", "data")
    assert ip_mock.called_once_with("params", "data")
    assert op_mock.called_once_with("params", "data")
    assert runner_mock().call_args_list[0][0][0] == aae_mock()
    assert runner_mock().call_args_list[1][0][0] == ip_mock()
    assert runner_mock().call_args_list[2][0][0] == op_mock()
    assert model_save_mock().post_runs.call_count == (1 if run_post_runs else 0)


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import run_model as r

    main_mock = mocker.patch("run_model.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
