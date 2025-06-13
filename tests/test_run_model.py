"""test run_model.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from nhp.model.__main__ import (  # TODO: change structure slightly to avoid importing from __main__
    _run_model,
    main,
    run_all,
    run_single_model_run,
    timeit,
    tqdm,
)
from nhp.model.aae import AaEModel
from nhp.model.inpatients import InpatientsModel
from nhp.model.outpatients import OutpatientsModel


def test_tqdm():
    tqdm.progress_callback = Mock()
    t = tqdm()
    t.update(5)
    tqdm.progress_callback.assert_called_once_with(5)


def test_tqdm_no_callback():
    tqdm.progress_callback = None
    t = tqdm()
    t.update(5)


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


def test_run_model(mocker):
    # arrange
    model_m = Mock()
    model_m.__name__ = "InpatientsModel"

    params = {"start_year": 2020, "end_year": 2022, "model_runs": 2}
    mocker.patch("os.cpu_count", return_value=2)

    pool_mock = mocker.patch("nhp.model.__main__.Pool")
    pool_ctm = pool_mock.return_value.__enter__.return_value
    pool_ctm.name = "pool"
    pool_ctm.imap = Mock(wraps=lambda f, i, **kwargs: map(f, i))

    pc_m = Mock()

    # act
    actual = _run_model(model_m, params, "data", "hsa", "run_params", pc_m, False)

    # assert
    pool_ctm.imap.assert_called_once_with(model_m().go, [0, 1, 2], chunksize=1)
    assert actual == [model_m().go()] * 3
    pc_m.assert_not_called()


def test_run_all(mocker):
    # arrange
    grp_m = mocker.patch(
        "nhp.model.__main__.Model.generate_run_params",
        return_value={"variant": "variants"},
    )
    hsa_m = mocker.patch(
        "nhp.model.__main__.HealthStatusAdjustmentInterpolated", return_value="hsa"
    )

    rm_m = mocker.patch("nhp.model.__main__._run_model", side_effect=["ip", "op", "aae"])
    cr_m = mocker.patch(
        "nhp.model.__main__.combine_results",
        return_value=({"default": "combined_results"}, "combined_step_counts"),
    )
    gr_m = mocker.patch(
        "nhp.model.__main__.generate_results_json", return_value="results_json_path"
    )
    sr_m = mocker.patch(
        "nhp.model.__main__.save_results_files", return_value="results_paths"
    )
    nd_m = mocker.patch("nhp.model.__main__.Local")

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

    # act
    actual = run_all(params, "data_path", pc_m, False)

    # assert
    assert actual == ("results_paths", "results_json_path")

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

    cr_m.assert_called_once_with(["ip", "op", "aae"])
    # weaker form of checking, but as we intended to drop this function in the future don't expend
    # effort to fixing this part of the test
    gr_m.assert_called_once()
    # gr_m.assert_called_once_with(
    #     {"default": "combined_results"},
    #     "combined_step_counts",
    #     params,
    #     {"variant": "variants"},
    # )
    sr_m.assert_called_once_with(
        {"default": "combined_results", "step_counts": "combined_step_counts"}, params
    )


def test_run_single_model_run(mocker, capsys):
    """it should run the model and display outputs"""
    # arrange
    mr_mock = Mock()
    ndl_mock = mocker.patch("nhp.model.__main__.Local")
    ndl_mock.create.return_value = "nhp_data"

    results_m = {
        "default": pd.DataFrame(
            {
                "pod": ["a", "b"] * 4 + ["c"],
                "measure": [i for i in ["x", "y"] for _ in [1, 2]] * 2 + ["x"],
                "value": range(9),
            }
        )
    }
    step_counts_m = pd.DataFrame(
        {
            "change_factor": ["a", "b"] * 4 + ["c"],
            "measure": [i for i in ["x", "y"] for _ in [1, 2]] * 2 + ["x"],
            "value": range(9),
        }
    )

    timeit_mock = mocker.patch(
        "nhp.model.__main__.timeit",
        side_effect=[None, mr_mock, (results_m, step_counts_m)],
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
            "              value    ",
            "measure           x   y",
            "change_factor          ",
            "a                 4   8",
            "b                 6  10",
            "c                 8   0",
            "total            18  18",
            "",
            "aggregated (default) results:",
            "        value      ",
            "measure     x     y",
            "pod                ",
            "a         4.0   8.0",
            "b         6.0  10.0",
            "c         8.0   0.0",
            "total    18.0  18.0",
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
    mocker.patch("nhp.model.__main__._run_model_argparser", return_value=args)
    ldp_mock = mocker.patch("nhp.model.__main__.load_params", return_value="params")

    run_all_mock = mocker.patch("nhp.model.__main__.run_all")
    run_single_mock = mocker.patch("nhp.model.__main__.run_single_model_run")

    # act
    main()

    # assert
    run_all_mock.assert_not_called()
    run_single_mock.assert_called_once_with("params", "data", model_class, 0)
    ldp_mock.assert_called_once_with("queue/params.json")


def test_main_debug_runs_model_invalid_type(mocker):
    # arrange
    args = Mock
    args.type = "invalid"
    args.data_path = "data"
    args.model_run = 0
    args.params_file = "queue/params.json"
    mocker.patch("nhp.model.__main__._run_model_argparser", return_value=args)
    mocker.patch("nhp.model.__main__.load_params", return_value="params")

    run_all_mock = mocker.patch("nhp.model.__main__.run_all")
    run_single_mock = mocker.patch("nhp.model.__main__.run_single_model_run")

    # act
    with pytest.raises(ValueError):
        main()

    # assert
    run_all_mock.assert_not_called()
    run_single_mock.assert_not_called()


def test_main_all_runs(mocker):
    # arrange
    args = Mock
    args.type = "all"
    args.data_path = "data"
    args.params_file = "queue/params.json"
    args.save_full_model_results = False
    mocker.patch("nhp.model.__main__._run_model_argparser", return_value=args)
    ldp_mock = mocker.patch("nhp.model.__main__.load_params", return_value="params")

    run_all_mock = mocker.patch("nhp.model.__main__.run_all")
    run_single_mock = mocker.patch("nhp.model.__main__.run_single_model_run")

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
    import nhp.model.__main__ as r  # pylint: disable=import-outside-toplevel

    main_mock = mocker.patch("nhp.model.__main__.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()
