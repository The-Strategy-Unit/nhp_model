"""test run_model.py"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name, missing-function-docstring

from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from nhp.model.__main__ import main
from nhp.model.aae import AaEModel
from nhp.model.inpatients import InpatientsModel
from nhp.model.outpatients import OutpatientsModel


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
    mocker.patch("nhp.model.__main__.parse_args", return_value=args)
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
    mocker.patch("nhp.model.__main__.parse_args", return_value=args)
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
    mocker.patch("nhp.model.__main__.parse_args", return_value=args)
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
