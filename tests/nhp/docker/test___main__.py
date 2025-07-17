"""test docker run."""

import time
from unittest.mock import patch

import pytest

from nhp.docker import config
from nhp.docker.__main__ import main, parse_args


def test_exit_container(mocker):
    m = mocker.patch("os._exit")
    import nhp.docker.__main__ as r

    r._exit_container()

    m.assert_called_once_with(1)


@pytest.mark.parametrize(
    "args, expected_file, expected_local_storage, expected_save_full_model_results",
    [
        ([], "params-sample.json", False, False),
        (["-l"], "params-sample.json", True, False),
        (["test.json"], "test.json", False, False),
        (["test.json", "-l"], "test.json", True, False),
        (["--save-full-model-results"], "params-sample.json", False, True),
    ],
)
def test_parse_args(
    mocker,
    args,
    expected_file,
    expected_local_storage,
    expected_save_full_model_results,
):
    # arrange
    mocker.patch("sys.argv", ["nhp.docker.run.py"] + args)

    # act
    actual = parse_args()

    # assert
    assert actual.params_file == expected_file
    assert actual.local_storage == expected_local_storage
    assert actual.save_full_model_results == expected_save_full_model_results


def test_main_local(mocker):
    # arrange
    m = mocker.patch("nhp.docker.__main__.parse_args")
    m().params_file = "params.json"
    m().local_storage = True
    m().save_full_model_results = False

    rwls = mocker.patch("nhp.docker.__main__.RunWithLocalStorage")
    rwas = mocker.patch("nhp.docker.__main__.RunWithAzureStorage")

    params = {
        "model_runs": 256,
        "start_year": 2019,
        "end_year": 2035,
        "app_version": "dev",
    }

    rwls().params = params
    rwls.reset_mock()

    ru_m = mocker.patch(
        "nhp.docker.__main__.run_all", return_value=("list_of_results", "results.json")
    )

    # act
    main()

    # assert
    rwls.assert_called_once_with("params.json")
    rwas.assert_not_called()

    s = rwls()
    ru_m.assert_called_once_with(params, "data", s.progress_callback(), False)
    s.finish.assert_called_once_with("results.json", "list_of_results", False)


def test_main_azure(mocker):
    # arrange
    m = mocker.patch("nhp.docker.__main__.parse_args")
    m().params_file = "params.json"
    m().local_storage = False
    m().save_full_model_results = False

    rwls = mocker.patch("nhp.docker.__main__.RunWithLocalStorage")
    rwas = mocker.patch("nhp.docker.__main__.RunWithAzureStorage")

    params = {
        "model_runs": 256,
        "start_year": 2019,
        "end_year": 2035,
        "app_version": "dev",
    }

    rwas().params = params
    rwas.reset_mock()

    ru_m = mocker.patch(
        "nhp.docker.__main__.run_all", return_value=("list_of_results", "results.json")
    )

    # act
    main()

    # assert
    rwls.assert_not_called()
    rwas.assert_called_once_with("params.json", "dev")

    s = rwas()
    ru_m.assert_called_once_with(params, "data", s.progress_callback(), False)
    s.finish.assert_called_once_with("results.json", "list_of_results", False)


def test_init(mocker):
    """It should run the main method if __name__ is __main__."""
    import nhp.docker.__main__ as r

    main_mock = mocker.patch("nhp.docker.__main__.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once()


def test_init_timeout_call_exit(mocker):
    config.CONTAINER_TIMEOUT_SECONDS = 0.1

    import nhp.docker.__main__ as r

    main_mock = mocker.patch("nhp.docker.__main__.main")
    exit_mock = mocker.patch("nhp.docker.__main__._exit_container")
    main_mock.side_effect = lambda: time.sleep(0.2)
    with patch.object(r, "__name__", "__main__"):
        r.init()

    exit_mock.assert_called_once()


def test_init_timeout_dont_call_exit(mocker):
    config.CONTAINER_TIMEOUT_SECONDS = 0.1

    import nhp.docker.__main__ as r

    main_mock = mocker.patch("nhp.docker.__main__.main")
    exit_mock = mocker.patch("nhp.docker.__main__._exit_container")
    main_mock.side_effect = lambda: time.sleep(0.02)
    with patch.object(r, "__name__", "__main__"):
        r.init()

    exit_mock.assert_not_called()
