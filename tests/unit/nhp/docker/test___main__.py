"""test docker run."""

from unittest.mock import Mock, patch

import pytest

from nhp.docker.__main__ import main, parse_args


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

    local_data_mock = mocker.patch("nhp.docker.__main__.Local")
    local_data_mock.create.return_value = "data"

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

    local_data_mock.create.assert_called_once_with("data")


def test_main_azure(mocker):
    # arrange
    m = mocker.patch("nhp.docker.__main__.parse_args")
    m().params_file = "params.json"
    m().local_storage = False
    m().save_full_model_results = False

    rwls = mocker.patch("nhp.docker.__main__.RunWithLocalStorage")
    rwas = mocker.patch("nhp.docker.__main__.RunWithAzureStorage")

    local_data_mock = mocker.patch("nhp.docker.__main__.Local")
    local_data_mock.create.return_value = "data"

    config = Mock()
    config.APP_VERSION = "dev"
    config.DATA_VERSION = "dev"
    config.STORAGE_ACCOUNT = "sa"

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
    main(config)

    # assert
    rwls.assert_not_called()
    rwas.assert_called_once_with("params.json", config)

    s = rwas()
    ru_m.assert_called_once_with(params, "data", s.progress_callback(), False)
    s.finish.assert_called_once_with("results.json", "list_of_results", False)

    local_data_mock.create.assert_called_once_with("data")


def test_init(mocker):
    """It should run the main method if __name__ is __main__."""
    config = mocker.patch("nhp.docker.__main__.Config")

    import nhp.docker.__main__ as r

    main_mock = mocker.patch("nhp.docker.__main__.main")

    r.init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(r, "__name__", "__main__"):
        r.init()  # should call main
        main_mock.assert_called_once_with(config())


def test_init_catches_exception(mocker):
    # arrange
    mocker.patch("nhp.docker.__main__.main", side_effect=Exception("Test error"))
    import nhp.docker.__main__ as r

    m = mocker.patch("logging.error")

    # act
    with patch.object(r, "__name__", "__main__"):
        with pytest.raises(Exception, match="Test error"):
            r.init()

    # assert
    m.assert_called_once_with("An error occurred: %s", "Test error")
