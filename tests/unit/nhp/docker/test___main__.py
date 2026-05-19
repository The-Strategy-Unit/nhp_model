"""test docker run."""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from nhp.docker.__main__ import main, parse_args


@pytest.mark.unit
@pytest.mark.parametrize(
    ", ".join(
        [
            "args",
            "expected_file",
            "expected_model_run_id",
            "expected_local_storage",
            "expected_save_full_model_results",
        ]
    ),
    [
        (["test.json"], "test.json", uuid.uuid4, False, False),
        (["test.json", "-l"], "test.json", uuid.uuid4, True, False),
        (
            ["test.json", "00000000-0000-0000-0000-000000000001"],
            "test.json",
            uuid.UUID("00000000-0000-0000-0000-000000000001"),
            False,
            False,
        ),
        (
            ["test.json", "00000000-0000-0000-0000-000000000001", "-l"],
            "test.json",
            uuid.UUID("00000000-0000-0000-0000-000000000001"),
            True,
            False,
        ),
        (["test.json", "--save-full-model-results"], "test.json", uuid.uuid4, False, True),
    ],
)
def test_parse_args(
    mocker,
    args,
    expected_file,
    expected_model_run_id,
    expected_local_storage,
    expected_save_full_model_results,
):
    # arrange
    mocker.patch("sys.argv", ["nhp.docker.run.py"] + args)

    # act
    actual = parse_args()

    # assert
    assert actual.params_file == expected_file
    assert actual.model_run_id == expected_model_run_id
    assert actual.local_storage == expected_local_storage
    assert actual.save_full_model_results == expected_save_full_model_results


@pytest.mark.unit
def test_main_local(mocker):
    # arrange
    m = mocker.patch("nhp.docker.__main__.parse_args")
    m().params_file = "params.json"
    m().model_run_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    m().local_storage = True
    m().save_full_model_results = False

    m_start_time = datetime(2025, 1, 1, 12, 0, 0)
    m_end_time = datetime(2025, 1, 1, 12, 0, 2)
    m_datetime = mocker.patch("nhp.docker.__main__.datetime")
    m_datetime.now.side_effect = [m_start_time, m_end_time]

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

    ru_m = mocker.patch("nhp.docker.__main__.run_all", return_value=("results", "variants"))

    expected_additional_metadata = {
        "model_run_start_time": m_start_time.isoformat(),
        "model_run_end_time": m_end_time.isoformat(),
        "model_run_elapsed_time_seconds": 2.0,
    }

    # act
    main(Mock())

    # assert
    rwls.assert_called_once_with("params.json")
    rwas.assert_not_called()

    s = rwls()
    ru_m.assert_called_once_with(params, "data", s.progress_callback(), False)
    s.finish.assert_called_once_with("results", "variants", False, expected_additional_metadata)

    local_data_mock.create.assert_called_once_with("data")


@pytest.mark.unit
def test_main_azure(mocker):
    # arrange
    m = mocker.patch("nhp.docker.__main__.parse_args")
    m().params_file = "params.json"
    m().model_run_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    m().local_storage = False
    m().save_full_model_results = False

    m_start_time = datetime(2025, 1, 1, 12, 0, 0)
    m_end_time = datetime(2025, 1, 1, 12, 0, 2)
    m_datetime = mocker.patch("nhp.docker.__main__.datetime")
    m_datetime.now.side_effect = [m_start_time, m_end_time]

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

    ru_m = mocker.patch("nhp.docker.__main__.run_all", return_value=("results", "variants"))

    expected_additional_metadata = {
        "model_run_start_time": m_start_time.isoformat(),
        "model_run_end_time": m_end_time.isoformat(),
        "model_run_elapsed_time_seconds": 2.0,
    }

    # act
    main(config)

    # assert
    rwls.assert_not_called()
    rwas.assert_called_once_with(
        uuid.UUID("00000000-0000-0000-0000-000000000001"), "params.json", config
    )

    s = rwas()
    ru_m.assert_called_once_with(params, "data", s.progress_callback(), False)
    s.finish.assert_called_once_with("results", "variants", False, expected_additional_metadata)

    local_data_mock.create.assert_called_once_with("data")


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "local_storage",
    [
        True,
        False,
    ],
)
def test_main_calls_runner_error_on_exception(mocker, local_storage):
    # arrange
    parse_args_mock = mocker.patch("nhp.docker.__main__.parse_args")
    parse_args_mock().params_file = "params.json"
    parse_args_mock().model_run_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    parse_args_mock().local_storage = local_storage
    parse_args_mock().save_full_model_results = False

    local_runner = mocker.patch("nhp.docker.__main__.RunWithLocalStorage")
    azure_runner = mocker.patch("nhp.docker.__main__.RunWithAzureStorage")
    runner = local_runner if local_storage else azure_runner
    runner().params = {
        "model_runs": 256,
        "start_year": 2019,
        "end_year": 2035,
        "app_version": "dev",
    }

    mocker.patch("nhp.docker.__main__.Local")
    mocker.patch("nhp.docker.__main__.run_all", side_effect=Exception("Test error"))
    log_error = mocker.patch("logging.error")
    config = Mock()

    # act
    main(config)

    # assert
    if local_storage:
        runner.assert_any_call("params.json")
    else:
        runner.assert_any_call(
            uuid.UUID("00000000-0000-0000-0000-000000000001"), "params.json", config
        )
    log_error.assert_called_once_with("An error occurred: %s", "Test error")
    runner().error.assert_called_once_with("Test error")
    if local_storage:
        azure_runner.assert_not_called()
    else:
        local_runner.assert_not_called()
