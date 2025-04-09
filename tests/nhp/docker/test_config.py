import importlib
import os
from unittest.mock import patch

import nhp.docker.config as c


def test_config_sets_values_from_envvars(mocker):
    # arrange
    mocker.patch("dotenv.load_dotenv")

    # act
    with patch.dict(
        os.environ,
        {
            "APP_VERSION": "app version",
            "DATA_VERSION": "data version",
            "CONTAINER_TIMEOUT_SECONDS": "123",
            "QUEUE_STORAGE_ACCOUNT_URL": "queue_url",
            "DATA_STORAGE_ACCOUNT_URL": "data_url",
            "RESULTS_STORAGE_ACCOUNT_URL": "results_url",
        },
    ):
        importlib.reload(c)

        # assert
        assert c.APP_VERSION == "app version"
        assert c.DATA_VERSION == "data version"
        assert c.CONTAINER_TIMEOUT_SECONDS == 123
        assert c.QUEUE_STORAGE_ACCOUNT_URL == "queue_url"
        assert c.DATA_STORAGE_ACCOUNT_URL == "data_url"
        assert c.RESULTS_STORAGE_ACCOUNT_URL == "results_url"


def test_config_uses_default_values(mocker):
    # arrange
    mocker.patch("dotenv.load_dotenv")

    # act
    importlib.reload(c)

    # assert
    assert c.APP_VERSION == "dev"
    assert c.DATA_VERSION == "dev"
    assert c.CONTAINER_TIMEOUT_SECONDS == 3600
    assert not c.QUEUE_STORAGE_ACCOUNT_URL
    assert not c.DATA_STORAGE_ACCOUNT_URL
    assert not c.RESULTS_STORAGE_ACCOUNT_URL


def test_config_calls_dotenv_load(mocker):
    # arrange
    m = mocker.patch("dotenv.load_dotenv")

    # act
    importlib.reload(c)

    # assert
    m.assert_called_once()
