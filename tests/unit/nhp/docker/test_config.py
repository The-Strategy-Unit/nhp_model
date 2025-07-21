import importlib
from unittest.mock import patch

import nhp.docker.config as c


def test_config_sets_values_from_envvars(mocker):
    # arrange
    mocker.patch(
        "dotenv.dotenv_values",
        return_value={
            "APP_VERSION": "app version",
            "DATA_VERSION": "data version",
            "STORAGE_ACCOUNT": "storage account",
            "CONTAINER_TIMEOUT_SECONDS": "123",
        },
    )

    # act
    importlib.reload(c)

    # assert
    assert c.APP_VERSION == "app version"
    assert c.DATA_VERSION == "data version"
    assert c.STORAGE_ACCOUNT == "storage account"
    assert c.CONTAINER_TIMEOUT_SECONDS == 123


def test_config_uses_default_values(mocker):
    # arrange
    mocker.patch("dotenv.dotenv_values", return_value={})

    # act
    importlib.reload(c)

    # assert
    assert c.APP_VERSION == "dev"
    assert c.DATA_VERSION == "dev"
    assert not c.STORAGE_ACCOUNT
    assert c.CONTAINER_TIMEOUT_SECONDS == 3600


def test_config_calls_dotenv_load(mocker):
    # arrange
    m = mocker.patch("dotenv.dotenv_values", return_value={})

    # act
    importlib.reload(c)

    # assert
    m.assert_called_once()
