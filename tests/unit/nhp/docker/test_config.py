import os
from unittest.mock import patch

import pytest

from nhp.docker.config import Config


def test_config_sets_values_from_envvars(mocker):
    # arrange
    mocker.patch("dotenv.load_dotenv")

    # act
    with patch.dict(
        os.environ,
        {
            "APP_VERSION": "app version",
            "DATA_VERSION": "data version",
            "STORAGE_ACCOUNT": "storage account",
            "CONTAINER_TIMEOUT_SECONDS": "123",
        },
    ):
        config = Config()

    # assert
    assert config.APP_VERSION == "app version"
    assert config.DATA_VERSION == "data version"
    assert config.STORAGE_ACCOUNT == "storage account"
    assert config.CONTAINER_TIMEOUT_SECONDS == 123


def test_config_uses_default_values(mocker):
    # arrange
    mocker.patch("dotenv.load_dotenv")

    # act
    config = Config()

    # assert
    assert config.APP_VERSION == "dev"
    assert config.DATA_VERSION == "dev"

    with pytest.raises(ValueError, match="STORAGE_ACCOUNT environment variable must be set"):
        config.STORAGE_ACCOUNT

    assert config.CONTAINER_TIMEOUT_SECONDS == 3600


def test_config_calls_dotenv_load(mocker):
    # arrange
    m = mocker.patch("dotenv.load_dotenv")

    # act
    config = Config()

    # assert
    m.assert_called_once()
