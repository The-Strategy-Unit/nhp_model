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
            "QUEUE_STORAGE_ACCOUNT": "queue storage account",
            "DATA_STORAGE_ACCOUNT": "data storage account",
            "RESULTS_STORAGE_ACCOUNT": "results storage account",
            "FULL_MODEL_RESULTS_STORAGE_ACCOUNT": "full model results storage account",
            "MODEL_RUNS_TABLE_STORAGE_ACCOUNT": "model runs table account",
        },
    ):
        config = Config()

    # assert
    assert config.APP_VERSION == "app version"
    assert config.DATA_VERSION == "data version"
    assert config.QUEUE_STORAGE_ACCOUNT == "queue storage account"
    assert config.DATA_STORAGE_ACCOUNT == "data storage account"
    assert config.RESULTS_STORAGE_ACCOUNT == "results storage account"
    assert config.FULL_MODEL_RESULTS_STORAGE_ACCOUNT == "full model results storage account"
    assert config.MODEL_RUNS_TABLE_STORAGE_ACCOUNT == "model runs table account"


def test_config_uses_default_values(mocker):
    # arrange
    mocker.patch("dotenv.load_dotenv")

    # act
    config = Config()

    # assert
    assert config.APP_VERSION == "dev"
    assert config.DATA_VERSION == "dev"

    with pytest.raises(ValueError, match="QUEUE_STORAGE_ACCOUNT environment variable must be set"):
        config.QUEUE_STORAGE_ACCOUNT

    with pytest.raises(ValueError, match="DATA_STORAGE_ACCOUNT environment variable must be set"):
        config.DATA_STORAGE_ACCOUNT

    with pytest.raises(
        ValueError, match="RESULTS_STORAGE_ACCOUNT environment variable must be set"
    ):
        config.RESULTS_STORAGE_ACCOUNT

    with pytest.raises(
        ValueError, match="FULL_MODEL_RESULTS_STORAGE_ACCOUNT environment variable must be set"
    ):
        config.FULL_MODEL_RESULTS_STORAGE_ACCOUNT

    with pytest.raises(
        ValueError, match="MODEL_RUNS_TABLE_STORAGE_ACCOUNT environment variable must be set"
    ):
        config.MODEL_RUNS_TABLE_STORAGE_ACCOUNT


def test_config_calls_dotenv_load(mocker):
    # arrange
    m = mocker.patch("dotenv.load_dotenv")

    # act
    config = Config()

    # assert
    m.assert_called_once()
