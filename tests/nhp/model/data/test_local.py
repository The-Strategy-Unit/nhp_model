"""test nhp data (local)."""

from unittest.mock import call, mock_open, patch

import pandas as pd

from nhp.model.data import Local


def test_init_sets_values():
    # arrange

    # act
    d = Local("data", 2019, "synthetic")

    # assert
    assert d._data_path == "data"


def test_file_path():
    # arrange

    # act
    d = Local("data", 2019, "synthetic")

    # assert
    assert d._file_path("ip") == "data/ip/fyear=2019/dataset=synthetic"


def test_create_returns_lambda():
    # arrange

    # act
    d = Local.create("data")(2019, "synthetic")

    # assert
    assert d._data_path == "data"


def test_get_ip(mocker):
    # arrange
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_ip()

    # assert
    assert actual == "data"
    m.assert_called_once_with("ip")


def test_get_ip_strategies(mocker):
    # arrange
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_ip_strategies()

    # assert
    assert actual == {"activity_avoidance": "data", "efficiencies": "data"}
    assert m.call_count == 2
    assert list(m.call_args_list) == [
        call("ip_activity_avoidance_strategies"),
        call("ip_efficiencies_strategies"),
    ]


def test_get_op(mocker):
    # arrange
    op_data = pd.DataFrame({"col_1": [1, 2], "col_2": [3, 4], "index": [5, 6]}, index=[2, 1])
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value=op_data)
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_op()

    # assert
    assert actual.col_1.to_list() == [1, 2]
    assert actual.col_2.to_list() == [3, 4]
    assert actual.rn.to_list() == [5, 6]
    m.assert_called_once_with("op")


def test_get_aae(mocker):
    # arrange
    ae_data = pd.DataFrame({"col_1": [1, 2], "col_2": [3, 4], "index": [5, 6]}, index=[2, 1])
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value=ae_data)
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_aae()

    # assert
    assert actual.col_1.to_list() == [1, 2]
    assert actual.col_2.to_list() == [3, 4]
    assert actual.rn.to_list() == [5, 6]
    m.assert_called_once_with("aae")


def test_get_birth_factors(mocker):
    # arrange
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_birth_factors()

    # assert
    assert actual == "data"
    m.assert_called_once_with("birth_factors")


def test_get_demographic_factors(mocker):
    # arrange
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_demographic_factors()

    # assert
    assert actual == "data"
    m.assert_called_once_with("demographic_factors")


def test_get_hsa_activity_table(mocker):
    # arrange
    m = mocker.patch("nhp.model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_hsa_activity_table()

    # assert
    assert actual == "data"
    m.assert_called_once_with("hsa_activity_tables")


def test_get_hsa_gams(mocker):
    # arrange
    m = mocker.patch("pickle.load", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    with patch("builtins.open", mock_open(read_data="hsa_gams")) as mock_file:
        actual = d.get_hsa_gams()

    # assert
    assert actual == "data"
    mock_file.assert_called_with("data/hsa_gams.pkl", "rb")
    m.assert_called_once_with(mock_file())


def test_get_parquet(mocker):
    # arrange
    fp = mocker.patch("nhp.model.data.Local._file_path", return_value="file_path")
    m = mocker.patch("pandas.read_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d._get_parquet("file")

    # assert
    assert actual == "data"
    fp.assert_called_once_with("file")
    m.assert_called_once_with("file_path")
