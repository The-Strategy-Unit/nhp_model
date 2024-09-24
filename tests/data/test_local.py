"""test nhp data (local)"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring

from unittest.mock import call, mock_open, patch

from model.data import Local


def test_init_sets_values():
    # arrange

    # act
    d = Local("data", 2019, "synthetic")

    # assert
    assert d._data_path == "data/2019/synthetic"


def test_create_returns_lambda():
    # arrange

    # act
    d = Local.create("data")(2019, "synthetic")

    # assert
    assert d._data_path == "data/2019/synthetic"


def test_get_ip(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_ip()

    # assert
    assert actual == "data"
    m.assert_called_once_with("ip.parquet")


def test_get_ip_strategies(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_ip_strategies()

    # assert
    assert actual == {"activity_avoidance": "data", "efficiencies": "data"}
    assert m.call_count == 2
    assert list(m.call_args_list) == [
        call("ip_activity_avoidance_strategies.parquet"),
        call("ip_efficiencies_strategies.parquet"),
    ]


def test_get_op(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_op()

    # assert
    assert actual == "data"
    m.assert_called_once_with("op.parquet")


def test_get_aae(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_aae()

    # assert
    assert actual == "data"
    m.assert_called_once_with("aae.parquet")


def test_get_birth_factors(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_csv", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_birth_factors()

    # assert
    assert actual == "data"
    m.assert_called_once_with("birth_factors.csv")


def test_get_demographic_factors(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_csv", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_demographic_factors()

    # assert
    assert actual == "data"
    m.assert_called_once_with("demographic_factors.csv")


def test_get_hsa_activity_table(mocker):
    # arrange
    m = mocker.patch("model.data.Local._get_csv", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d.get_hsa_activity_table()

    # assert
    assert actual == "data"
    m.assert_called_once_with("hsa_activity_table.csv")


def test_get_hsa_gams(mocker):
    # arrange
    m = mocker.patch("pickle.load", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    with patch("builtins.open", mock_open(read_data="hsa_gams")) as mock_file:
        actual = d.get_hsa_gams()

    # assert
    assert actual == "data"
    mock_file.assert_called_with("data/2019/synthetic/hsa_gams.pkl", "rb")
    m.assert_called_once_with(mock_file())


def test_get_parquet(mocker):
    # arrange
    m = mocker.patch("pandas.read_parquet", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d._get_parquet("file")

    # assert
    assert actual == "data"
    m.assert_called_once_with("data/2019/synthetic/file")


def test_get_csv(mocker):
    # arrange
    m = mocker.patch("pandas.read_csv", return_value="data")
    d = Local("data", 2019, "synthetic")

    # act
    actual = d._get_csv("file")

    # assert
    assert actual == "data"
    m.assert_called_once_with("data/2019/synthetic/file")
