"""test hsa gams"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

import os
import sys
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.hsa_gams import (
    _create_activity_type_gams,
    _generate_activity_table,
    create_gams,
    init,
    main,
    run,
)


def test_create_activity_type_gams(mocker):
    """it creates the gams for the given activity type"""
    # arrange
    data = pd.DataFrame(
        {
            "age": [17, 18, 50, 90, 91] * 6,
            "sex": [1] * 15 + [2] * 15,
            "hsagrp": (["a"] * 5 + ["b"] * 5 + ["c"] * 5) * 2,
        }
    )
    mocker.patch("pandas.read_parquet", return_value=data)

    gam_mock = Mock()
    mocker.patch("model.hsa_gams.GAM", return_value=gam_mock)
    gam_mock.gridsearch.return_value = "GAM"

    pop = pd.DataFrame(
        {
            "age": list(range(100)) * 2,
            "sex": [1] * 100 + [2] * 100,
            "base_year": list(range(1, 201)),
        }
    )
    path_fn = Mock()
    data = {}
    gams = {}

    # act
    _create_activity_type_gams(pop, path_fn, data, gams, "test", ["c"])

    # assert
    assert data["test"].activity_rate.to_list() == [
        1 / (i + 1 + j) if i in [18, 50, 90] else 0
        for i in range(18, 91)
        for j in [0, 0, 100, 100]
    ]
    assert gams == {(h, s): "GAM" for h in ["a", "b"] for s in [1, 2]}


def test_create_activity_type_gams_no_ignored_hsagrps(mocker):
    """it creates the gams for the given activity type"""
    # arrange
    data = pd.DataFrame(
        {
            "age": [17, 18, 50, 90, 91] * 6,
            "sex": [1] * 15 + [2] * 15,
            "hsagrp": (["a"] * 5 + ["b"] * 5 + ["c"] * 5) * 2,
        }
    )
    mocker.patch("pandas.read_parquet", return_value=data)

    gam_mock = Mock()
    mocker.patch("model.hsa_gams.GAM", return_value=gam_mock)
    gam_mock.gridsearch.return_value = "GAM"

    pop = pd.DataFrame(
        {
            "age": list(range(100)) * 2,
            "sex": [1] * 100 + [2] * 100,
            "base_year": list(range(1, 201)),
        }
    )
    path_fn = Mock()
    data = {}
    gams = {}

    # act
    _create_activity_type_gams(pop, path_fn, data, gams, "test")

    # assert
    assert data["test"].activity_rate.to_list() == [
        1 / (i + 1 + j) if i in [18, 50, 90] else 0
        for i in range(18, 91)
        for j in [0, 0, 0, 100, 100, 100]
    ]
    assert gams == {(h, s): "GAM" for h in ["a", "b", "c"] for s in [1, 2]}


def test_create_gams(mocker):
    """it creates all of the gams"""
    # arrange
    read_csv_mock = mocker.patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            {
                "age": [89, 90, 91] * 2,
                "sex": [1] * 6,
                "2020": [1, 2, 3] * 2,
                "variant": ["principal_proj"] * 3 + ["other"] * 3,
            }
        ),
    )
    catg_mock = mocker.patch("model.hsa_gams._create_activity_type_gams")
    mocker.patch("pandas.concat", return_value="data")
    pop_expected = pd.DataFrame({"sex": [1, 1], "age": [89, 90], "base_year": [1, 5]})
    gat_mock = mocker.patch("model.hsa_gams._generate_activity_table")

    # act
    data, gams, path_fn = create_gams("test", "2020")

    # assert
    read_csv_mock.assert_called_once_with(
        os.path.join("data", "2020", "test", "demographic_factors.csv")
    )
    assert data == "data"
    assert not gams
    assert path_fn("file") == os.path.join("data", "2020", "test", "file")

    for i in catg_mock.call_args_list:
        assert i[0][0].equals(pop_expected)
        assert i[0][1] == path_fn
        assert i[0][2] == {}
        assert i[0][3] == {}

    catg_mock.call_args_list[0][0][4] == "ip"
    catg_mock.call_args_list[0][0][5] == ["birth", "maternity", "paeds"]
    catg_mock.call_args_list[1][0][4] == "op"
    catg_mock.call_args_list[1][0][5] == None
    catg_mock.call_args_list[2][0][4] == "aae"
    catg_mock.call_args_list[2][0][5] == None

    gat_mock.assert_called_once_with(
        os.path.join("data", "2020", "test", "hsa_activity_table.csv"), {}
    )


def test_hsa_gam_generate_activity_table(tmp_path):
    # arrange
    hsa_mock = type("mocked_hsa", (object,), {"predict": lambda x: x})
    gams = {(h, s): hsa_mock for h in ["a", "b"] for s in [1, 2]}
    filename = f"{tmp_path}/hsa_activity_table.csv"

    # act
    _generate_activity_table(filename, gams)
    actual = pd.read_csv(filename)

    # assert
    assert actual.to_dict("list") == {
        "hsagrp": ["a"] * 202 + ["b"] * 202,
        "sex": ([1] * 101 + [2] * 101) * 2,
        "age": list(range(0, 101)) * 4,
        "activity": list(range(0, 101)) * 4,
    }


def test_run(mocker):
    """it should create the gams and save them"""
    path_fn = lambda x: x
    mocker.patch("model.hsa_gams.create_gams", return_value=("data", "gams", path_fn))
    pickle_dump_mock = mocker.patch("pickle.dump")
    with patch("builtins.open", mock_open()) as mock_file:
        run("test", "2020")
        assert pickle_dump_mock.call_args_list[0][0][0] == "gams"
        mock_file.assert_called_with("hsa_gams.pkl", "wb")


@pytest.mark.parametrize("passed_args", [[], [1], [1, 2, 3]])
def test_main_invalid_args(passed_args):
    """the main method should raise an error if incorrect arguments are provided"""
    with patch.object(sys, "argv", ["filename.py"] + passed_args):
        with pytest.raises(Exception):
            main()


def test_main_valid_args(mocker):
    """the main method should call run if correct arguments are provided"""
    with patch.object(sys, "argv", ["filename.py", "test", "2020"]):
        run_mock = mocker.patch("model.hsa_gams.run", return_value="run")
        main()
        run_mock.assert_called_once_with("test", "2020")


def test_init(mocker):
    """it should run the main method if __name__ is __main__"""
    import model.hsa_gams as h

    main_mock = mocker.patch("model.hsa_gams.main")

    init()  # should't call main
    main_mock.assert_not_called()

    with patch.object(h, "__name__", "__main__"):
        init()  # should call main
        main_mock.assert_called_once()
