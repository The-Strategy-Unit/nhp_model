"""test inpatient efficiencies"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name

from datetime import datetime, timedelta
from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from model.inpatients import InpatientEfficiencies


@pytest.fixture
def mock_ipe():
    """create a mock Model instance"""
    with patch.object(InpatientEfficiencies, "__init__", lambda s, m: None):
        ipe = InpatientEfficiencies(None)
    ipe._model_run = Mock()
    ipe.losr = pd.DataFrame(
        {
            "type": [x for x in ["all", "aec", "pre-op"] for _ in [0, 1]]
            + ["day_procedures_daycase", "day_procedures_outpatients"],
            "pre-op_days": [pd.NA] * 4 + [1, 2] + [pd.NA] * 2,
            "losr_f": [1 - 1 / (2**x) for x in range(8)],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h"],
    )
    return ipe


def test_init(mocker):
    # arrange
    mocker.patch("model.inpatients.InpatientEfficiencies._select_single_strategy")
    mocker.patch("model.inpatients.InpatientEfficiencies._generate_losr_df")

    model_run = Mock()
    model_run.model_run = 0
    model_run.data = pd.DataFrame({"speldur": [1, 2, 3]})
    model_run.step_counts = "step_counts"
    model_run.model.strategies = {"efficiencies": "efficiencies"}

    # act
    actual = InpatientEfficiencies(model_run)

    # assert
    assert actual._model_run == model_run
    assert actual.data.equals(model_run.data)
    assert actual.step_counts == "step_counts"
    assert actual.strategies == "efficiencies"
    assert actual.speldur_before.to_list() == [1, 2, 3]

    actual._select_single_strategy.assert_called_once()
    actual._generate_losr_df.assert_called_once()


def test_select_single_strategy(mock_ipe):
    # arrange
    m = mock_ipe
    m._model_run.rng = np.random.default_rng(0)
    m._model_run.data = pd.DataFrame(
        {"rn": list(range(5)), "admimeth": ["0"] * 4 + ["3"]}
    )
    m._model_run.model.strategies = {
        "efficiencies": pd.DataFrame(
            {"strategy": ["a"] * 3 + ["b"] * 3}, index=[1, 2, 3] * 2
        )
    }
    m._model_run.params = {"efficiencies": {"ip": {"a": 2, "b": 3, "c": 4}}}

    # act
    m._select_single_strategy()

    # assert
    assert m._model_run.data.index.fillna("NULL").to_list() == [
        "NULL",
        "b",
        "b",
        "a",
        "NULL",
    ]


def test_generate_losr_df(mock_ipe):
    # arrange
    m = mock_ipe

    m._model_run.params = {
        "efficiencies": {
            "ip": {
                "a": {"type": "1", "interval": [1, 3]},
                "b": {"type": "1", "interval": [2, 4]},
                "c": {"type": "2", "other": 1, "interval": [3, 5]},
            }
        }
    }
    m._model_run.run_params = {"efficiencies": {"ip": {"a": 2, "b": 3, "c": 4}}}

    expected = {
        "type": ["1", "1", "2"],
        "interval": [[1, 3], [2, 4], [3, 5]],
        "other": [None, None, 1.0],
        "losr_f": [2, 3, 4],
    }

    # act
    m._generate_losr_df()
    actual = m.losr.to_dict(orient="list")
    actual["other"] = [None if np.isnan(i) else i for i in actual["other"]]

    # assert
    assert actual == expected


@pytest.mark.parametrize("losr_type", ["all", "aec", "pre-op"])
def test_losr_empty(mock_ipe, losr_type):
    """test that if no preop strategy provided losr functions return self"""
    # arrange
    m = mock_ipe
    m.losr = m.losr[m.losr.type != losr_type]
    m._model_run.data = pd.DataFrame(
        {"speldur": list(range(9))}, index=["x", "a", "b"] * 3
    )

    # act
    if losr_type == "all":
        actual = m.losr_all()
    if losr_type == "aec":
        actual = m.losr_aec()
    if losr_type == "pre-op":
        actual = m.losr_preop()

    # assert
    assert actual == m


def test_losr_all(mock_ipe):
    """test that it reduces the speldur column for 'all' types"""
    # arrange
    m = mock_ipe
    m._model_run.data = pd.DataFrame(
        {"speldur": list(range(9))}, index=["x", "a", "b"] * 3
    )
    m._model_run.rng.binomial.return_value = list(range(6))

    m._update = Mock(return_value="update")

    # act
    actual = m.losr_all()
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == m

    assert m.data["speldur"].to_list() == [0, 0, 3, 3, 1, 4, 6, 2, 5]

    assert binomial_call_args[0].to_list() == [1, 4, 7, 2, 5, 8]
    assert binomial_call_args[1].to_list() == [0, 0, 0, 0.5, 0.5, 0.5]


def test_losr_aec(mock_ipe):
    """test that it reduces the speldur column for 'aec' types"""
    # arrange
    m = mock_ipe
    m._model_run.data = pd.DataFrame(
        {"speldur": list(range(9))}, index=["x", "c", "d"] * 3
    )
    m._model_run.rng.binomial.return_value = [0, 0, 0, 1, 1, 1]

    # act
    actual = m.losr_aec()
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == m

    assert m.data["speldur"].to_list() == [0, 0, 2, 3, 0, 5, 6, 0, 8]

    assert binomial_call_args[0] == 1
    assert binomial_call_args[1].equals(m.losr.loc[["c"] * 3 + ["d"] * 3, "losr_f"])


def test_losr_preop(mock_ipe):
    """test that is reduces the speldur column for 'pre-op' types"""
    # arrange
    m = mock_ipe
    m._model_run.data = pd.DataFrame(
        {"speldur": list(range(9))}, index=["x", "e", "f"] * 3
    )
    m._model_run.rng.binomial.return_value = [0, 1, 0, 1, 0, 1]

    # act
    actual = m.losr_preop()
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == m

    assert m.data["speldur"].to_list() == [0, 1, 0, 3, 3, 5, 6, 7, 6]

    assert binomial_call_args[0] == 1
    assert binomial_call_args[1].equals(1 - m.losr.loc[["e"] * 3 + ["f"] * 3, "losr_f"])


@pytest.mark.parametrize(
    "day_procedures_type, expected_speldur, expected_classpat",
    [
        (
            "day_procedures_daycase",
            [0, 1, 2, 3, 0, 5, 6, 7, 8] * 2,
            (["1"] * 4 + ["2"] + ["1"] * 4) * 2,
        ),
        (
            "day_procedures_outpatients",
            [0, 1, 2, 3, 4, 5, 6, 0, 8] * 2,
            (["1"] * 7 + ["-1"] + ["1"]) * 2,
        ),
    ],
)
def test_losr_day_procedures(
    mock_ipe, day_procedures_type, expected_speldur, expected_classpat
):
    """test that it reduces the speldur column for 'day_procedures' types"""
    # arrange
    m = mock_ipe
    strats = ["day_procedures_usually_dc", "day_procedures_usually_op"]
    # replace the index
    i = m.losr.index[~m.losr.type.str.startswith("day_procedures_")].to_list() + strats
    m.losr.index = i

    m._model_run.data = pd.DataFrame(
        {
            "speldur": list(range(9)) * 2,
            "classpat": ["1"] * 18,
        },
        index=[x for x in ["x"] + strats for _ in range(3)] * 2,
    )
    m._model_run.rng.binomial.return_value = np.tile([1, 0, 1], 2)
    m.step_counts = {}

    # act
    actual = m.losr_day_procedures(day_procedures_type)

    # assert
    assert actual == m

    assert m._model_run.rng.binomial.call_args[0][0] == 1
    assert (
        m._model_run.rng.binomial.call_args[0][1]
        == m.losr[m.losr.type == day_procedures_type]["losr_f"].repeat(6)
    ).all()

    assert m._model_run.data["speldur"].to_list() == expected_speldur
    assert m._model_run.data["classpat"].to_list() == expected_classpat


def test_update_step_counts(mock_ipe):
    # arrange
    mock_ipe._model_run.data = pd.DataFrame(
        {
            "rn": ["1", "2", "3", "1"],
            "classpat": ["-1", "1", "1", "1"],
            "speldur": [1, 2, 3, 4],
        },
        index=["a", "b", "a", "a"],
    )
    mock_ipe.speldur_before = [3, 4, 5, 6]
    mock_ipe._model_run.step_counts = {}
    mock_ipe._model_run.model.data = pd.DataFrame(
        {"rn": ["1", "2", "3", "4"]}
    )

    # act
    actual = mock_ipe.update_step_counts()

    # assert
    assert actual == mock_ipe
    assert {k: v.tolist() for k, v in actual._model_run.step_counts.items()} == {
        ("efficiencies", "a"): [[-1.0, 0.0, 0.0, 0.0], [-5.0, 0.0, -2.0, 0.0]],
        ("efficiencies", "b"): [[0.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]],
    }
