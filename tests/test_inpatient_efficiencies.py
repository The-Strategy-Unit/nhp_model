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
            + ["bads"] * 3,
            "baseline_target_rate": [pd.NA] * 6 + [0.25, 0.50, 0.75],
            "op_dc_split": [pd.NA] * 6 + [0, 1, 0.5],
            "pre-op_days": [pd.NA] * 4 + [1, 2] + [pd.NA] * 3,
            "losr_f": [1 - 1 / (2**x) for x in range(9)],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    )
    return ipe


def test_init(mocker):
    # arrange
    mocker.patch("model.inpatients.InpatientEfficiencies._select_single_strategy")
    mocker.patch("model.inpatients.InpatientEfficiencies._generate_losr_df")

    model_run = Mock()
    model_run.model_run = 0
    model_run.data = "data"
    model_run.step_counts = "step_counts"
    model_run.model.strategies = {"efficiencies": "efficiencies"}

    # act
    actual = InpatientEfficiencies(model_run)

    # assert
    actual._model_run == 0
    actual.data == "data"
    actual.step_counts == "step_counts"
    actual.strategies == "efficiencies"

    actual._select_single_strategy.assert_called_once()
    actual._generate_losr_df.assert_called_once()


def test_select_single_strategy(mock_ipe):
    # arrange
    m = mock_ipe
    m._model_run.rng = np.random.default_rng(0)
    m._model_run.data = pd.DataFrame({"rn": [1, 2, 3, 4]})
    m._model_run.model.strategies = {
        "efficiencies": pd.DataFrame(
            {"strategy": ["a"] * 3 + ["b"] * 3}, index=[1, 2, 3] * 2
        )
    }

    # act
    m._select_single_strategy()

    # assert
    assert m._model_run.data.index.to_list() == ["b", "b", "a", "NULL"]


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


def test_update(mock_ipe):
    # arrange
    m = mock_ipe
    m.step_counts = {}

    indexes = ["a", "b", "c"]
    ixl = len(indexes)

    m._model_run.data = pd.DataFrame(
        {"bedday_rows": [True, False] * 2 * ixl, "speldur": range(4 * ixl)},
        index=[i for i in indexes for _ in range(4)],
    )

    # act
    i = indexes[1:]
    m._update(i, range(4 * (ixl - 1)))

    # assert
    assert m._model_run.data["speldur"].tolist()[:4] == list(range(4))
    assert m._model_run.data["speldur"].tolist()[4:] == list(range(8))
    assert {k: v.tolist() for k, v in m.step_counts.items()} == {
        ("efficiencies", "b"): [0, -8],
        ("efficiencies", "c"): [0, -8],
    }


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
    update_call_args = m._update.call_args_list[0][0]
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == "update"

    assert update_call_args[0].to_list() == ["a", "b"]
    assert update_call_args[1] == [0, 1, 2, 3, 4, 5]

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

    m._update = Mock(return_value="update")

    # act
    actual = m.losr_aec()
    update_call_args = m._update.call_args_list[0][0]
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == "update"

    assert update_call_args[0].to_list() == ["c", "d"]
    assert update_call_args[1].to_list() == [0, 0, 0, 2, 5, 8]

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

    m._update = Mock(return_value="update")

    # act
    actual = m.losr_preop()
    update_call_args = m._update.call_args_list[0][0]
    binomial_call_args = m._model_run.rng.binomial.call_args_list[0][0]

    # assert
    assert actual == "update"

    assert update_call_args[0].to_list() == ["e", "f"]
    assert update_call_args[1].to_list() == [1, 3, 7, 0, 5, 6]

    assert binomial_call_args[0] == 1
    assert binomial_call_args[1].equals(1 - m.losr.loc[["e"] * 3 + ["f"] * 3, "losr_f"])


def test_losr_bads(mock_ipe):
    """test that it reduces the speldur column for 'bads' types"""
    # arrange
    m = mock_ipe
    bads_strats = ["bads_daycase", "bads_daycase_occassional", "bads_outpatients"]
    # replace the index
    i = m.losr.index[m.losr.type != "bads"].to_list() + bads_strats
    m.losr.index = i

    m._model_run.data = pd.DataFrame(
        {
            "speldur": list(range(12)) * 2,
            "classpat": (["1"] * 6 + ["2"] * 3 + ["1"] * 3) * 2,
            "bedday_rows": [True] * 12 + [False] * 12,
        },
        index=[x for x in ["x"] + bads_strats for _ in range(3)] * 2,
    )
    m._model_run.rng.binomial.return_value = np.tile([1, 1, 0], 3 * 2)
    m.step_counts = {}

    # act
    actual = m.losr_bads()

    # assert
    assert actual == m

    assert m._model_run.rng.binomial.call_args[0][0] == 1
    assert (
        m._model_run.rng.binomial.call_args[0][1]
        == m.losr[m.losr.type == "bads"]["losr_f"].repeat(6)
    ).all()

    assert (
        m._model_run.data["speldur"].to_list()
        == [0, 1, 2, 3, 4, 0, 6, 7, 0, 9, 10, 0] * 2
    )
    assert (
        m._model_run.data["classpat"].to_list()
        == (["1"] * 5 + ["2"] * 4 + ["1", "1", "-1"]) * 2
    )
    assert {k: v.tolist() for k, v in m.step_counts.items()} == {
        ("efficiencies", "bads_daycase"): [0, -5],
        ("efficiencies", "bads_daycase_occassional"): [0, -8],
        ("efficiencies", "bads_outpatients"): [-1, -12],
    }
