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
    model_run._model = Mock()
    model_run._model._strategies = {"efficiencies": "efficiencies"}

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
    m.data = pd.DataFrame({"rn": [1, 2, 3, 4]})
    m.strategies = pd.DataFrame(
        {"strategy": ["a"] * 3 + ["b"] * 3}, index=[1, 2, 3] * 2
    )

    # act
    m._select_single_strategy()

    # assert
    assert m.data.index.to_list() == ["b", "b", "a", "NULL"]


def test_generate_losr_df(mocker):
    # arrange

    # act

    # assert
    assert False


def test_update(mocker):
    # arrange

    # act

    # assert
    assert False


def test_losr_all(mock_ipe):
    """test that it reduces the speldur column for 'all' types"""
    # arrange
    m = mock_ipe
    m.data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "a", "b"] * 3)
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
    m.data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "c", "d"] * 3)
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
    m.data = pd.DataFrame({"speldur": list(range(9))}, index=["x", "e", "f"] * 3)
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
    m = mock_ipe
    m.data = pd.DataFrame(
        {
            "speldur": list(range(12)) * 2,
            "classpat": (["1"] * 6 + ["2"] * 3 + ["1"] * 3) * 2,
            "bedday_rows": [True] * 12 + [False] * 12,
        },
        index=[x for x in ["x", "g", "h", "i"] for _ in range(3)] * 2,
    )
    # 1: rvr < ur0
    # 2: rvr < ur0 + ur1
    # 3: rvr >= ur0 + ur1
    m._model_run.rng.uniform.return_value = (
        [0.02, 0.02, 0.01] + [0.04, 0.03, 1.0] + [0.04, 0.03, 0.55]
    ) * 2
    m.step_counts = {}

    # act
    actual = m.losr_bads()

    # assert
    assert actual == m

    assert m._model_run.rng.uniform.call_args_list[0] == call(size=18)

    assert (
        m.data["speldur"].to_list()
        == [0, 1, 2, 3, 4, 5] + [0] * 6 + [0, 1, 2] + [0] * 9
    )
    assert {k: v.tolist() for k, v in m.step_counts.items()} == {
        ("efficiencies", "g"): [-3, -15],
        ("efficiencies", "h"): [0, -21],
        ("efficiencies", "i"): [-1, -31],
    }


def test_apply(mock_ipe):
    # arrange
    mock_ipe.data = pd.DataFrame({"a": [1, 2, 3]}, index=["a", "b", "c"])
    # act
    mock_ipe.apply()

    # assert
    assert mock_ipe._model_run.data.to_dict(orient="list") == {"a": [1, 2, 3]}
