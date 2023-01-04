"""test InpatientsAdmissionsCounter"""


from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from model.inpatients_admissions_counter import InpatientsAdmissionsCounter


@pytest.fixture
def mock_iac():
    """create a mock InpatientsAdmissionsCounter instance"""
    rng = Mock()
    rng.poisson.return_value = 1
    rng.binomial.return_value = 0
    return InpatientsAdmissionsCounter(pd.DataFrame({"speldur": [1, 2, 3]}), rng)


def test_init():
    """test that the object is created as expected"""
    # arrange
    data = pd.DataFrame({"speldur": [1, 2, 3]})
    # act
    iac = InpatientsAdmissionsCounter(data, "rng")
    # assert
    assert iac._data.equals(data)
    assert iac._rng == "rng"
    assert iac._beddays.to_list() == [2, 3, 4]
    assert iac._admissions.tolist() == [1, 1, 1]
    assert iac.step_counts == {("baseline", "-"): {"admissions": 3, "beddays": 9}}


def test_update_step_counts_no_split_by(mock_iac):
    """test that the state is updated as expected"""
    # arrange
    new_admissions = np.array([2, 3, 4])
    # act
    actual = mock_iac._update_step_counts(new_admissions, "step")
    # assert
    assert actual == mock_iac
    assert np.array_equal(mock_iac._admissions, new_admissions)
    assert mock_iac.step_counts[("step", "-")] == {"admissions": 6, "beddays": 20}


def test_update_step_counts_with_split_by(mock_iac):
    """test that the state is updated as expected"""
    # arrange
    new_admissions = np.array([2, 3, 4])
    # act
    actual = mock_iac._update_step_counts(new_admissions, "step", ["a", "b", "b"])
    # assert
    assert actual == mock_iac
    assert np.array_equal(mock_iac._admissions, new_admissions)
    assert mock_iac.step_counts[("step", "a")] == {"admissions": 1, "beddays": 2}
    assert mock_iac.step_counts[("step", "b")] == {"admissions": 5, "beddays": 18}


def test_poisson_step(mock_iac):
    """test that it calls the _update_step_counts correctly"""
    # arrange
    factor = np.array([1, 2, 3])
    mock_iac._admissions = factor
    mock_iac._update_step_counts = Mock(return_value="update_step_counts")
    # act
    actual = mock_iac.poisson_step(factor, "step", "split")
    # assert
    assert actual == "update_step_counts"
    mock_iac._rng.poisson.assert_called_once()
    assert mock_iac._rng.poisson.call_args_list[0][0][0].tolist() == [1, 4, 9]
    mock_iac._update_step_counts.assert_called_once_with(1, "step", "split")


def test_modified_poisson_step(mock_iac):
    """test that it calls the _update_step_counts correctly"""
    # arrange
    factor = np.array([1, 2, 3])
    mock_iac._admissions = factor
    mock_iac._update_step_counts = Mock(return_value="update_step_counts")
    # act
    actual = mock_iac.modified_poisson_step(factor, "step", "split")
    # assert
    assert actual == "update_step_counts"
    mock_iac._rng.poisson.assert_called_once()
    assert mock_iac._rng.poisson.call_args_list[0][0][0].tolist() == [0, 1, 2]
    i = mock_iac._update_step_counts.call_args_list[0][0]
    assert i[0].tolist() == [2, 4, 6]
    assert i[1] == "step"
    assert i[2] == "split"


def test_binomial_step(mock_iac):
    """test that it calls the _update_step_counts correctly"""
    # arrange
    factor = np.array([1, 2, 3])
    mock_iac._update_step_counts = Mock(return_value="update_step_counts")
    # act
    actual = mock_iac.binomial_step(factor, "step", "split")
    # assert
    assert actual == "update_step_counts"
    mock_iac._rng.binomial.assert_called_once()
    assert mock_iac._rng.binomial.call_args_list[0][1]["n"].tolist() == [1, 1, 1]
    assert mock_iac._rng.binomial.call_args_list[0][1]["p"].tolist() == [1, 2, 3]
    mock_iac._update_step_counts.assert_called_once_with(0, "step", "split")


def test_get_data(mock_iac):
    """test that it returns the rows sampled by the admissions counter"""
    # arrange
    mock_iac._admissions = np.array([0, 2, 3])
    # act
    actual = mock_iac.get_data()
    # assert
    assert actual.speldur.to_list() == [2, 2, 3, 3, 3]
