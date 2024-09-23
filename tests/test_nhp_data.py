"""test nhp data (local)"""

# pylint: disable=protected-access,redefined-outer-name,no-member,invalid-name,missing-function-docstring

import pytest

from model.nhp_data import NHPData


def test_get_ip():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_ip()


def test_get_ip_strategies():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_ip_strategies()


def test_get_op():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_op()


def test_get_aae():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_aae()


def test_get_birth_factors():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_birth_factors()


def test_get_demographic_factors():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_demographic_factors()


def test_get_hsa_activity_table():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_hsa_activity_table()


def test_get_hsa_gams():
    d = NHPData()
    with pytest.raises(NotImplementedError):
        d.get_hsa_gams()
