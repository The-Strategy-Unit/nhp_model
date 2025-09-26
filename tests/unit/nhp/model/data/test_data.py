"""test nhp data (local)."""

import pytest

from nhp.model.data import Data


def test_get_ip():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_ip()


def test_get_ip_strategies():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_ip_strategies()


def test_get_op():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_op()


def test_get_aae():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_aae()


def test_get_birth_factors():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_birth_factors()


def test_get_demographic_factors():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_demographic_factors()


def test_get_hsa_activity_table():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_hsa_activity_table()


def test_get_hsa_gams():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_hsa_gams()


def test_get_inequalities():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_inequalities()
