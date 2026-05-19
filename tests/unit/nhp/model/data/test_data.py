"""test nhp data (local)."""

import pytest

from nhp.model.data import Data


@pytest.mark.unit
def test_get_ip():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_ip()


@pytest.mark.unit
def test_get_ip_strategies():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_ip_strategies()


@pytest.mark.unit
def test_get_op():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_op()


@pytest.mark.unit
def test_get_aae():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_aae()


@pytest.mark.unit
def test_get_birth_factors():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_birth_factors()


@pytest.mark.unit
def test_get_demographic_factors():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_demographic_factors()


@pytest.mark.unit
def test_get_hsa_activity_table():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_hsa_activity_table()


@pytest.mark.unit
def test_get_hsa_gams():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_hsa_gams()


@pytest.mark.unit
def test_get_inequalities():
    d = Data()
    with pytest.raises(NotImplementedError):
        d.get_inequalities()
