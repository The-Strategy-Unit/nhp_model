import pytest


def pytest_addoption(parser):
    parser.addoption("--data-dir", help="Directory containing data", default="data/synth")


@pytest.fixture
def data_dir(request):
    return request.config.getoption("--data-dir")
