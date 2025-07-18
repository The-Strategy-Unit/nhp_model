import pytest


def pytest_addoption(parser):
    parser.addoption("--test-data-dir", help="Directory containing test data")


@pytest.fixture
def test_data_dir(request):
    return request.config.getoption("--test-data-dir")
