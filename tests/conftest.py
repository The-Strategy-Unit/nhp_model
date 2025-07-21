import pytest


def pytest_addoption(parser):
    parser.addoption("--data-dir", help="Directory containing data", default="data/dev")
    parser.addoption(
        "--params-file", help="Path to the parameters file", default="queue/params-sample.json"
    )


@pytest.fixture
def data_dir(request):
    return request.config.getoption("--data-dir")


@pytest.fixture
def params_path(request):
    return request.config.getoption("--params-file")
