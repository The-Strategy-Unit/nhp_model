import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--data-dir", help="Directory containing data", default="data/synth")


@pytest.fixture(scope="session")
def data_dir(request):
    return request.config.getoption("--data-dir")


# force tests to run on CPU to avoid JAX issues
os.environ.setdefault("JAX_PLATFORMS", "cpu")
