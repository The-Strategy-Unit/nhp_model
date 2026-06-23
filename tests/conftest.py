import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--data-dir", help="Directory containing data", default="data/synth")


@pytest.fixture(scope="session")
def data_dir(request):
    return request.config.getoption("--data-dir")


# force tests to run on CPU to avoid JAX issues
os.environ.setdefault("JAX_PLATFORMS", "cpu")


TEST_TYPES = {"unit", "integration", "e2e"}


def pytest_collection_modifyitems(config, items):
    errors = []

    for item in items:
        found = TEST_TYPES.intersection(m.name for m in item.iter_markers())

        if len(found) == 0:
            errors.append(f"{item.nodeid}: missing marker")
        elif len(found) > 1:
            errors.append(f"{item.nodeid}: multiple test-type markers ({', '.join(sorted(found))})")

    if errors:
        raise pytest.UsageError("Test marker validation failed:\n\n" + "\n".join(errors))
