"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--runmpi", action="store_true", default=False, help="run mpi test")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "skip_ci: mark test to be skipped in CI")
    config.addinivalue_line("markers", "mpi: mark test to be run with MPI")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    if config.getoption("--runmpi"):
        # --runmpi given in cli: do not skip mpi tests
        # Note that you should only run one mpi test at a time
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_mpi = pytest.mark.skip(reason="need --runmpi option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "mpi" in item.keywords:
            item.add_marker(skip_mpi)
