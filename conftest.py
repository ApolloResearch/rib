"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--runmpi", action="store_true", default=False, help="run mpi test")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "mpi: mark test to be run with MPI")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    run_mpi = config.getoption("--runmpi")

    if run_mpi:
        if len(items) > 1 or "mpi" not in items[0].keywords:
            pytest.exit("--runmpi can only be used with a single test marked with @pytest.mark.mpi")
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_mpi = pytest.mark.skip(reason="need --runmpi option to run")

    for item in items:
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "mpi" in item.keywords:
            item.add_marker(skip_mpi)
