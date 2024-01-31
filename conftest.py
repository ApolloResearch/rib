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


def pytest_runtest_setup(item):
    if "mpi" in item.keywords:
        if item.config.getoption("--runmpi"):
            # Check that there is only 1 test and it is an mpi test
            if len(item.session.items) > 1 or "mpi" not in item.keywords:
                pytest.exit(
                    "--runmpi can only be used with a single test marked with @pytest.mark.mpi"
                )
        else:
            pytest.skip("need --runmpi option to run MPI tests")
    elif "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run slow tests")
