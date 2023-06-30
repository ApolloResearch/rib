"""Setup logging that can be used by all modules in the package."""
import logging
from logging import Logger


def setup_logger() -> Logger:
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    return _logger


logger = setup_logger()
