"""Utility module to configure the logger."""

import logging
import logging.config
from typing import Optional

import yaml

from quantfinlib.util._fs_utils import get_project_root

# Global logger
logger = logging.getLogger("quantfinlib")
logger.setLevel(logging.WARN)


def configure_logger(verbosity=logging.WARNING, log_to_file=False, log_file_path="quantfinlib.log"):
    """Configure the logger using the base.yaml configuration file.

    Returns
    -------
    logging.Logger
        The logger object with the configuration settings.

    Example
    -------
    >>> configure_logger()
    """
    global logger

    logger.setLevel(verbosity)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Stream handler for console logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(verbosity)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler for file logging
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(verbosity)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logger configured successfully")


if __name__ == "__main__":
    configure_logger(verbosity=logging.WARN, log_to_file=False)
    logger.info("Logger INFO test 1,2,3...")
    logger.warning("Logger WARNING test 1,2,3...")
