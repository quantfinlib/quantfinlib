"""Utility module to configure the logger."""

import logging
import logging.config
from typing import Optional

import yaml

from quantfinlib.util._fs_utils import get_project_root

logger: Optional[logging.Logger] = None


def configure_logger():
    """Configure the logger using the base.yaml configuration file.

    Returns
    -------
    logging.Logger
        The logger object with the configuration settings.
    """
    global logger
    if logger is None:
        LOGGER_CONFIG_FILE = get_project_root("config/logger/base.yaml")
        try:
            with open(LOGGER_CONFIG_FILE, "r") as stream:
                config = yaml.safe_load(stream)
            logging.config.dictConfig(config)
            logger = logging.getLogger("main")
            logger.info("Logger configured successfully")
        except FileNotFoundError | yaml.YAMLError as e:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("main")
            logger.warning(
                f"Configuration file not found or couldn't be parsed properly: {e}. Using basic configuration."
            )


def get_logger() -> logging.Logger:
    """Return a configured logger object.
    Ensures that the logger is configured only once.

    Returns
    -------
    logging.Logger
        The logger object with the configuration settings.
    """
    global logger
    if logger is None:
        configure_logger()
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Logger test 1,2,3...")
