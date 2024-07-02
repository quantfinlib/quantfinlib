import logging
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from quantfinlib.util import logger_config

# Simplified mock configuration data
mock_log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "loggers": {
        "main": {
            "handlers": ["console"],
            "level": "INFO",
        },
    },
}


@pytest.fixture
def clean_logger():
    """Fixture to clean up the global logger before and after each test."""
    logger_config.logger = None
    yield
    logger_config.logger = None


@patch("builtins.open", new_callable=mock_open, read_data=yaml.safe_dump(mock_log_config))
@patch("logging.config.dictConfig")
@patch("logging.getLogger")
def test_configure_logger(mock_get_logger, mock_dictConfig, mock_open, clean_logger):
    mock_logger_instance = MagicMock()
    mock_logger_instance.name = "main"
    mock_get_logger.return_value = mock_logger_instance

    logger_config.configure_logger()

    mock_open.assert_called_once()
    mock_dictConfig.assert_called_once_with(mock_log_config)
    mock_get_logger.assert_called_with("main")
    assert logger_config.logger is not None
    assert logger_config.logger.name == "main"


@patch.object(logger_config, "configure_logger", wraps=logger_config.configure_logger)
def test_get_logger(mock_configure_logger, clean_logger):
    logger = logger_config.get_logger()

    mock_configure_logger.assert_called_once()
    assert logger is not None
    assert logger.name == "main"
    # Ensure subsequent calls do not reconfigure the logger
    second_logger = logger_config.get_logger()
    mock_configure_logger.assert_called_once()  # Still called only once
    assert logger is second_logger


if __name__ == "__main__":
    pytest.main([__file__])
