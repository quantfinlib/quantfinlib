import logging
import os
import tempfile

import pytest

from quantfinlib.util._logging import configure_logger, logger


@pytest.fixture(scope="function")
def logger():
    return logging.getLogger("quantfinlib")


def test_default_configuration(logger):
    configure_logger()
    assert logger.level == logging.WARNING


def test_custom_configuration(logger):
    configure_logger(verbosity=logging.DEBUG)
    assert logger.level == logging.DEBUG
    configure_logger(verbosity=logging.INFO)
    assert logger.level == logging.INFO


def test_console_logging(caplog, logger):
    configure_logger(verbosity=logging.DEBUG)
    logger.propagate = True

    with caplog.at_level(logging.DEBUG, logger="main"):
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")

    # Retrieve the captured log records
    log_records = caplog.record_tuples
    assert len(log_records) == 4 + 1  # one log message inside the configure_logger()

    log_text = caplog.text
    assert "This is an info message" in log_text
    assert "This is a warning message" in log_text
    assert "This is an error message" in log_text


def test_file_logging(logger):
    configure_logger(log_to_file=True)

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
        log_file_path = temp_file.name
        temp_file.close()

        try:
            configure_logger(log_to_file=True, log_file_path=log_file_path)
            logger.warning("Test warning message")
            logger.error("Test error message")
            logger.info("Test info message")
        finally:

            with open(log_file_path, "r") as file:
                log_file_content = file.read()
                assert "Test warning message" in log_file_content
                assert "Test error message" in log_file_content
                assert "Test info message" not in log_file_content

            # Close the file handle and delete the temporary file
            temp_file.close()


if __name__ == "__main__":
    pytest.main([__file__])
