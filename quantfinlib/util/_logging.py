"""Utility module to configure the logger."""

import logging
import logging.config

# Global logger
logger = logging.getLogger("quantfinlib")
logger.setLevel(logging.WARN)


def configure_logger(verbosity=logging.WARNING, log_to_file=False, log_file_path="quantfinlib.log"):
    """Configure the logger

    Example
    -------
    >>> configure_logger(verbosity=logging.INFO, log_to_file=False)
    >>> logger.info("Logger INFO test 1,2,3...")
    >>> logger.warning("Logger WARNING test 1,2,3...")
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
