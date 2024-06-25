import logging
import logging.config
from pathlib import Path
from typing import Union

import yaml


def get_project_root(relative_filepath: Union[Path, str, None] = None) -> Path:
    """Returns the project's root directory: '../quantfinlib/'.

    Parameters:
    dependent_file_path (Union[Path, str, None], optional):
        additional filepath relative to the root, by default None

    Returns:
    Path: The project's root directory with optional dependent file path.
    """
    root_path = Path(__file__).parent.parent.parent
    if relative_filepath is not None:
        root_path = root_path / relative_filepath
    return root_path


def get_logger() -> logging.Logger:
    """Configures the logger using the base.yaml configuration file.

    Returns
    -------
    logging.Logger
        The logger object with the configuration settings.
    """
    with open(get_project_root("config/logger/base.yaml"), "r") as stream:
        config = yaml.safe_load(stream)
    logging.config.dictConfig(config)
    logger = logging.getLogger("main")
    logger.info("Logger configured successfully")
    return logger


if __name__ == "__main__":
    print(get_project_root())
    print(get_project_root().parent)
    print(get_project_root("quanfinlib"))
    print(get_project_root(Path("quanfinlib")))
