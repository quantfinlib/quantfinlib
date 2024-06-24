import logging
import logging.config

import yaml

from quantfinlib.util._fs_utils import get_project_root

__version__ = "0.0.4"

# Configure logging
with open(get_project_root("config/logger/base.yaml"), "r") as stream:
    config = yaml.safe_load(stream)
logging.config.dictConfig(config)
logger = logging.getLogger("main")
logger.info("Logger configured successfully")
