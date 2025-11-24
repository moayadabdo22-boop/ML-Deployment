
import logging
from .config import LOG_PATH

def get_logger(name: str = "task3"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
