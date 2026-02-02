import logging
import sys
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Args:
        name: Name of the logger, usually __name__.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(f"{datetime.now()}-training.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
