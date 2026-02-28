from __future__ import annotations

import logging
import sys


def get_logger(name: str = "model_selection_kfp", level: str = "INFO") -> logging.Logger:
    """
    Standard logger factory.
    - Console output
    - Consistent formatting
    - No duplicate handlers
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    logger.propagate = False

    return logger