"""Minimal logging entry used by EB-Manipulation evaluator and planners."""
import logging

logger = logging.getLogger("EB_logger")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
