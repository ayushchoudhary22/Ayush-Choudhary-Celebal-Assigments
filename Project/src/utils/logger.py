"""
Enterprise Logging Module for RetailSense AI Platform.
Configures rotating file logging and console output with formatted timestamps.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from pathlib import Path
from typing import Optional
from src.utils.config import get_setting


def setup_logger(
    name: str = "retailsense",
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Creates and returns a configured logger instance with rotating file and stream handlers.

    Parameters
    ----------
    name : str
        Logger name identifier (default: 'retailsense')
    log_file : Optional[str]
        Log file name inside logs directory (default: 'retailsense.log')
    level : Optional[str]
        Logging level ('INFO', 'DEBUG', 'WARNING', 'ERROR')

    Returns
    -------
    logging.Logger
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    log_level_str = level or get_setting("logging.level", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Base directory resolution
    base_dir = Path(__file__).resolve().parent.parent.parent
    logs_dir = base_dir / get_setting("paths.logs_dir", "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    filename = log_file or get_setting("logging.log_filename", "retailsense.log")
    file_path = logs_dir / filename

    # Standard formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console / Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # Timed Rotating File Handler (Rotates daily, retains 7 backups)
    file_handler = TimedRotatingFileHandler(
        filename=file_path,
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Utility helper to get module-specific logger instance."""
    return setup_logger(name=f"retailsense.{module_name}")
