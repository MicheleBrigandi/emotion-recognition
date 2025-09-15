# Michele Brigandì - 2156373

import logging
import sys
from config.config import Config
from logging.handlers import RotatingFileHandler
from pathlib import Path

_config = Config()

def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger configured according to the global Config.

    The logger is set up with:
        - Log level from the configuration (default: INFO).
        - Console handler if enabled in the config.
        - Rotating file handler if enabled in the config.

    Args:
        name (str): Name of the logger (usually `__name__` of the module).

    Returns:
        logging.Logger: A configured logger instance.

    Notes:
        - If the logger already has handlers, it is returned as is to avoid duplicate logging.
        - File logs use DEBUG level to capture more detailed information than the console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, _config.logger_level.upper(), logging.INFO))

    # Avoid adding duplicate handlers if logger already configured
    if logger.hasHandlers():
        return logger

    # Log message format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if _config.logger_enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, _config.logger_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Rotating file handler
    if _config.logger_enable_file:
        log_path = Path(_config.logger_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_config.logger_max_file_size_mb * 1024 * 1024,
            backupCount=_config.logger_backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
