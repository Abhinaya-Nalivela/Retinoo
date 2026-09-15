"""
Structured logging infrastructure for SIH 26038.
Provides standardized formatting, console colorization, and persistent audit logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "SIH26038",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Creates or retrieves a standardized logger.
    Logs are written to stdout and optionally to a persistent file in reports/logs/.
    """
    logger = logging.getLogger(name)

    # If handlers exist and no specific log_file requested, return existing
    if logger.handlers and log_file is None:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [LOGGER_NAME] - Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Persistent File Handler
    if log_file is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "reports" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "dr_screening.log")

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logging at {log_file}: {e}")

    return logger
