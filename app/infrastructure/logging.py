import os
import sys
from pathlib import Path

from loguru import logger


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=log_format,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level=level,
        format=log_format,
        rotation=os.getenv("LOG_ROTATION", "10 MB"),
        retention=os.getenv("LOG_RETENTION", "7 days"),
        compression="zip",
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
