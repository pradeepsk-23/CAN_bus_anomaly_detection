"""
src/utils/logging.py
Centralised logging setup using loguru.
Call `setup_logger(cfg)` once at process start; every module then does:
    from loguru import logger
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "artifacts/logs", level: str = "INFO") -> None:
    """Configure loguru for console + rotating file output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()  # drop default handler

    # Console — coloured, concise
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # File — full detail, rotating 50 MB
    logger.add(
        Path(log_dir) / "run_{time:YYYYMMDD_HHmmss}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        rotation="50 MB",
        retention="14 days",
        compression="gz",
    )

    logger.info("Logger initialised — level={}", level)
