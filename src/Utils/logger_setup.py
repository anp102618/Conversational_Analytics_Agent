from datetime import datetime
from functools import wraps
import logging
from pathlib import Path
import sys
import time
import tracemalloc
from contextvars import ContextVar
from typing import Callable

# ======================================================
# SETUP
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

tracemalloc.start()

# Current logger for the active pipeline context
current_logger: ContextVar[logging.Logger | None] = ContextVar(
    "current_logger",
    default=None,
)

LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(filename)s:%(lineno)d] "
    "[%(funcName)s] "
    "%(message)s"
)


# ======================================================
# MEMORY HANDLER
# ======================================================

class MemoryHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.logs.append(self.format(record))
        except Exception:
            self.handleError(record)


# ======================================================
# LOGGER
# ======================================================

def setup_logger(
    log_id: str,
    log_dir: Path | str | None = None,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configures and returns a logger instance.
    
    Accepts an explicit log_file path or a log_dir folder path.
    """
    logger = logging.getLogger(f"app.{log_id}")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine file handler destination
    if log_file is not None:
        target_file = Path(log_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_file = target_dir / f"{log_id}_{timestamp}.log"

    file_handler = logging.FileHandler(
        target_file,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    memory_handler = MemoryHandler()
    memory_handler.setLevel(logging.INFO)
    memory_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(memory_handler)

    logger.memory_handler = memory_handler  # type: ignore[attr-defined]

    return logger


def get_log(
    name: str | None = None,
    log_dir: Path | str | None = None,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Return the logger for the current execution context or create a new one."""
    logger = current_logger.get()
    if logger is not None:
        return logger
    return setup_logger(name or "default", log_dir=log_dir, log_file=log_file)


def set_pipeline_logger(
    log_id: str,
    log_dir: Path | str | None = None,
    log_file: Path | str | None = None,
):
    """Set the logger for the current pipeline context."""
    logger = setup_logger(log_id, log_dir=log_dir, log_file=log_file)
    return current_logger.set(logger)


def reset_pipeline_logger(token) -> None:
    """Restore the previous logger context."""
    current_logger.reset(token)


# ======================================================
# MEMORY
# ======================================================

def format_memory(value: int) -> str:
    sign = "+" if value >= 0 else "-"
    value = abs(value)

    if value < 1024:
        return f"{sign}{value} B"
    if value < 1024 ** 2:
        return f"{sign}{value / 1024:.2f} KB"
    if value < 1024 ** 3:
        return f"{sign}{value / (1024 ** 2):.2f} MB"
    return f"{sign}{value / (1024 ** 3):.2f} GB"


# ======================================================
# PERFORMANCE
# ======================================================

def track_performance(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_log()
        start_time = time.perf_counter()
        start_current, start_peak = tracemalloc.get_traced_memory()

        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception(f"{func.__name__} failed", stacklevel=2)
            raise
        finally:
            end_time = time.perf_counter()
            end_current, end_peak = tracemalloc.get_traced_memory()

            logger.info(
                f"{func.__name__} completed | "
                f"Time: {end_time - start_time:.2f}s | "
                f"Memory: {format_memory(end_current - start_current)} | "
                f"Peak: {format_memory(end_peak - start_peak)}",
                stacklevel=2,
            )

    return wrapper