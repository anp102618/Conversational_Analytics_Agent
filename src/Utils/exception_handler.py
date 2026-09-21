import logging
import traceback
from typing import Optional


def error_message_detail(error: Exception) -> str:
    """Return detailed information about the original exception."""
    tb = error.__traceback__
    if not tb:
        return f"Error: {error}"
    
    last_trace = traceback.extract_tb(tb)[-1]
    return (
        f"Error in [{last_trace.filename}] "
        f"function [{last_trace.name}] "
        f"at line [{last_trace.lineno}] "
        f"→ Message: {error}"
    )


class CustomException(Exception):
    """Application-level exception wrapper."""

    def __init__(self, error: Exception, logger: Optional[logging.Logger] = None) -> None:
        self.original_error = error
        self.logger = logger
        self.detailed_error = error_message_detail(error)
        super().__init__(self.detailed_error)

    def __str__(self) -> str:
        return self.detailed_error