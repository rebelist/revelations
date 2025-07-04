from __future__ import annotations

from typing import Any

import loguru

from rebelist.revelations.domain.services import LoggerPort


class Logger(LoggerPort):
    """Wrapper around the Loguru logging with dependency injection."""

    def __init__(self, logger: loguru.Logger):
        self.__logger = logger.opt(depth=1)
        self.__logger.remove()
        self.__logger.add(
            'var/logs/app.log',
            rotation='10 MB',
            retention='10 days',
            compression='zip',
            format='{time:YYYY-MM-DD at HH:mm:ss} | {file.name}:{line} | {message}',
            level='INFO',
        )

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log information messages."""
        self.__logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning messages."""
        self.__logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error messages."""
        self.__logger.error(message, *args, **kwargs)
