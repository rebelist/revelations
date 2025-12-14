from unittest.mock import create_autospec

import loguru

from rebelist.revelations.infrastructure.logging import Logger


class TestLogger:
    """Test suite for the Logger adapter."""

    def test_logger_forwards_calls_to_underlying_loguru_logger(self) -> None:
        """Ensures Logger delegates info, warning, and error calls to Loguru."""
        base_logger = create_autospec(loguru.logger, instance=True)
        configured_logger = create_autospec(loguru.logger, instance=True)
        base_logger.opt.return_value = configured_logger

        logger = Logger(base_logger)

        logger.info('info message', 1)
        logger.warning('warning message', key='value')
        logger.error('error message')

        configured_logger.info.assert_called_once_with('info message', 1)
        configured_logger.warning.assert_called_once_with('warning message', key='value')
        configured_logger.error.assert_called_once_with('error message')
