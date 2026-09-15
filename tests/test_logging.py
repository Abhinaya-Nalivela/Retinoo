"""
Unit tests for Structured Logging System (SIH 26038).
"""

import logging
import tempfile
import unittest
from pathlib import Path

from python.utils.logger import get_logger


class TestLoggingSystem(unittest.TestCase):
    """Verifies logger initialization, console formatting, and file audit logging."""

    def test_logger_creation(self):
        logger = get_logger("TestModuleInit")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)
        self.assertFalse(logger.propagate)
        self.assertGreaterEqual(len(logger.handlers), 1)

    def test_file_logging_in_custom_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_run.log"
            logger = get_logger(name="TempTestModule", log_file=str(log_path))
            test_msg = "SIH 26038 Audit Log Verification Message"
            logger.info(test_msg)

            # Flush and close handlers on Windows
            for h in list(logger.handlers):
                h.flush()
                h.close()
                logger.removeHandler(h)

            self.assertTrue(log_path.exists(), "Log file was not created")
            log_content = log_path.read_text(encoding="utf-8")
            self.assertIn(test_msg, log_content)
            self.assertIn("TempTestModule", log_content)
            self.assertIn("[INFO]", log_content)


if __name__ == "__main__":
    unittest.main()
