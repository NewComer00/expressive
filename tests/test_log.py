"""Tests for logging utilities."""

import sys
import logging
from io import StringIO
from unittest.mock import MagicMock

from utils.log import LoggedStream, StreamToLogger, TeeStream, _sanitize_line


class TestSanitizeLine:
    """Test line sanitization function."""

    def test_sanitize_line_basic(self):
        """Test basic line sanitization."""
        result = _sanitize_line("Hello World")
        assert result == "Hello World"

    def test_sanitize_line_with_carriage_return(self):
        """Test handling carriage returns (keeps last overwrite)."""
        result = _sanitize_line("First\rSecond\rThird")
        assert result == "Third"

    def test_sanitize_line_with_backspace(self):
        """Test handling backspaces."""
        result = _sanitize_line("Hello\b\b\blo")
        # Backspaces remove characters: "Hello" -> "He" (removed 3) -> "Helo" (added "lo")
        assert result == "Helo"

    def test_sanitize_line_with_whitespace(self):
        """Test stripping whitespace."""
        result = _sanitize_line("  Hello World  ")
        assert result == "Hello World"

    def test_sanitize_line_empty(self):
        """Test empty string."""
        result = _sanitize_line("")
        assert result == ""

    def test_sanitize_line_progress_bar(self):
        """Test progress bar simulation with carriage returns."""
        result = _sanitize_line("\r[###-----] 30%\r[######--] 60%\r[########] 100%")
        assert result == "[########] 100%"


class TestLoggedStream:
    """Test LoggedStream class."""

    def test_logged_stream_creation(self):
        """Test creating a LoggedStream."""
        logger = logging.getLogger("test")
        stream = LoggedStream(logger, logging.INFO)
        assert stream.logger == logger
        assert stream.level == logging.INFO

    def test_logged_stream_write_single_line(self):
        """Test writing a single line."""
        logger = MagicMock()
        stream = LoggedStream(logger, logging.INFO)

        stream.write("Test message\n")

        logger.log.assert_called_once_with(logging.INFO, "Test message")

    def test_logged_stream_write_multiple_lines(self):
        """Test writing multiple lines."""
        logger = MagicMock()
        stream = LoggedStream(logger, logging.INFO)

        stream.write("Line 1\nLine 2\nLine 3\n")

        assert logger.log.call_count == 3
        logger.log.assert_any_call(logging.INFO, "Line 1")
        logger.log.assert_any_call(logging.INFO, "Line 2")
        logger.log.assert_any_call(logging.INFO, "Line 3")

    def test_logged_stream_ignores_empty_lines(self):
        """Test that empty lines are ignored."""
        logger = MagicMock()
        stream = LoggedStream(logger, logging.INFO)

        stream.write("\n\n  \n")

        logger.log.assert_not_called()


class TestTeeStream:
    """Test TeeStream class."""

    def test_tee_stream_creation(self):
        """Test creating a TeeStream."""
        original = StringIO()
        logger = logging.getLogger("test")
        stream = TeeStream(original, logger, logging.INFO)

        assert stream.original == original
        assert stream.logger == logger
        assert stream.level == logging.INFO

    def test_tee_stream_write_to_both(self):
        """Test that TeeStream writes to both original and logger."""
        original = StringIO()
        logger = MagicMock()
        stream = TeeStream(original, logger, logging.INFO)

        stream.write("Test message\n")

        # Check original stream
        assert original.getvalue() == "Test message\n"
        # Check logger
        logger.log.assert_called_once_with(logging.INFO, "Test message")

    def test_tee_stream_sanitizes_lines(self):
        """Test that TeeStream sanitizes lines before logging."""
        original = StringIO()
        logger = MagicMock()
        stream = TeeStream(original, logger, logging.INFO)

        stream.write("First\rSecond\n")

        # Original gets raw output
        assert original.getvalue() == "First\rSecond\n"
        # Logger gets sanitized
        logger.log.assert_called_once_with(logging.INFO, "Second")

    def test_tee_stream_flush(self):
        """Test TeeStream flush method."""
        original = MagicMock()
        logger = MagicMock()
        stream = TeeStream(original, logger, logging.INFO)

        stream.flush()

        original.flush.assert_called_once()

    def test_tee_stream_isatty(self):
        """Test TeeStream isatty method."""
        original = MagicMock()
        original.isatty.return_value = True
        logger = MagicMock()
        stream = TeeStream(original, logger, logging.INFO)

        assert stream.isatty() is True
        original.isatty.assert_called_once()


class TestStreamToLogger:
    """Test StreamToLogger context manager."""

    def test_stream_to_logger_creation(self):
        """Test creating StreamToLogger."""
        logger = logging.getLogger("test")
        redirector = StreamToLogger(logger)

        assert redirector.logger == logger
        assert redirector._stdout == sys.stdout
        assert redirector._stderr == sys.stderr

    def test_stream_to_logger_context_manager(self):
        """Test StreamToLogger as context manager."""
        logger = MagicMock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with StreamToLogger(logger):
            # Inside context, streams should be redirected
            assert sys.stdout != original_stdout
            assert sys.stderr != original_stderr

        # After context, streams should be restored
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

    def test_stream_to_logger_captures_stdout(self):
        """Test that stdout is captured and logged."""
        logger = MagicMock()

        with StreamToLogger(logger, level_stdout=logging.INFO):
            print("Test stdout message")

        logger.log.assert_called_with(logging.INFO, "Test stdout message")

    def test_stream_to_logger_captures_stderr(self):
        """Test that stderr is captured and logged."""
        logger = MagicMock()

        with StreamToLogger(logger, level_stderr=logging.ERROR):
            print("Test stderr message", file=sys.stderr)

        logger.log.assert_called_with(logging.ERROR, "Test stderr message")

    def test_stream_to_logger_tee_mode(self):
        """Test StreamToLogger in tee mode."""
        logger = MagicMock()
        original_stdout = sys.stdout

        # Capture actual stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with StreamToLogger(logger, tee=True):
                print("Tee message")

            # Should appear in both logger and original stdout
            logger.log.assert_called()
            assert "Tee message" in captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

    def test_stream_to_logger_different_levels(self):
        """Test StreamToLogger with different log levels."""
        logger = MagicMock()

        with StreamToLogger(logger, level_stdout=logging.DEBUG, level_stderr=logging.CRITICAL):
            print("Debug message")
            print("Critical message", file=sys.stderr)

        # Check that correct levels were used
        calls = logger.log.call_args_list
        assert any(call[0][0] == logging.DEBUG for call in calls)
        assert any(call[0][0] == logging.CRITICAL for call in calls)

    def test_stream_to_logger_exception_handling(self):
        """Test that streams are restored even if exception occurs."""
        logger = MagicMock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            with StreamToLogger(logger):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Streams should still be restored
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr
