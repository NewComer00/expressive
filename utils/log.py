import sys
import logging
from typing import TextIO


def _sanitize_line(s: str) -> str:
    # Handle carriage returns: keep only the last overwrite
    if '\r' in s:
        s = s.split('\r')[-1]
    # Simulate backspaces
    result = []
    for c in s:
        if c == '\b':
            if result:
                result.pop()
        else:
            result.append(c)
    return ''.join(result).strip()


class LoggedStream(TextIO):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        for line in message.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.level, line)


class TeeStream(TextIO):
    def __init__(self, original, logger, level):
        self.original = original
        self.logger = logger
        self.level = level

    def write(self, message):
        self.original.write(message)
        self.original.flush()
        for line in message.rstrip().splitlines():
            sanitized = _sanitize_line(line)
            if sanitized:
                self.logger.log(self.level, sanitized)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return self.original.isatty()


class StreamToLogger:
    def __init__(
        self, logger: logging.Logger | logging.LoggerAdapter,
        level_stdout: int = logging.INFO,
        level_stderr: int = logging.ERROR,
        tee: bool = False,
    ):
        self.logger = logger
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        if tee:
            self._stdout_redirect = TeeStream(self._stdout, logger, level_stdout)
            self._stderr_redirect = TeeStream(self._stderr, logger, level_stderr)
        else:
            self._stdout_redirect = LoggedStream(logger, level_stdout)
            self._stderr_redirect = LoggedStream(logger, level_stderr)

    def __enter__(self):
        sys.stdout = self._stdout_redirect
        sys.stderr = self._stderr_redirect
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


if __name__ == "__main__":
    import time
    import tempfile

    logger = logging.getLogger("example")
    logger.setLevel(logging.DEBUG)

    tempfile = tempfile.NamedTemporaryFile(delete=False)
    handler = logging.FileHandler(tempfile.name)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set tee Flag
    tee = True

    # Test: Redirect stdout and stderr to logger with
    with StreamToLogger(logger, tee=tee):
        print("This is a test message.")
        print("This is an error message.", file=sys.stderr)

    # Test: Progress bar with logger
    def progress_bar(progress, total, bar_length=40):
        percent = progress / total
        arrow = '#' * int(percent * bar_length)
        spaces = '-' * (bar_length - len(arrow))
        sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent * 100)}%")
        sys.stdout.flush()

    with StreamToLogger(logger, tee=tee):
        for i in range(101):
            progress_bar(i, 100)
            time.sleep(0.01)
        print("\nDone.")

    with open(tempfile.name, 'r') as f:
        print(f.read())
        
    tempfile.close()
