import os
import atexit
import logging
import argparse
import tempfile

import librosa
import soundfile as sf

from utils.i18n import _


def timestamp2sec(value: str) -> float:
    """Parse a timestamp string in M:S format (e.g. '0:10.01') into seconds.

    Intended for use as ``type=timestamp2sec`` in
    :func:`argparse.ArgumentParser.add_argument`, so argparse stores the
    result directly as a ``float`` number of seconds.

    Args:
        value (str): The timestamp string to parse.

    Returns:
        float: Total time in seconds (e.g. '1:30.5' -> ``90.5``).

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid M:S timestamp.
    """
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp '{value}'. Expected M:S (e.g. '0:10.01')."
        )
    minutes_str, seconds_str = parts
    try:
        minutes = int(minutes_str)
        seconds = float(seconds_str)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp '{value}'. "
            "Minutes must be an integer and seconds must be a number (e.g. '0:10.01')."
        ) from err
    if minutes < 0:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp '{value}': minutes must be non-negative, got {minutes}."
        )
    if not (0 <= seconds < 60):
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp '{value}': seconds must be in [0, 60), got {seconds}."
        )
    return minutes * 60.0 + seconds


def validate_timestamp(value: str | None, arg_name: str) -> bool:
    """Validate a timestamp argument in M:S format (e.g. '0:10.01').

    Wraps :func:`timestamp2sec` for use outside argparse.
    Accepts ``None`` silently (meaning "use default boundary").

    Args:
        value (str | None): The timestamp string to validate, or None to skip.
        arg_name (str): The argument name, used in error messages.

    Returns:
        bool: ``True`` if *value* is ``None`` or a valid M:S timestamp,
              ``False`` otherwise.
    """
    if value is None:
        return True
    try:
        timestamp2sec(value)
        return True
    except argparse.ArgumentTypeError:
        return False


def sec2timestamp(sec: float) -> str:
    """Format seconds as a M:SS.ss timestamp string (e.g. '1:05.30').

    Args:
        sec (float): Time in seconds.

    Returns:
        str: Formatted timestamp string.
    """
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m}:{s:05.2f}"


def get_wav_end_ts(wav_path: str):
    return sec2timestamp(librosa.get_duration(path=wav_path))


class ClampedWav:
    """Trim a WAV file to [ts_start, ts_end] and manage the resulting temp file.

    The trimmed audio is written to a temporary WAV file on construction.
    The temp file is deleted automatically when:

    * the instance is garbage-collected (``__del__``), or
    * the Python process exits normally or via an unhandled exception
      (``atexit`` handler).

    Use as a plain object **or** as a context manager (``with`` statement) for
    deterministic, prompt cleanup:

    .. code-block:: python

        with ClampedWav(wav_path, "0:10", "1:30") as clamped:
            process(clamped.path)
        # temp file already gone here

    Attributes:
        path (str): Path to the temporary trimmed WAV file.
        offset_sec (float): Start position inside the original file (seconds).
        duration_sec (float): Length of the trimmed segment (seconds).
    """

    def __init__(
        self,
        wav_path: str,
        ts_start: str | None,
        ts_end: str | None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Trim *wav_path* to [ts_start, ts_end] and write it to a temp file.

        Both timestamps are clamped to ``[0, duration]`` before trimming.

        Args:
            wav_path (str): Path to the source WAV file.
            ts_start (str | None): Start timestamp in M:S format, or ``None``
                for the beginning of the file.
            ts_end (str | None): End timestamp in M:S format, or ``None`` for
                the end of the file.
            logger: Optional logger for clamp warnings.
        """
        total_duration = librosa.get_duration(path=wav_path)

        start_sec = timestamp2sec(ts_start) if ts_start is not None else 0.0
        end_sec   = timestamp2sec(ts_end)   if ts_end   is not None else total_duration

        # Clamp to valid range
        start_clamped = max(0.0, min(start_sec, total_duration))
        end_clamped   = max(0.0, min(end_sec,   total_duration))

        if logger is not None:
            if start_clamped != start_sec:
                logger.warning(
                    _("start {:.3f}s clamped to {:.3f}s (total duration: {:.3f}s)").format(
                        start_sec, start_clamped, total_duration
                    )
                )
            if end_clamped != end_sec:
                logger.warning(
                    _("end {:.3f}s clamped to {:.3f}s (total duration: {:.3f}s)").format(
                        end_sec, end_clamped, total_duration
                    )
                )

        self.offset_sec   = start_clamped
        self.duration_sec = end_clamped - start_clamped

        # Write trimmed audio to a named temp file
        y, sr = librosa.load(
            wav_path, sr=None, offset=self.offset_sec, duration=self.duration_sec
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        tmp.close()

        self.path: str = tmp.name

        # Register atexit so the file is removed even if __del__ is skipped
        # (e.g. interpreter shutdown, unhandled exception, or reference cycles).
        atexit.register(self._cleanup)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Delete the temp file if it still exists. Safe to call multiple times."""
        path, self.path = getattr(self, "path", None), ""
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass  # already gone — that's fine

    def __del__(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ClampedWav":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()
        return None  # do not suppress exceptions
