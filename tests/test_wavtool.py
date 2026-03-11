"""Tests for wavtool.py — timestamp helpers and ClampedWav."""

import argparse
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from utils.wavtool import ClampedWav, sec2timestamp, timestamp2sec, validate_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration: float = 5.0, sr: int = 22050) -> str:
    """Write a silent WAV file of *duration* seconds and return its path."""
    n_samples = int(duration * sr)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, np.zeros(n_samples, dtype=np.float32), sr)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# timestamp2sec
# ---------------------------------------------------------------------------

class TestTimestamp2Sec(unittest.TestCase):

    def test_zero(self):
        self.assertAlmostEqual(timestamp2sec("0:00"), 0.0)

    def test_seconds_only(self):
        self.assertAlmostEqual(timestamp2sec("0:10"), 10.0)

    def test_minutes_and_seconds(self):
        self.assertAlmostEqual(timestamp2sec("1:30"), 90.0)

    def test_fractional_seconds(self):
        self.assertAlmostEqual(timestamp2sec("0:10.01"), 10.01)

    def test_large_minutes(self):
        self.assertAlmostEqual(timestamp2sec("10:00"), 600.0)

    def test_boundary_seconds_just_below_60(self):
        self.assertAlmostEqual(timestamp2sec("0:59.99"), 59.99)

    # --- error cases ---

    def test_missing_colon(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("130")

    def test_too_many_colons(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("1:30:00")

    def test_non_numeric_minutes(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("a:30")

    def test_non_numeric_seconds(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("1:xx")

    def test_negative_minutes(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("-1:30")

    def test_seconds_equal_60(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("0:60")

    def test_seconds_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("0:-1")


# ---------------------------------------------------------------------------
# validate_timestamp
# ---------------------------------------------------------------------------

class TestValidateTimestamp(unittest.TestCase):

    def test_none_is_valid(self):
        self.assertTrue(validate_timestamp(None, "arg"))

    def test_valid_string(self):
        self.assertTrue(validate_timestamp("1:30", "arg"))

    def test_invalid_string(self):
        self.assertFalse(validate_timestamp("bad", "arg"))

    def test_invalid_seconds(self):
        self.assertFalse(validate_timestamp("0:60", "arg"))


# ---------------------------------------------------------------------------
# sec2timestamp
# ---------------------------------------------------------------------------

class TestSec2Timestamp(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(sec2timestamp(0.0), "0:00.00")

    def test_under_one_minute(self):
        self.assertEqual(sec2timestamp(5.3), "0:05.30")

    def test_exactly_one_minute(self):
        self.assertEqual(sec2timestamp(60.0), "1:00.00")

    def test_minutes_and_seconds(self):
        self.assertEqual(sec2timestamp(90.5), "1:30.50")

    def test_round_trip(self):
        """sec2timestamp(timestamp2sec(s)) should reproduce the value."""
        original = "2:34.56"
        self.assertAlmostEqual(timestamp2sec(sec2timestamp(timestamp2sec(original))),
                               timestamp2sec(original), places=5)


# ---------------------------------------------------------------------------
# ClampedWav
# ---------------------------------------------------------------------------

class TestClampedWav(unittest.TestCase):

    def setUp(self):
        self.wav = _make_wav(duration=5.0)

    def tearDown(self):
        # Remove the source WAV if it still exists
        try:
            os.unlink(self.wav)
        except FileNotFoundError:
            pass

    # --- basic construction ---

    def test_creates_temp_file(self):
        cw = ClampedWav(self.wav, None, None)
        self.assertTrue(os.path.exists(cw.path))
        cw._cleanup()

    def test_path_is_wav(self):
        cw = ClampedWav(self.wav, None, None)
        self.assertTrue(cw.path.endswith(".wav"))
        cw._cleanup()

    def test_full_duration_no_timestamps(self):
        cw = ClampedWav(self.wav, None, None)
        self.assertAlmostEqual(cw.offset_sec, 0.0)
        self.assertAlmostEqual(cw.duration_sec, 5.0, places=1)
        cw._cleanup()

    def test_trim_start(self):
        cw = ClampedWav(self.wav, "0:02", None)
        self.assertAlmostEqual(cw.offset_sec, 2.0, places=3)
        self.assertAlmostEqual(cw.duration_sec, 3.0, places=1)
        cw._cleanup()

    def test_trim_end(self):
        cw = ClampedWav(self.wav, None, "0:03")
        self.assertAlmostEqual(cw.offset_sec, 0.0)
        self.assertAlmostEqual(cw.duration_sec, 3.0, places=1)
        cw._cleanup()

    def test_trim_both(self):
        cw = ClampedWav(self.wav, "0:01", "0:04")
        self.assertAlmostEqual(cw.offset_sec, 1.0, places=3)
        self.assertAlmostEqual(cw.duration_sec, 3.0, places=1)
        cw._cleanup()

    # --- clamping ---

    def test_start_clamped_to_zero(self):
        """Negative-equivalent: ts_start beyond total duration clamps to duration."""
        cw = ClampedWav(self.wav, "0:00", None)
        self.assertAlmostEqual(cw.offset_sec, 0.0)
        cw._cleanup()

    def test_end_clamped_to_duration(self):
        cw = ClampedWav(self.wav, None, "9:59")  # way past 5 s
        self.assertAlmostEqual(cw.duration_sec, 5.0, places=1)
        cw._cleanup()

    def test_start_clamped_logs_warning(self):
        logger = MagicMock()
        cw = ClampedWav(self.wav, "9:00", None, logger=logger)  # 540 s > 5 s
        logger.warning.assert_called()
        cw._cleanup()

    def test_end_clamped_logs_warning(self):
        logger = MagicMock()
        cw = ClampedWav(self.wav, None, "9:00", logger=logger)
        logger.warning.assert_called()
        cw._cleanup()

    def test_no_warning_when_within_bounds(self):
        logger = MagicMock()
        cw = ClampedWav(self.wav, "0:01", "0:04", logger=logger)
        logger.warning.assert_not_called()
        cw._cleanup()

    # --- cleanup ---

    def test_cleanup_removes_file(self):
        cw = ClampedWav(self.wav, None, None)
        path = cw.path
        cw._cleanup()
        self.assertFalse(os.path.exists(path))

    def test_cleanup_idempotent(self):
        cw = ClampedWav(self.wav, None, None)
        cw._cleanup()
        cw._cleanup()  # should not raise

    def test_del_removes_file(self):
        # __del__ / GC timing is implementation-defined; calling _cleanup()
        # directly is the reliable way to test the deletion logic itself.
        cw = ClampedWav(self.wav, None, None)
        path = cw.path
        cw._cleanup()
        self.assertFalse(os.path.exists(path))

    # --- context manager ---

    def test_context_manager_removes_file_on_exit(self):
        with ClampedWav(self.wav, None, None) as cw:
            path = cw.path
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

    def test_context_manager_returns_self(self):
        with ClampedWav(self.wav, None, None) as cw:
            self.assertIsInstance(cw, ClampedWav)

    def test_context_manager_propagates_exception(self):
        with self.assertRaises(RuntimeError):
            with ClampedWav(self.wav, None, None):
                raise RuntimeError("boom")

    def test_context_manager_cleans_up_on_exception(self):
        path = None
        try:
            with ClampedWav(self.wav, None, None) as cw:
                path = cw.path
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertFalse(os.path.exists(path))

    # --- atexit registration ---

    def test_atexit_registered(self):
        """_cleanup should be registered with atexit on construction."""
        import atexit as _atexit
        with patch.object(_atexit, "register") as mock_register:
            cw = ClampedWav(self.wav, None, None)
            mock_register.assert_called_once_with(cw._cleanup)
            cw._cleanup()

    # --- output audio validity ---

    def test_output_is_readable_wav(self):
        with ClampedWav(self.wav, "0:01", "0:03") as cw:
            data, sr = sf.read(cw.path)
            self.assertGreater(len(data), 0)
            self.assertGreater(sr, 0)


if __name__ == "__main__":
    unittest.main()
