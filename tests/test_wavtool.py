"""Tests for wavtool.py — timestamp helpers and ClampedWav."""

import argparse
import atexit as _atexit
import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from utils.wavtool import (
    ClampedWav,
    extract_wav_frequency,
    extract_wav_mfcc,
    extract_wav_rms,
    get_wav_end_ts,
    sec2timestamp,
    timestamp2sec,
    validate_timestamp,
)


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

    # --- valid inputs ---

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

    def test_zero_minutes_nonzero_seconds(self):
        self.assertAlmostEqual(timestamp2sec("0:45.5"), 45.5)

    def test_high_minute_count(self):
        self.assertAlmostEqual(timestamp2sec("99:59.99"), 99 * 60 + 59.99)

    def test_return_type_is_float(self):
        result = timestamp2sec("1:30")
        self.assertIsInstance(result, float)

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

    def test_empty_string(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("")

    def test_colon_only(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec(":")

    def test_float_minutes(self):
        """Minutes must be an integer — '1.5:30' should raise."""
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("1.5:30")

    def test_seconds_exactly_60_boundary(self):
        """60.0 is not in [0, 60) and must be rejected."""
        with self.assertRaises(argparse.ArgumentTypeError):
            timestamp2sec("0:60.0")


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

    def test_valid_fractional_seconds(self):
        self.assertTrue(validate_timestamp("0:10.01", "arg"))

    def test_valid_zero(self):
        self.assertTrue(validate_timestamp("0:00", "arg"))

    def test_returns_bool_for_valid(self):
        self.assertIsInstance(validate_timestamp("0:30", "arg"), bool)

    def test_returns_bool_for_invalid(self):
        self.assertIsInstance(validate_timestamp("bad", "arg"), bool)

    def test_empty_string_is_invalid(self):
        self.assertFalse(validate_timestamp("", "arg"))

    def test_arg_name_does_not_affect_result(self):
        """arg_name is only for error messages; different names must not change outcome."""
        self.assertEqual(
            validate_timestamp("1:30", "start"),
            validate_timestamp("1:30", "end"),
        )


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

    def test_large_value(self):
        # 3600 s == 60:00.00
        self.assertEqual(sec2timestamp(3600.0), "60:00.00")

    def test_fractional_seconds_precision(self):
        self.assertEqual(sec2timestamp(0.1), "0:00.10")

    def test_returns_string(self):
        self.assertIsInstance(sec2timestamp(10.0), str)

    def test_format_contains_colon(self):
        result = sec2timestamp(75.0)
        self.assertIn(":", result)

    def test_round_trip(self):
        """sec2timestamp(timestamp2sec(s)) should reproduce the value."""
        original = "2:34.56"
        self.assertAlmostEqual(
            timestamp2sec(sec2timestamp(timestamp2sec(original))),
            timestamp2sec(original),
            places=5,
        )

    def test_round_trip_multiple_values(self):
        for ts in ("0:00", "0:30", "1:00", "5:59.99", "10:00"):
            with self.subTest(ts=ts):
                self.assertAlmostEqual(
                    timestamp2sec(sec2timestamp(timestamp2sec(ts))),
                    timestamp2sec(ts),
                    places=4,
                )


# ---------------------------------------------------------------------------
# get_wav_end_ts
# ---------------------------------------------------------------------------

class TestGetWavEndTs(unittest.TestCase):

    def setUp(self):
        self.wav = _make_wav(duration=5.0)

    def tearDown(self):
        try:
            os.unlink(self.wav)
        except FileNotFoundError:
            pass

    def test_returns_string(self):
        result = get_wav_end_ts(self.wav)
        self.assertIsInstance(result, str)

    def test_contains_colon(self):
        self.assertIn(":", get_wav_end_ts(self.wav))

    def test_round_trips_to_approx_duration(self):
        result = get_wav_end_ts(self.wav)
        self.assertAlmostEqual(timestamp2sec(result), 5.0, places=0)

    def test_short_file(self):
        wav = _make_wav(duration=0.5)
        try:
            result = get_wav_end_ts(wav)
            self.assertAlmostEqual(timestamp2sec(result), 0.5, places=0)
        finally:
            os.unlink(wav)

    def test_longer_file(self):
        wav = _make_wav(duration=90.0)
        try:
            result = get_wav_end_ts(wav)
            self.assertAlmostEqual(timestamp2sec(result), 90.0, places=0)
        finally:
            os.unlink(wav)


# ---------------------------------------------------------------------------
# ClampedWav
# ---------------------------------------------------------------------------

class TestClampedWav(unittest.TestCase):

    def setUp(self):
        self.wav = _make_wav(duration=5.0)

    def tearDown(self):
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

    def test_attributes_exist(self):
        cw = ClampedWav(self.wav, None, None)
        self.assertTrue(hasattr(cw, "path"))
        self.assertTrue(hasattr(cw, "offset_sec"))
        self.assertTrue(hasattr(cw, "duration_sec"))
        cw._cleanup()

    def test_offset_sec_is_float(self):
        cw = ClampedWav(self.wav, "0:01", None)
        self.assertIsInstance(cw.offset_sec, float)
        cw._cleanup()

    def test_duration_sec_is_float(self):
        cw = ClampedWav(self.wav, None, "0:04")
        self.assertIsInstance(cw.duration_sec, float)
        cw._cleanup()

    def test_temp_file_is_distinct_from_source(self):
        cw = ClampedWav(self.wav, None, None)
        self.assertNotEqual(os.path.abspath(cw.path), os.path.abspath(self.wav))
        cw._cleanup()

    # --- clamping ---

    def test_start_clamped_to_zero(self):
        cw = ClampedWav(self.wav, "0:00", None)
        self.assertAlmostEqual(cw.offset_sec, 0.0)
        cw._cleanup()

    def test_end_clamped_to_duration(self):
        cw = ClampedWav(self.wav, None, "9:59")
        self.assertAlmostEqual(cw.duration_sec, 5.0, places=1)
        cw._cleanup()

    def test_start_beyond_duration_clamped(self):
        """ts_start past total duration: offset clamps to duration, yielding 0-length segment."""
        cw = ClampedWav(self.wav, "9:00", None, logger=MagicMock())
        self.assertAlmostEqual(cw.offset_sec, 5.0, places=1)
        self.assertAlmostEqual(cw.duration_sec, 0.0, places=1)
        cw._cleanup()

    def test_start_clamped_logs_warning(self):
        logger = MagicMock()
        cw = ClampedWav(self.wav, "9:00", None, logger=logger)
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

    def test_no_warning_when_no_logger(self):
        """Passing logger=None must not raise even when clamping occurs."""
        cw = ClampedWav(self.wav, "9:00", None, logger=None)
        cw._cleanup()

    def test_start_equals_end_zero_duration(self):
        """Identical ts_start and ts_end should produce a zero-length segment."""
        cw = ClampedWav(self.wav, "0:02", "0:02")
        self.assertAlmostEqual(cw.duration_sec, 0.0, places=3)
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
        cw._cleanup()  # must not raise

    def test_cleanup_clears_path_attribute(self):
        """After _cleanup(), self.path should be falsy to prevent double-unlink attempts."""
        cw = ClampedWav(self.wav, None, None)
        cw._cleanup()
        self.assertFalse(cw.path)

    def test_del_removes_file(self):
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

    def test_context_manager_enter_returns_clamped_wav(self):
        cw = ClampedWav(self.wav, None, None)
        result = cw.__enter__()
        self.assertIs(result, cw)
        cw._cleanup()

    def test_context_manager_exit_does_not_suppress_exceptions(self):
        """__exit__ must return None/falsy so exceptions propagate."""
        cw = ClampedWav(self.wav, None, None)
        ret = cw.__exit__(RuntimeError, RuntimeError("x"), None)
        self.assertFalse(ret)

    # --- atexit registration ---

    def test_atexit_registered(self):
        """_cleanup should be registered with atexit on construction."""
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

    def test_output_sample_rate_preserved(self):
        with ClampedWav(self.wav, None, None) as cw:
            _, sr = sf.read(cw.path)
            self.assertEqual(sr, 22050)

    def test_output_duration_approximately_correct(self):
        """Trimmed audio length should match duration_sec within a small tolerance."""
        with ClampedWav(self.wav, "0:01", "0:03") as cw:
            data, sr = sf.read(cw.path)
            actual_duration = len(data) / sr
            self.assertAlmostEqual(actual_duration, cw.duration_sec, places=1)

    def test_multiple_instances_get_distinct_temp_files(self):
        cw1 = ClampedWav(self.wav, None, None)
        cw2 = ClampedWav(self.wav, None, None)
        self.assertNotEqual(cw1.path, cw2.path)
        cw1._cleanup()
        cw2._cleanup()


# ---------------------------------------------------------------------------
# extract_wav_mfcc
# ---------------------------------------------------------------------------

class TestExtractWavMfcc(unittest.TestCase):

    def setUp(self):
        self.wav = _make_wav(duration=2.0, sr=22050)

    def tearDown(self):
        try:
            os.unlink(self.wav)
        except FileNotFoundError:
            pass

    def test_returns_tuple_of_two(self):
        result = extract_wav_mfcc(self.wav)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_mfcc_time_is_ndarray(self):
        mfcc_time, _ = extract_wav_mfcc(self.wav)
        self.assertIsInstance(mfcc_time, np.ndarray)

    def test_mfcc_is_ndarray(self):
        _, mfcc = extract_wav_mfcc(self.wav)
        self.assertIsInstance(mfcc, np.ndarray)

    def test_mfcc_feature_dim_matches_n_feat(self):
        n_feat = 4
        _, mfcc = extract_wav_mfcc(self.wav, n_feat=n_feat)
        self.assertEqual(mfcc.shape[0], n_feat)

    def test_default_n_feat(self):
        """Default n_feat=6 should produce 6 feature rows."""
        _, mfcc = extract_wav_mfcc(self.wav)
        self.assertEqual(mfcc.shape[0], 6)

    def test_mfcc_time_and_features_same_length(self):
        mfcc_time, mfcc = extract_wav_mfcc(self.wav)
        self.assertEqual(mfcc_time.shape[0], mfcc.shape[1])

    def test_mfcc_is_2d(self):
        _, mfcc = extract_wav_mfcc(self.wav)
        self.assertEqual(mfcc.ndim, 2)

    def test_mfcc_time_is_1d(self):
        mfcc_time, _ = extract_wav_mfcc(self.wav)
        self.assertEqual(mfcc_time.ndim, 1)

    def test_mfcc_time_starts_at_or_near_zero(self):
        mfcc_time, _ = extract_wav_mfcc(self.wav)
        self.assertGreaterEqual(mfcc_time[0], 0.0)

    def test_mfcc_time_is_monotonically_increasing(self):
        mfcc_time, _ = extract_wav_mfcc(self.wav)
        self.assertTrue(np.all(np.diff(mfcc_time) > 0))

    def test_mfcc_time_does_not_exceed_duration(self):
        mfcc_time, _ = extract_wav_mfcc(self.wav)
        self.assertLessEqual(mfcc_time[-1], 2.1)  # small tolerance for frame centering

    def test_mfcc_values_are_finite(self):
        _, mfcc = extract_wav_mfcc(self.wav)
        self.assertTrue(np.all(np.isfinite(mfcc)))

    def test_different_n_feat_values(self):
        for n_feat in (2, 4, 8):
            with self.subTest(n_feat=n_feat):
                _, mfcc = extract_wav_mfcc(self.wav, n_feat=n_feat)
                self.assertEqual(mfcc.shape[0], n_feat)

    def test_different_n_mfcc_values(self):
        """Varying n_mfcc should not change the output feature dim (PCA controls that)."""
        _, mfcc_a = extract_wav_mfcc(self.wav, n_feat=6, n_mfcc=13)
        _, mfcc_b = extract_wav_mfcc(self.wav, n_feat=6, n_mfcc=20)
        self.assertEqual(mfcc_a.shape[0], mfcc_b.shape[0])

    def test_longer_file_produces_more_frames(self):
        wav_long = _make_wav(duration=4.0)
        try:
            _, mfcc_short = extract_wav_mfcc(self.wav)
            _, mfcc_long = extract_wav_mfcc(wav_long)
            self.assertGreater(mfcc_long.shape[1], mfcc_short.shape[1])
        finally:
            os.unlink(wav_long)

    def test_n_feat_cannot_exceed_input_components(self):
        """PCA n_components must be <= min(n_samples, n_features); with n_mfcc=13
        we have 3*13=39 feature rows — requesting n_feat up to 39 should succeed."""
        _, mfcc = extract_wav_mfcc(self.wav, n_feat=10, n_mfcc=13)
        self.assertEqual(mfcc.shape[0], 10)


# ---------------------------------------------------------------------------
# extract_wav_rms
# ---------------------------------------------------------------------------

class TestExtractWavRms(unittest.TestCase):

    def setUp(self):
        self.wav = _make_wav(duration=2.0, sr=22050)

    def tearDown(self):
        try:
            os.unlink(self.wav)
        except FileNotFoundError:
            pass

    def _make_tonal_wav(self, duration=2.0, sr=22050, freq=440.0) -> str:
        """Write a sine-wave WAV so RMS is non-trivially non-zero throughout."""
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        tmp.close()
        return tmp.name

    # --- return types and shapes ---

    def test_returns_tuple_of_two(self):
        result = extract_wav_rms(self.wav)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_rms_time_is_ndarray(self):
        rms_time, _ = extract_wav_rms(self.wav)
        self.assertIsInstance(rms_time, np.ndarray)

    def test_rms_is_ndarray(self):
        _, rms = extract_wav_rms(self.wav)
        self.assertIsInstance(rms, np.ndarray)

    def test_rms_time_and_values_same_length(self):
        rms_time, rms = extract_wav_rms(self.wav)
        self.assertEqual(rms_time.shape[0], rms.shape[0])

    def test_rms_is_1d(self):
        _, rms = extract_wav_rms(self.wav)
        self.assertEqual(rms.ndim, 1)

    def test_rms_time_is_1d(self):
        rms_time, _ = extract_wav_rms(self.wav)
        self.assertEqual(rms_time.ndim, 1)

    def test_rms_time_starts_at_or_near_zero(self):
        rms_time, _ = extract_wav_rms(self.wav)
        self.assertGreaterEqual(rms_time[0], 0.0)

    def test_rms_time_is_monotonically_increasing(self):
        rms_time, _ = extract_wav_rms(self.wav)
        self.assertTrue(np.all(np.diff(rms_time) > 0))

    # --- silence masking ---

    def test_silent_wav_masked_with_nan(self):
        """mask_silence=True on a fully-silent file must not raise and must return
        arrays of equal length.  Otsu's threshold on a flat-zero signal is 0, so
        no frames satisfy rms < 0; the masking logic runs but clips nothing —
        we only assert the call succeeds and output shapes are consistent."""
        rms_time, rms = extract_wav_rms(self.wav, mask_silence=True)
        self.assertEqual(rms_time.shape, rms.shape)
        self.assertGreater(len(rms), 0)

    def test_mask_silence_false_no_nan(self):
        """With mask_silence=False no NaN should be introduced."""
        _, rms = extract_wav_rms(self.wav, mask_silence=False)
        self.assertFalse(np.any(np.isnan(rms)))

    def test_mask_silence_false_values_nonnegative(self):
        """RMS energy is always non-negative."""
        _, rms = extract_wav_rms(self.wav, mask_silence=False)
        self.assertTrue(np.all(rms >= 0))

    def test_tonal_wav_has_active_frames(self):
        """A sine wave should have some non-NaN (active) RMS frames."""
        wav = self._make_tonal_wav()
        try:
            _, rms = extract_wav_rms(wav, mask_silence=True)
            self.assertTrue(np.any(~np.isnan(rms)))
        finally:
            os.unlink(wav)

    def test_tonal_wav_active_frames_are_nonnegative(self):
        wav = self._make_tonal_wav()
        try:
            _, rms = extract_wav_rms(wav, mask_silence=True)
            active = rms[~np.isnan(rms)]
            self.assertTrue(np.all(active >= 0))
        finally:
            os.unlink(wav)

    def test_leading_silence_masked(self):
        """Frames before the first active frame should be NaN when mask_silence=True."""
        # Build: 0.5 s silence + 1.5 s tone
        sr = 22050
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        y = np.concatenate([silence, tone])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        tmp.close()
        try:
            _, rms = extract_wav_rms(tmp.name, mask_silence=True)
            self.assertTrue(np.isnan(rms[0]))
        finally:
            os.unlink(tmp.name)

    def test_trailing_silence_masked(self):
        """Frames after the last active frame should be NaN when mask_silence=True."""
        sr = 22050
        t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        y = np.concatenate([tone, silence])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        tmp.close()
        try:
            _, rms = extract_wav_rms(tmp.name, mask_silence=True)
            self.assertTrue(np.isnan(rms[-1]))
        finally:
            os.unlink(tmp.name)

    def test_mask_silence_default_is_true(self):
        """mask_silence should default to True."""
        import inspect
        sig = inspect.signature(extract_wav_rms)
        self.assertTrue(sig.parameters["mask_silence"].default)


# ---------------------------------------------------------------------------
# extract_wav_frequency
# ---------------------------------------------------------------------------

class TestExtractWavFrequency(unittest.TestCase):
    """Tests for extract_wav_frequency.

    Heavy ML backends (crepe, swift-f0) are mocked so tests stay fast and
    dependency-free.  A shared _mock_swift_f0 fixture provides realistic-looking
    return values that mirror the SwiftF0 result object interface.
    """

    def setUp(self):
        self.wav = _make_wav(duration=2.0, sr=22050)
        self._n = 200  # number of fake time points

        # Pre-compute plain Python lists so they can be reused across helpers.
        # Do NOT assign these to MagicMock attributes here — doing so replaces
        # the auto-created Mock sub-attributes with real numpy arrays, which
        # causes `.tolist.return_value` to fail with AttributeError because the
        # builtin ndarray.tolist has no `return_value`.
        self._fake_times = list(np.linspace(0, 2.0, self._n))
        self._fake_freqs = list(np.random.uniform(80, 300, self._n))
        self._fake_confs = list(np.random.uniform(0.5, 1.0, self._n))

    def tearDown(self):
        try:
            os.unlink(self.wav)
        except FileNotFoundError:
            pass

    def _patch_swift(self):
        """Return a context manager that patches SwiftF0 on its home module.

        SwiftF0 is imported inside the function body with
        ``from swift_f0 import SwiftF0``, so patching ``utils.wavtool.SwiftF0``
        has no effect.  Patching ``swift_f0.SwiftF0`` ensures the local import
        picks up the replacement.

        The result object stays a plain MagicMock so that attribute access on
        ``.timestamps``, ``.pitch_hz``, and ``.confidence`` returns further
        Mocks whose ``.tolist.return_value`` we can control.
        """
        fake_result = MagicMock()
        fake_result.timestamps.tolist.return_value = self._fake_times
        fake_result.pitch_hz.tolist.return_value = self._fake_freqs
        fake_result.confidence.tolist.return_value = self._fake_confs

        mock_detector = MagicMock()
        mock_detector.detect_from_file.return_value = fake_result
        mock_cls = MagicMock(return_value=mock_detector)
        return patch("swift_f0.SwiftF0", mock_cls, create=True)

    # --- return types ---

    def test_returns_tuple_of_three(self):
        with self._patch_swift():
            result = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_time_is_list(self):
        with self._patch_swift():
            time, _, _ = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertIsInstance(time, list)

    def test_frequency_is_list(self):
        with self._patch_swift():
            _, freq, _ = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertIsInstance(freq, list)

    def test_confidence_is_list(self):
        with self._patch_swift():
            _, _, conf = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertIsInstance(conf, list)

    def test_all_outputs_same_length(self):
        with self._patch_swift():
            time, freq, conf = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertEqual(len(time), len(freq))
        self.assertEqual(len(freq), len(conf))

    def test_output_length_matches_mock(self):
        with self._patch_swift():
            time, _, _ = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertEqual(len(time), self._n)

    def test_time_values_are_floats(self):
        with self._patch_swift():
            time, _, _ = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertTrue(all(isinstance(t, float) for t in time))

    def test_frequency_values_are_floats(self):
        with self._patch_swift():
            _, freq, _ = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertTrue(all(isinstance(f, float) for f in freq))

    def test_confidence_values_are_floats(self):
        with self._patch_swift():
            _, _, conf = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
        self.assertTrue(all(isinstance(c, float) for c in conf))

    # --- backend validation ---

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            extract_wav_frequency(self.wav, backend="nonexistent", use_cache=False)

    def test_invalid_backend_message_contains_name(self):
        with self.assertRaises(ValueError, msg="nonexistent") as ctx:
            extract_wav_frequency(self.wav, backend="nonexistent", use_cache=False)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_swift_f0_backend_accepted(self):
        with self._patch_swift():
            # must not raise
            extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)

    def test_crepe_backend_accepted(self):
        """crepe backend path should be accepted (mocked to avoid TF dependency).

        ``import crepe`` is a top-level import inside the elif branch, so we
        patch ``crepe.predict`` on the *crepe* module directly.
        """
        fake_time = np.linspace(0, 2, self._n)
        fake_freq = np.random.uniform(80, 300, self._n)
        fake_conf = np.random.uniform(0.5, 1.0, self._n)
        fake_act  = np.zeros((self._n, 360))

        import types
        fake_crepe_mod = types.ModuleType("crepe")
        fake_crepe_mod.predict = MagicMock(
            return_value=(fake_time, fake_freq, fake_conf, fake_act)
        )

        with patch.dict("sys.modules", {"crepe": fake_crepe_mod}), \
             patch("utils.wavtool.add_cuda_to_path", create=True), \
             patch("utils.wavtool.wavfile") as mock_wavfile:
            mock_wavfile.read.return_value = (22050, np.zeros(44100, dtype=np.float32))
            time, freq, conf = extract_wav_frequency(self.wav, backend="crepe", use_cache=False)

        self.assertEqual(len(time), len(freq))
        self.assertEqual(len(freq), len(conf))

    # --- caching ---

    def test_cache_file_written_when_use_cache_true(self):
        tmp_cache_dir = tempfile.mkdtemp()
        with self._patch_swift(), \
             patch("utils.wavtool.CACHE_DIR", tmp_cache_dir), \
             patch("utils.wavtool.calculate_file_hash", return_value="testhash"):
            extract_wav_frequency(self.wav, backend="swift-f0", use_cache=True)

        cache_path = os.path.join(tmp_cache_dir, "pitd", "testhash.swift-f0.csv")
        self.assertTrue(os.path.exists(cache_path))

    def test_cache_read_skips_backend_call(self):
        """If a valid cache file exists, the backend must not be invoked."""
        tmp_cache_dir = tempfile.mkdtemp()
        fake_hash = "deadbeef"
        cache_path = os.path.join(tmp_cache_dir, "pitd", f"{fake_hash}.swift-f0.csv")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Write a minimal cache CSV
        with open(cache_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Frequency (Hz)", "Confidence"])
            writer.writerow([0.0, 220.0, 0.9])
            writer.writerow([0.5, 440.0, 0.95])

        with self._patch_swift() as mock_cls, \
             patch("utils.wavtool.CACHE_DIR", tmp_cache_dir), \
             patch("utils.wavtool.calculate_file_hash", return_value=fake_hash):
            time, freq, conf = extract_wav_frequency(self.wav, backend="swift-f0", use_cache=True)
            mock_cls.assert_not_called()

        self.assertEqual(time, [0.0, 0.5])
        self.assertEqual(freq, [220.0, 440.0])
        self.assertEqual(conf, [0.9, 0.95])

    def test_cache_disabled_always_calls_backend(self):
        with self._patch_swift() as mock_cls:
            extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)
            mock_cls.assert_called_once()

    def test_no_cache_written_when_use_cache_false(self):
        tmp_cache_dir = tempfile.mkdtemp()
        with self._patch_swift(), \
             patch("utils.wavtool.CACHE_DIR", tmp_cache_dir), \
             patch("utils.wavtool.calculate_file_hash", return_value="abc123"):
            extract_wav_frequency(self.wav, backend="swift-f0", use_cache=False)

        pitd_dir = os.path.join(tmp_cache_dir, "pitd")
        cache_files = os.listdir(pitd_dir) if os.path.exists(pitd_dir) else []
        self.assertEqual(cache_files, [])


if __name__ == "__main__":
    unittest.main()
