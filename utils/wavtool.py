import os
import csv
import atexit
import logging
import argparse
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu

from utils.i18n import _
from utils.cache import CACHE_DIR, calculate_file_hash


def extract_wav_mfcc(wav_path, n_feat=6, n_mfcc=13):
    """Extract MFCC features from a WAV file.

    This function extracts Mel-frequency cepstral coefficients (MFCC) from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        n_feat (int, optional): Number of features to extract. Defaults to 6.
        n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to 13.

    Returns:
        tuple: (mfcc_time, mfcc), where:
            - mfcc_time (numpy.ndarray): Time points for the MFCC features. Shape: (n_time_points).
            - mfcc (numpy.ndarray): Extracted MFCC features. Shape: (n_features, n_time_points).
    """
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)

    # Extract MFCC features
    _mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_time = librosa.times_like(_mfcc, sr=sr)

    # Add dynamic features into the MFCC
    delta_mfcc = librosa.feature.delta(_mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(_mfcc, order=2)
    mfcc = np.vstack([_mfcc, delta_mfcc, delta2_mfcc])

    # PCA to reduce dimensionality
    pca = PCA(n_components=n_feat)
    mfcc = pca.fit_transform(mfcc.T).T
    return mfcc_time, mfcc


def extract_wav_frequency(file_path, backend="swift-f0", use_cache=True):
    """Extract pitch frequency from a WAV file.

    This function processes an audio file to extract pitch information.
    It supports caching to improve performance when processing
    the same file multiple times.

    Args:
        file_path (str): Path to the WAV file.
        backend (str, optional): Pitch detection backend. One of "crepe" or "swift-f0".
            "crepe" uses the CREPE model (requires TensorFlow, GPU-accelerated).
            "swift-f0" uses SwiftF0 (faster CPU inference, requires swift-f0 package).
            Defaults to "swift-f0".
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (list of float): Time points in seconds. Shape: (n_time_points).
            - frequency (list of float): Detected pitch frequencies in Hz. Shape: (n_time_points).
            - confidence (list of float): Confidence values for the detected pitches. Shape: (n_time_points).
    """
    _SUPPORTED_BACKENDS = ("crepe", "swift-f0")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {_SUPPORTED_BACKENDS}")

    time = []
    frequency = []
    confidence = []
    cache_dir = Path(CACHE_DIR) / "pitd"
    # Try reading data from cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        wav_hash = calculate_file_hash(file_path)

        cache_path = cache_dir / f"{wav_hash}.{backend}.csv"
        if cache_path.is_file():
            print(_("Loading F0 data from cache file: '{}'").format(cache_path))
            with open(cache_path, "r", newline="") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    time.append(float(row[0]))
                    frequency.append(float(row[1]))
                    confidence.append(float(row[2]))

    # If cache is unavailable
    if not all([time, frequency, confidence]):
        # Extract pitch using the specified backend
        if backend == "crepe":
            import crepe
            from utils.gpu import add_cuda_to_path
            add_cuda_to_path(skip_missing=True)
            sr, audio = wavfile.read(file_path)
            time, frequency, confidence, _unused = crepe.predict(audio, sr, viterbi=True)
        elif backend == "swift-f0":
            from swift_f0 import SwiftF0
            detector = SwiftF0(confidence_threshold=0.0)
            result = detector.detect_from_file(file_path)
            time = result.timestamps.tolist()
            frequency = result.pitch_hz.tolist()
            confidence = result.confidence.tolist()

        # Save data to cache
        if use_cache:
            with open(cache_path, mode="w+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Frequency (Hz)", "Confidence"])
                for t, f, c in zip(time, frequency, confidence, strict=False):
                    writer.writerow([t, f, c])
            print(_("F0 data saved to cache file: '{}'").format(cache_path))

    return time, frequency, confidence


def extract_wav_rms(wav_path, mask_silence=True):
    """Extract RMS energy from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        mask_silence (bool, optional): If True, masks leading and trailing silence with NaN
            using Otsu's method to auto-detect the silence threshold. Defaults to True.

    Returns:
        tuple: (rms_time, rms), where:
            - rms_time (numpy.ndarray): Time values for each RMS frame. Shape: (n_frames,).
            - rms (numpy.ndarray): RMS energy values, with NaN at silent edges if mask_silence
              is True. Shape: (n_frames,).
    """
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_time = librosa.times_like(rms, sr=sr)
    if mask_silence:
        threshold   = threshold_otsu(rms)
        is_silent   = rms < threshold
        start_frame = np.argmax(~is_silent)
        end_frame   = len(is_silent) - np.argmax(~is_silent[::-1])
        rms[:start_frame] = np.nan
        rms[end_frame:]   = np.nan
    return rms_time, rms


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
