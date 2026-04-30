import os
import csv
import atexit
import logging
import argparse
import tempfile
from pathlib import Path

import scipy
import librosa
import numpy as np

from utils.i18n import _
from utils.fs import APP_CACHE_DIR, calculate_args_hash, calculate_file_hash


def extract_wav_embedding(
    wav_path,
    embedder: str = "mhubert",
    pca_dims: int | None = None,
    use_cache=True,
    **embedder_kwargs,
):
    """Extract frame-level embedding features from a WAV file.

    Args:
        wav_path:          Path to the input WAV file.
        embedder:          Registered embedder name — one of
                           :meth:`EmbedderFactory.list`.
        pca_dims:          If set, reduce the 5 feature rows to *pca_dims*
                           principal components using
                           :class:`sklearn.decomposition.PCA`.
                           ``None`` returns all 5 rows unchanged.
        use_cache:         Load from / save to a per-file ``.npz`` cache keyed
                           by file hash and embedder name.
        **embedder_kwargs: Forwarded to the embedder constructor (e.g.
                           ``variant``, ``device``).

    Returns:
        tuple: ``(frame_times, features)`` where

        - ``frame_times``: ``(T,)`` float32 array, seconds.
        - ``features``:    ``(5, T)`` or ``(pca_dims, T)`` float32 array.

    Raises:
        KeyError: If *embedder* is not in :meth:`EmbedderFactory.list`.
    """
    from utils.embedder import EmbedderFactory, emb_features

    if embedder not in EmbedderFactory.list():
        raise KeyError(
            f"Unknown embedder {embedder!r}. "
            f"Available: {EmbedderFactory.list()}"
        )

    cache_dir = Path(APP_CACHE_DIR) / "embedder"
    suffix    = f"{embedder}.{calculate_args_hash(**embedder_kwargs)}"
    if pca_dims is not None:
        suffix += f".pca{pca_dims}"

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        wav_hash   = calculate_file_hash(wav_path)
        cache_path = cache_dir / f"{wav_hash}.{suffix}.npz"

        if cache_path.is_file():
            print(f"[{embedder}] " + _("Loading embedding data from cache file: '{}'").format(cache_path))
            data = np.load(cache_path)
            return np.asarray(data["frame_times"]), np.asarray(data["embeddings"])

    emb, frame_times = EmbedderFactory.create(embedder, **embedder_kwargs)(wav_path)

    norms    = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)   # (T, D)
    features = emb_features(emb_norm)           # (5, T)

    if pca_dims is not None:
        from sklearn.decomposition import PCA
        features = PCA(n_components=pca_dims).fit_transform(features.T).T.astype(np.float32)  # (pca_dims, T)

    if use_cache:
        np.savez(cache_path, frame_times=frame_times, embeddings=features)
        print(f"[{embedder}] " + _("Embedding data saved to cache file: '{}'").format(cache_path))

    return np.asarray(frame_times), np.asarray(features)


def extract_wav_breath_voice(wav_path, breath_band=(10000, np.inf), voice_band=(0, 4000), use_cache=True):
    """Extract breath and voice intensity indices from a WAV file.

    Separates harmonic content via HPSS, then computes per-frame RMS energy
    in the specified frequency bands as dB indices.

    Args:
        wav_path (str): Path to the WAV file.
        breath_band (tuple, optional): Frequency range (Hz) for breath detection.
            Defaults to (10000, inf).
        voice_band (tuple, optional): Frequency range (Hz) for voice detection.
            Defaults to (0, 4000).
        use_cache (bool, optional): Whether to use cached data if available.
            Defaults to True.

    Returns:
        tuple: (time, breath_index, voice_index), where:
            - time (np.ndarray of float): Time points in seconds. Shape: (n_frames,).
            - breath_index (np.ndarray of float): Breath intensity in dB. Shape: (n_frames,).
            - voice_index (np.ndarray of float): Voice intensity in dB. Shape: (n_frames,).
    """
    cache_dir = Path(APP_CACHE_DIR) / "bv"

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        wav_hash   = calculate_file_hash(wav_path)
        bands_hash = calculate_args_hash(breath_band, voice_band)
        cache_path = cache_dir / f"{wav_hash}.{bands_hash}.npz"
        if cache_path.is_file():
            print(_("Loading breath/voice data from cache file: '{}'").format(cache_path))
            data = np.load(cache_path)
            return data["time"], data["breath_index"], data["voice_index"]

    y, sr  = librosa.load(wav_path, sr=None, mono=True)
    y_h, y_n = librosa.effects.hpss(y, margin=(1.0, 5.0))

    D     = librosa.stft(y_h, n_fft=2048, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_rms(mask):
        return np.sqrt(np.mean(np.abs(D[mask, :]) ** 2, axis=0))

    voice_mask   = (freqs >= voice_band[0])  & (freqs < voice_band[1])
    voice_index  = 20 * np.log10(band_rms(voice_mask)  + 1e-9)

    breath_mask  = (freqs >= breath_band[0]) & (freqs < breath_band[1])
    breath_index = 20 * np.log10(band_rms(breath_mask) + 1e-9)

    time = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=512)

    if use_cache:
        np.savez(cache_path, time=time, breath_index=breath_index, voice_index=voice_index)
        print(_("Breath/voice data saved to cache file: '{}'").format(cache_path))

    return time, breath_index, voice_index


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
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_feat)
    mfcc = pca.fit_transform(mfcc.T).T
    return mfcc_time, mfcc


def extract_wav_frequency(file_path, backend="rmvpe-onnx", use_cache=True):
    """Extract pitch frequency from a WAV file.

    This function processes an audio file to extract pitch information.
    It supports caching to improve performance when processing
    the same file multiple times.

    Args:
        file_path (str): Path to the WAV file.
        backend (str, optional): Pitch detection backend.
            "crepe" uses the CREPE model (requires TensorFlow, GPU-accelerated).
            "swift-f0" uses SwiftF0 (faster CPU inference, requires swift-f0 package).
            "rmvpe-onnx" uses RMVPE ONNX model (fast CPU inference, requires rmvpe-onnx package).
            "hybrid" uses a hybrid strategy based on "rmvpe-onnx" and "swift-f0".
            Defaults to "rmvpe-onnx".
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (np.ndarray of float): Time points in seconds. Shape: (n_time_points).
            - frequency (np.ndarray of float): Detected pitch frequencies in Hz. Shape: (n_time_points).
            - confidence (np.ndarray of float): Confidence values for the detected pitches. Shape: (n_time_points).
    """
    _SUPPORTED_BACKENDS = ("crepe", "swift-f0", "rmvpe-onnx", "hybrid")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {_SUPPORTED_BACKENDS}")

    time = []
    frequency = []
    confidence = []
    cache_dir = Path(APP_CACHE_DIR) / "f0"
    # Try reading data from cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        wav_hash = calculate_file_hash(file_path)

        cache_path = cache_dir / f"{wav_hash}.{backend}.csv"
        if cache_path.is_file():
            print(f"[{backend}] " + _("Loading F0 data from cache file: '{}'").format(cache_path))
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
            sr, audio = scipy.io.wavfile.read(file_path)
            time, frequency, confidence, _unused = crepe.predict(audio, sr, viterbi=True)
        elif backend == "swift-f0":
            from swift_f0 import SwiftF0
            detector = SwiftF0(confidence_threshold=0.0)
            result = detector.detect_from_file(file_path)
            time = result.timestamps.tolist()
            frequency = result.pitch_hz.tolist()
            confidence = result.confidence.tolist()
        elif backend == "rmvpe-onnx":
            from rmvpe_onnx import RMVPE
            import soundfile as sf
            audio, sr = sf.read(file_path)
            rmvpe = RMVPE(device="cpu")
            timestamp, frequency, confidence, _unused = rmvpe.predict(audio=audio, sr=sr)
            time = timestamp.tolist()
            frequency = frequency.tolist()
            confidence = confidence.tolist()
        elif backend == "hybrid":
            time, frequency, confidence = _merge_rmvpe_and_swift_f0(file_path, use_cache)

        # Save data to cache
        if use_cache:
            with open(cache_path, mode="w+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Frequency (Hz)", "Confidence"])
                for t, f, c in zip(time, frequency, confidence, strict=False):
                    writer.writerow([t, f, c])
            print(f"[{backend}] " + _("F0 data saved to cache file: '{}'").format(cache_path))

    return np.asarray(time), np.asarray(frequency), np.asarray(confidence)


def _merge_rmvpe_and_swift_f0(file_path, use_cache):
    """Merge rmvpe-onnx and swift-f0 pitch predictions into a single output.

    Uses rmvpe-onnx as the base prediction and selectively replaces frames with
    swift-f0 results where all three conditions are met:
      1. The frame falls within a voiced region (RMS energy >= Otsu threshold).
      2. rmvpe-onnx confidence is low, indicating uncertain prediction.
      3. swift-f0 confidence is high, indicating a reliable prediction.

    Voiced regions are detected by computing per-frame RMS energy with librosa,
    then thresholding with Otsu's method to separate voiced from unvoiced frames.
    swift-f0 frames are aligned to the rmvpe-onnx time grid via nearest-neighbour
    lookup before comparison.

    Args:
        file_path (str): Path to the WAV file. Passed directly to
            extract_wav_frequency for both backends.
        use_cache (bool): Whether to use cached predictions. Passed directly to
            extract_wav_frequency for both backends.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (list of float): Time points in seconds from the rmvpe-onnx grid.
            - frequency (list of float): Merged pitch frequencies in Hz.
            - confidence (list of float): Confidence values corresponding to
              whichever backend's frequency was selected per frame.
    """
    _CONFIDENCE_THRESHOLDS = {
        "rmvpe-onnx": 0.80,
        "swift-f0":   0.95,
    }

    from skimage.filters import threshold_otsu
    from librosa.feature import rms as librosa_rms

    r_time, r_freq, r_conf = extract_wav_frequency(file_path, backend="rmvpe-onnx", use_cache=use_cache)
    s_time, s_freq, s_conf = extract_wav_frequency(file_path, backend="swift-f0",   use_cache=use_cache)

    # --- Base prediction: start from rmvpe-onnx ---
    out_freq = r_freq.copy()
    out_conf = r_conf.copy()

    # --- Voiced region via Otsu threshold on RMS ---
    import soundfile as sf
    audio, sr = sf.read(file_path)
    hop_length = 512
    frame_rms = librosa_rms(y=audio, hop_length=hop_length)[0]
    rms_times  = np.arange(len(frame_rms)) * hop_length / sr
    otsu_thr   = threshold_otsu(frame_rms)
    # Snap RMS voiced mask to rmvpe-onnx time grid
    rms_indices   = np.searchsorted(rms_times, r_time).clip(0, len(frame_rms) - 1)
    voiced_region = frame_rms[rms_indices] >= otsu_thr

    # --- Align swift-f0 to rmvpe-onnx time grid via nearest-neighbour lookup ---
    snap_indices = np.searchsorted(s_time, r_time).clip(0, len(s_time) - 1)
    prev_indices = np.maximum(snap_indices - 1, 0)
    use_prev     = np.abs(s_time[prev_indices] - r_time) < np.abs(s_time[snap_indices] - r_time)
    snap_indices = np.where(use_prev, prev_indices, snap_indices)

    s_freq_aligned = s_freq[snap_indices]
    s_conf_aligned = s_conf[snap_indices]

    # --- Replace: voiced region + rmvpe low confidence + swift-f0 high confidence ---
    r_low  = r_conf < _CONFIDENCE_THRESHOLDS["rmvpe-onnx"]
    s_high = s_conf_aligned >= _CONFIDENCE_THRESHOLDS["swift-f0"]
    replace_mask = voiced_region & r_low & s_high

    out_freq = np.where(replace_mask, s_freq_aligned, out_freq)
    out_conf = np.where(replace_mask, s_conf_aligned, out_conf)

    return r_time.tolist(), out_freq.tolist(), out_conf.tolist()


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
        from skimage.filters import threshold_otsu
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
        import soundfile as sf

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
