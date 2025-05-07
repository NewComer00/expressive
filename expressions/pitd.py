import os
import csv
from pathlib import Path
from types import SimpleNamespace

import crepe
import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from librosa import hz_to_midi

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)
from utils.log import StreamToLogger
from utils.cache import CACHE_DIR, calculate_file_hash


@register_expression
class PitdLoader(ExpressionLoader):
    expression_name = "pitd"
    expression_info = _("Pitch Deviation (curve)")
    args = SimpleNamespace(
        confidence_utau = Args(name="confidence_utau", type=float, default=0.8 , help=_("Confidence threshold for filtering uncertain pitch values in UTAU WAV")),
        confidence_ref  = Args(name="confidence_ref" , type=float, default=0.6 , help=_("Confidence threshold for filtering uncertain pitch values in reference WAV")),
        align_radius    = Args(name="align_radius"   , type=int  , default=1   , help=_("Radius for the FastDTW algorithm; larger radius allows for more flexible alignment but increases computation time")),
        semitone_shift  = Args(name="semitone_shift" , type=int  , default=None, help=_("Semitone shift between the UTAU and reference WAV; if the USTX WAV is an octave higher than the reference WAV, set to 12, otherwise -12; leave it empty to enable automatic shift estimation")),
        smoothness      = Args(name="smoothness"     , type=int  , default=2   , help=_("Smoothness of the expression curve")),
        scaler          = Args(name="scaler"         , type=float, default=2.0 , help=_("Scaling factor for the expression curve")),
    )

    def get_expression(
        self,
        confidence_utau = args.confidence_utau.default,
        confidence_ref  = args.confidence_ref .default,
        align_radius    = args.align_radius   .default,
        semitone_shift  = args.semitone_shift .default,
        smoothness      = args.smoothness     .default,
        scaler          = args.scaler         .default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Extract pitch features from WAV files
        with StreamToLogger(self.logger, tee=True):
            utau_time, utau_pitch, utau_features = get_wav_features(
                wav_path=self.utau_path, confidence_threshold=confidence_utau
            )

        # Extract pitch features from reference WAV file
        with StreamToLogger(self.logger, tee=True):
            ref_time, ref_pitch, ref_features = get_wav_features(
                wav_path=self.ref_path, confidence_threshold=confidence_ref
            )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        pitd_tick, (time_aligned_ref_pitch, *_unused), (unified_utau_pitch, *_unused) = (
            align_sequence_tick(
                query_time=ref_time,
                queries=(ref_pitch, *ref_features),
                reference_time=utau_time,
                references=(utau_pitch, *utau_features),
                tempo=self.tempo,
                align_radius=align_radius,
            )
        )

        # Align pitch sequences in pitch axis
        with StreamToLogger(self.logger, tee=True):
            time_pitch_aligned_ref_pitch, _unused = align_sequence_pitch(
                time_aligned_ref_pitch,
                unified_utau_pitch,
                semitone_shift=semitone_shift,
                smoothness=smoothness,
            )

        # Calculate pitch delta for USTX pitch editing
        pitd_val = get_pitch_delta(
            time_pitch_aligned_ref_pitch,
            unified_utau_pitch,
            scaler=scaler,
        )

        self.expression_tick, self.expression_val = pitd_tick, pitd_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


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


# TODO: Deal with different tempo or ppqn within the same USTX file
def get_wav_features(wav_path, confidence_threshold=0.8, confidence_filter_size=9):
    """Extract features from a WAV file.

    This function extracts pitch and MFCC features from a WAV file, aligning them to a common time base.

    Args:
        wav_path (str): Path to the WAV file.
        confidence_threshold (float, optional): Confidence threshold for pitch detection. Defaults to 0.8.
        confidence_filter_size (int, optional): Size of the median filter for confidence. Defaults to 9.

    Returns:
        tuple: (wav_tick, wav_pitch, wav_features), where:
            - wav_tick (numpy.ndarray): MIDI ticks for the extracted features. Shape: (n_time_points).
            - wav_pitch (numpy.ndarray): Extracted pitch values in Hz. Shape: (n_time_points).
            - wav_features (tuple): Extracted feature sequences. Shape: (n_features, n_time_points).
    """
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []  # List of feature sequences(list of lists)

    # Extract features from WAV file
    time, frequency, confidence = extract_wav_frequency(wav_path, True)
    mask = (
        medfilt(np.array(confidence), kernel_size=confidence_filter_size)
        < confidence_threshold
    )
    (pitch := np.array(frequency))[mask] = np.nan

    pitch_time = time
    feature_times += [pitch_time]
    feature_vals += [pitch]

    # Extract pitch dynamics trends
    pitch_features = seq_dynamics_trends(pitch)
    feature_times += [pitch_time] * len(pitch_features)
    feature_vals += list(pitch_features)

    # Extract MFCC features
    mfcc_time, mfcc = extract_wav_mfcc(wav_path)
    feature_times += [mfcc_time] * len(mfcc)
    feature_vals += list(mfcc)

    # Unified time and features
    wav_time, (wav_pitch, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_pitch, wav_features


def align_sequence_pitch(query, reference, semitone_shift=None, smoothness=0):
    """Align pitch sequences by shifting in semitones and applying smoothing.

    This function adjusts the pitch sequence to match the reference pitch, allowing for optional smoothing.

    Args:
        query (numpy.ndarray): Pitch values to be aligned. Shape: (n_time_points).
        reference (numpy.ndarray): Target reference pitch values. Shape: (n_time_points).
        semitone_shift (int, optional): Number of semitones to shift the query pitch. If None, it is calculated automatically.
        smoothness (int, optional): Smoothing factor for the aligned pitch. Defaults to 0 (no smoothing).

    Returns:
        tuple: (pitch_aligned_query, semitone_shift), where:
            - pitch_aligned_query (numpy.ndarray): Aligned pitch values. Shape: (n_time_points).
            - semitone_shift (int): Applied semitone shift.
    """
    if semitone_shift is None:
        base_pitch_wav = np.nanmedian(query)
        base_pitch_vocal = np.nanmedian(reference)
        semitone_shift = int(np.round(hz_to_midi(base_pitch_vocal)) - np.round(
            hz_to_midi(base_pitch_wav)
        ).astype(int))
        print(_("Estimated Semitone-shift: {}").format(semitone_shift))

    pitch_aligned_query = query * np.exp2(semitone_shift / 12)

    pitch_aligned_query = gaussian_filter1d_with_nan(
        pitch_aligned_query, sigma=smoothness
    )

    return pitch_aligned_query, semitone_shift


def get_pitch_delta(query, reference, scaler=2.5):
    """Calculate the difference between two pitch sequences.

    The delta represents the pitch correction needed to align the query sequence with the reference sequence.

    Args:
        query (numpy.ndarray): Pitch values from the query sequence.
        reference (numpy.ndarray): Pitch values from the reference sequence.
        scaler (float, optional): Scaling factor for the pitch difference. Defaults to 2.5.

    Returns:
        numpy.ndarray: Scaled pitch difference values.
    """
    return scaler * (query - reference)


def extract_wav_frequency(file_path, use_cache=True):
    """Extract pitch frequency from a WAV file using Crepe.

    This function processes an audio file to extract pitch information using the
    CREPE algorithm. It supports caching to improve performance when processing
    the same file multiple times.

    Args:
        file_path (str): Path to the WAV file.
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (list of float): Time points in seconds. Shape: (n_time_points).
            - frequency (list of float): Detected pitch frequencies in Hz. Shape: (n_time_points).
            - confidence (list of float): Confidence values for the detected pitches. Shape: (n_time_points).
    """
    time = []
    frequency = []
    confidence = []
    cache_dir = Path(CACHE_DIR) / "pitd"
    # Try reading data from cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        wav_hash = calculate_file_hash(file_path)

        cache_path = cache_dir / f"{wav_hash}.csv"
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
        sr, audio = wavfile.read(file_path)
        time, frequency, confidence, _unused = crepe.predict(audio, sr, viterbi=True)

        # Save data to cache
        if use_cache:
            with open(cache_path, mode="w+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Frequency (Hz)", "Confidence"])
                for t, f, c in zip(time, frequency, confidence):
                    writer.writerow([t, f, c])
            print(_("F0 data saved to cache file: '{}'").format(cache_path))

    return time, frequency, confidence
