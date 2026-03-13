from types import SimpleNamespace

import numpy as np
from scipy.signal import medfilt
from librosa import hz_to_midi

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _, _l, _lf
from utils.seqtool import (
    time_to_ticks,
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)
from utils.log import StreamToLogger
from utils.wavtool import extract_wav_mfcc, extract_wav_frequency


@register_expression
class PitdLoader(ExpressionLoader):
    expression_name = "pitd"
    expression_info = _l("Pitch Deviation (curve)")
    backend_choices = {
        "swift-f0": _l("fast, CPU-based (ONNX Runtime)"),
        "crepe": _l("classic but slow, CPU & NVIDIA GPU (TensorFlow)"),
    }
    confidence_utau_recommended = {"swift-f0": 0.95, "crepe": 0.8}
    confidence_ref_recommended = {"swift-f0": 0.93, "crepe": 0.6}
    args = SimpleNamespace(
        backend         = Args(name="backend"        , type=str  , default="swift-f0", choices=list(backend_choices.keys()), help=_lf("**F0 detection backend** for extracting pitch from WAV files. Available options:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.backend_choices.items()]))),  # noqa: E501
        confidence_utau = Args(name="confidence_utau", type=float, default=None, help=_lf("Minimum **confidence level** for keeping detected pitch values in the **UTAU** WAV. Lower values retain more frames but may include errors. Omit to use the recommended value for the selected backend:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.confidence_utau_recommended.items()]))),  # noqa: E501
        confidence_ref  = Args(name="confidence_ref" , type=float, default=None, help=_lf("Minimum **confidence level** for keeping detected pitch values in the **reference** WAV. Lower values retain more frames but may include errors. Omit to use the recommended value for the selected backend:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.confidence_ref_recommended.items()]))),  # noqa: E501
        align_radius    = Args(name="align_radius"   , type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        semitone_shift  = Args(name="semitone_shift" , type=int  , default=None, help=_l("**Semitone shift** between the UTAU and reference WAV. If the UTAU WAV is an octave higher than the reference WAV, set to 12; if lower, set to -12. Omit to enable automatic shift estimation")),  # noqa: E501
        smoothness      = Args(name="smoothness"     , type=int  , default=2   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler          = Args(name="scaler"         , type=float, default=2.0 , help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
    )

    def get_expression(
        self,
        backend         = args.backend        .default,
        confidence_utau = args.confidence_utau.default,
        confidence_ref  = args.confidence_ref .default,
        align_radius    = args.align_radius   .default,
        semitone_shift  = args.semitone_shift .default,
        smoothness      = args.smoothness     .default,
        scaler          = args.scaler         .default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Resolve per-backend confidence defaults
        if confidence_utau is None:
            confidence_utau = self.__class__.confidence_utau_recommended[backend]
        if confidence_ref is None:
            confidence_ref = self.__class__.confidence_ref_recommended[backend]

        # Extract pitch features from WAV files
        with StreamToLogger(self.logger, tee=True):
            utau_time, utau_pitch, utau_features = get_wav_features(
                wav_path=self.utau_path, confidence_threshold=confidence_utau, backend=backend
            )

        # Extract pitch features from reference WAV file
        with StreamToLogger(self.logger, tee=True):
            ref_time, ref_pitch, ref_features = get_wav_features(
                wav_path=self.ref_path, confidence_threshold=confidence_ref, backend=backend
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

        # Shift ticks to absolute MIDI position using the UTAU trim offset
        utau_offset_ticks = time_to_ticks(self.utau_offset, self.tempo)
        self.expression_tick = pitd_tick + utau_offset_ticks
        self.expression_val  = pitd_val

        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


# TODO: Deal with different tempo or ppqn within the same USTX file
def get_wav_features(wav_path, backend="swift-f0", confidence_threshold=0.8, confidence_filter_size=9):
    """Extract features from a WAV file.

    This function extracts pitch and MFCC features from a WAV file, aligning them to a common time base.

    Args:
        wav_path (str): Path to the WAV file.
        backend (str, optional): F0 detection backend ("crepe" or "swift-f0"). Defaults to "swift-f0".
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
    time, frequency, confidence = extract_wav_frequency(wav_path, backend=backend)
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
