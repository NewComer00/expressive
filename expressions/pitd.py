from types import SimpleNamespace

import numpy as np
from scipy.signal import medfilt
from librosa import hz_to_midi

from .base import (
    Args,
    Plot,
    ExpressionLoader,
    register_expression
)
from utils.i18n import _, _l, _lf
from utils.seqtool import (
    seq_spline_smoothing,
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)
from utils.log import StreamToLogger
from utils.wavtool import extract_wav_mfcc, extract_wav_frequency, extract_wav_rms


@register_expression
class PitdLoader(ExpressionLoader):
    expression_name = "pitd"
    expression_info = _l("Pitch Deviation (curve)")
    backend_choices = {
        "rmvpe-onnx": _l("finest accuracy, fast, CPU only (ONNX Runtime)"),
        "swift-f0": _l("fair accuracy, fastest, CPU only (ONNX Runtime)"),
        "crepe": _l("good accuracy, slow, CPU & NVIDIA GPU (TensorFlow)"),
        "hybrid": _l("based on rmvpe-onnx, improved by swift-f0, CPU only (ONNX Runtime)"),
    }
    confidence_utau_recommended = {"rmvpe-onnx": 0.03, "swift-f0": 0.95, "crepe": 0.80, "hybrid": 0.03}
    confidence_ref_recommended  = {"rmvpe-onnx": 0.03, "swift-f0": 0.93, "crepe": 0.60, "hybrid": 0.03}
    args = SimpleNamespace(
        backend          = Args(name="backend"         , type=str  , default="rmvpe-onnx", choices=list(backend_choices.keys()), help=_lf("**F0 detection backend** for extracting pitch from WAV files. Available options:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.backend_choices.items()]))),  # noqa: E501
        confidence_utau  = Args(name="confidence_utau" , type=float, default=None, help=_lf("Minimum **confidence level** for keeping detected pitch values in the **UTAU** WAV. Lower values retain more frames but may include errors. Omit to use the recommended value for the selected backend:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.confidence_utau_recommended.items()]))),  # noqa: E501
        confidence_ref   = Args(name="confidence_ref"  , type=float, default=None, help=_lf("Minimum **confidence level** for keeping detected pitch values in the **reference** WAV. Lower values retain more frames but may include errors. Omit to use the recommended value for the selected backend:\n\n%s\n\n", lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.confidence_ref_recommended.items()]))),  # noqa: E501
        align_radius     = Args(name="align_radius"    , type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        semitone_shift   = Args(name="semitone_shift"  , type=int  , default=None, help=_l("**Semitone shift** between the UTAU and reference WAV. If the UTAU WAV is an octave higher than the reference WAV, set to 12; if lower, set to -12. Omit to enable automatic shift estimation")),  # noqa: E501
        smoothness       = Args(name="smoothness"      , type=int  , default=2   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler           = Args(name="scaler"          , type=float, default=1.0 , help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
        spline_smoothing = Args(name="spline_smoothing", type=bool , default=True, help=_l("Perform **spline smoothing** on the final expression curve for extra smoothness")),  # noqa: E501
    )
    plots = SimpleNamespace(
        expression    = Plot(tag=expression_info    , title=expression_info                   , x_label=_l("Tick")    , y_label=expression_name , legends=[expression_name]            ),  # noqa: E501
        confidence    = Plot(tag=_l("confidence")   , title=_l("Pitch Extraction Confidence") , x_label=_l("Time (s)"), y_label=_l("Confidence"), legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
        raw_pitch     = Plot(tag=_l("raw_pitch")    , title=_l("Raw Pitch")                   , x_label=_l("Time (s)"), y_label=_l("Pitch (Hz)"), legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
        aligned_pitch = Plot(tag=_l("aligned_pitch"), title=_l("Aligned Pitch")               , x_label=_l("Tick")    , y_label=_l("Pitch (Hz)"), legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
    )

    def get_expression(
        self,
        backend          = args.backend         .default,
        confidence_utau  = args.confidence_utau .default,
        confidence_ref   = args.confidence_ref  .default,
        align_radius     = args.align_radius    .default,
        semitone_shift   = args.semitone_shift  .default,
        smoothness       = args.smoothness      .default,
        scaler           = args.scaler          .default,
        spline_smoothing = args.spline_smoothing.default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Resolve per-backend confidence defaults
        if confidence_utau is None:
            confidence_utau = self.__class__.confidence_utau_recommended[backend]
        if confidence_ref is None:
            confidence_ref = self.__class__.confidence_ref_recommended[backend]

        # Extract pitch features from WAV files
        with StreamToLogger(self.logger, tee=True):
            utau_time, utau_pitch, utau_confidence, utau_features = get_wav_features(
                wav_path=self.utau_path, confidence_threshold=confidence_utau, backend=backend
            )
            ref_time, ref_pitch, ref_confidence, ref_features = get_wav_features(
                wav_path=self.ref_path, confidence_threshold=confidence_ref, backend=backend
            )

        # Align all sequences to a common MIDI tick time base.
        # Features from the UTAU WAV are the reference; Ref. WAV features are the query.
        pitd_tick, (time_aligned_ref_pitch, *_unused), (unified_utau_pitch, *_unused) = (
            align_sequence_tick(
                query_time=ref_time,
                queries=(ref_pitch, ref_confidence, *ref_features),
                reference_time=utau_time,
                references=(utau_pitch, utau_confidence, *utau_features),
                align_radius=align_radius,
            )
        )

        # Align pitch sequences along the pitch axis
        with StreamToLogger(self.logger, tee=True):
            time_pitch_aligned_ref_pitch, _unused = align_sequence_pitch(
                time_aligned_ref_pitch,
                unified_utau_pitch,
                semitone_shift=semitone_shift,
            )

        # Calculate pitch delta for USTX pitch editing
        pitd_val = get_pitch_delta(
            time_pitch_aligned_ref_pitch,
            unified_utau_pitch,
            smoothness=smoothness,
            scaler=scaler,
        )

        if spline_smoothing:
            # Final spline smoothing of the expression curve
            # NOTE: All NaN positions except the leading and trailing ones will be interpolated
            # Only preserving NaN at the head/tail to avoid edge artifacts of spline smoothing
            pitd_val = seq_spline_smoothing(pitd_tick, pitd_val, nan_policy='preserve_head_tail')

        # Collect plots
        self.collect_plot(self.plots.expression,    (pitd_tick, pitd_val))
        self.collect_plot(self.plots.confidence,    (ref_time,  ref_confidence), (utau_time, utau_confidence))
        self.collect_plot(self.plots.raw_pitch,     (ref_time,  ref_pitch), (utau_time, utau_pitch))
        self.collect_plot(self.plots.aligned_pitch, (pitd_tick, time_pitch_aligned_ref_pitch), (pitd_tick, unified_utau_pitch))

        self.expression_tick, self.expression_val = pitd_tick, pitd_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def get_wav_features(wav_path, backend="rmvpe-onnx", confidence_threshold=0.8, confidence_filter_size=9):
    """Extract features from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        backend (str, optional): F0 detection backend ("crepe" or "swift-f0" or "rmvpe-onnx"). Defaults to "rmvpe-onnx".
        confidence_threshold (float, optional): Confidence threshold for pitch detection. Defaults to 0.8.
        confidence_filter_size (int, optional): Size of the median filter for confidence. Defaults to 9.

    Returns:
        tuple: (wav_time, wav_pitch, wav_confidence, wav_features)
    """
    feature_times = []
    feature_vals  = []

    time, frequency, confidence = extract_wav_frequency(wav_path, backend=backend)

    mask_confidence = (
        medfilt(confidence, kernel_size=confidence_filter_size)
        < confidence_threshold
    )
    (pitch := frequency)[mask_confidence] = np.nan

    pitch_time = time
    feature_times += [pitch_time]
    feature_vals  += [pitch]

    feature_times += [pitch_time]
    feature_vals  += [confidence]

    pitch_features = seq_dynamics_trends(pitch)
    feature_times += [pitch_time] * len(pitch_features)
    feature_vals  += list(pitch_features)

    mfcc_time, mfcc = extract_wav_mfcc(wav_path)
    feature_times += [mfcc_time] * len(mfcc)
    feature_vals  += list(mfcc)

    rms_time, rms = extract_wav_rms(wav_path, mask_silence=True)
    feature_times += [rms_time]
    feature_vals  += [rms]

    rms_dynamics_trends = seq_dynamics_trends(rms)
    feature_times += [rms_time] * len(rms_dynamics_trends)
    feature_vals  += list(rms_dynamics_trends)

    wav_time, (wav_pitch, wav_confidence, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_pitch, wav_confidence, wav_features


def align_sequence_pitch(query, reference, semitone_shift=None):
    """Align pitch sequences by shifting in semitones and applying smoothing.

    Args:
        query (numpy.ndarray):          Pitch values to be aligned.
        reference (numpy.ndarray):      Target reference pitch values.
        semitone_shift (int, optional): Semitones to shift the query pitch.
                                        If None, estimated automatically.

    Returns:
        tuple: (pitch_aligned_query, semitone_shift)
    """
    if semitone_shift is None:
        base_pitch_wav   = np.nanmedian(query)
        base_pitch_vocal = np.nanmedian(reference)
        semitone_shift   = int(
            np.round(hz_to_midi(base_pitch_vocal))
            - np.round(hz_to_midi(base_pitch_wav)).astype(int)
        )
        print(_("Estimated Semitone-shift: {}").format(semitone_shift))

    pitch_aligned_query = query * np.exp2(semitone_shift / 12)
    return pitch_aligned_query, semitone_shift


def get_pitch_delta(query, reference, smoothness=2, scaler=1.0):
    """Calculate the scaled pitch difference between two sequences.

    PITD is expressed in cents (100 cents = 1 semitone).
    The renderer applies PITD on top of the base pitch, so the delta
    must be in the same unit OpenUtau expects for the PITD expression.

    Args:
        query (numpy.ndarray):       Pitch values from the query sequence.
        reference (numpy.ndarray):   Pitch values from the reference sequence.
        smoothness (int, optional):  Smoothing sigma. Defaults to 2.
        scaler (float, optional):    Scaling factor. Defaults to 1.0.

    Returns:
        numpy.ndarray: Scaled pitch difference in cents, preserving NaN for unvoiced frames.
    """
    voiced = (query > 0) & (reference > 0)

    delta = np.full_like(query, fill_value=np.nan)
    delta[voiced] = 1200.0 * np.log2(query[voiced] / reference[voiced])

    delta = gaussian_filter1d_with_nan(delta, sigma=smoothness)
    return scaler * delta
