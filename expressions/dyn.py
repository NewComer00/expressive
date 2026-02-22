from types import SimpleNamespace

import librosa
from scipy.stats import zscore

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _, _l
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)


@register_expression
class DynLoader(ExpressionLoader):
    expression_name = "dyn"
    expression_info = _l("Dynamics (curve)")
    args = SimpleNamespace(
        align_radius    = Args(name="align_radius", type=int  , default=1  , help=_l("Radius for the FastDTW algorithm; larger radius allows for more flexible alignment but increases computation time")),  # noqa: E501
        smoothness      = Args(name="smoothness"  , type=int  , default=2  , help=_l("Smoothness of the expression curve")),
        scaler          = Args(name="scaler"      , type=float, default=1.5, help=_l("Scaling factor for the expression curve")),  # noqa: E501
    )

    def get_expression(
        self,
        align_radius = args.align_radius.default,
        smoothness   = args.smoothness  .default,
        scaler       = args.scaler      .default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Extract rms features from WAV files
        utau_time, utau_rms, utau_features = get_wav_features(
            wav_path=self.utau_path,
        )
        ref_time, ref_rms, ref_features = get_wav_features(
            wav_path=self.ref_path,
        )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        dyn_tick, (time_aligned_ref_rms, *_unused), *_unused = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_rms, *ref_features),
            reference_time=utau_time,
            references=(utau_rms, *utau_features),
            tempo=self.tempo,
            align_radius=align_radius,
        )

        dyn_val = get_experssion_dynamics(time_aligned_ref_rms, smoothness, scaler)

        self.expression_tick, self.expression_val = dyn_tick, dyn_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def extract_wav_rms(wav_path):
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_time = librosa.times_like(rms, sr=sr)
    return rms_time, rms


def get_wav_features(wav_path):
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []  # List of feature sequences(list of lists)

    # Extract RMS feature
    rms_time, rms = extract_wav_rms(wav_path)
    feature_times += [rms_time]
    feature_vals += [rms]

    # Extract RMS dynamics and trends
    rms_dynamics_trends = seq_dynamics_trends(rms)
    feature_times += [rms_time] * len(rms_dynamics_trends)
    feature_vals += list(rms_dynamics_trends)

    # Unified time and features
    wav_time, (wav_rms, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_rms, wav_features


def get_experssion_dynamics(time_aligned_rms, smoothness=2, scaler=1.0):
    base_scaler = 10.0
    smoothed_dyn = gaussian_filter1d_with_nan(
        base_scaler * zscore(time_aligned_rms),
        sigma=smoothness,
    )
    return scaler * smoothed_dyn
