from itertools import accumulate
from types import SimpleNamespace

import librosa
import numpy as np
from scipy.stats import zscore

from .base import Args, ExpressionLoader, register_expression
from utils.i18n import _
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
)


@register_expression
class DynLoader(ExpressionLoader):
    expression_name = "dyn"
    args = SimpleNamespace(
        align_radius    = Args(name="align_radius",    type=int,   default=1,    help=_("Radius for the FastDTW algorithm; larger radius allows for more flexible alignment but increases computation time")),
        smoothness      = Args(name="smoothness",      type=int,   default=2,    help=_("Smoothness of the expression curve")),
        scaler          = Args(name="scaler",          type=float, default=2.0,  help=_("Scaling factor for the expression curve")),
    )

    def get_expression(self, align_radius=1, smoothness=2, scaler=2.0):
        # Extract rms features from WAV files
        utau_time, utau_rms, utau_features = get_wav_features(
            wav_path=self.utau_path,
        )
        ref_time, ref_rms, ref_features = get_wav_features(
            wav_path=self.ref_path,
        )

        # Align all sequences to a common MIDI tick time base
        # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
        dyn_tick, (time_aligned_ref_rms, *_), *_ = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_rms, *ref_features),
            reference_time=utau_time,
            references=(utau_rms, *utau_features),
            tempo=self.tempo,
            align_radius=align_radius,
        )

        dyn_val = get_experssion_dynamics(time_aligned_ref_rms, smoothness, scaler)

        self.expression_tick, self.expression_val = dyn_tick, dyn_val
        return self.expression_tick, self.expression_val


def extract_wav_rms(wav_path):
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_time = librosa.times_like(rms, sr=sr)
    return rms_time, rms


def extract_rms_dynamics_trends(rms, n_order=3):
    rms_grads = list(accumulate([rms] * n_order, lambda x, _: np.gradient(x)))
    rms_grads = np.vstack(rms_grads)

    rms_trends = list(accumulate([rms] * n_order, lambda x, _: np.cumsum(x)))
    rms_trends = np.vstack(rms_trends)
    return np.vstack([rms_grads, rms_trends])


def get_wav_features(wav_path):
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []  # List of feature sequences(list of lists)

    # Extract RMS feature
    rms_time, rms = extract_wav_rms(wav_path)
    feature_times += [rms_time]
    feature_vals += [rms]

    # Extract RMS dynamics and trends
    rms_dynamics_trends = extract_rms_dynamics_trends(rms)
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
