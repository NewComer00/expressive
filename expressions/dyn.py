from types import SimpleNamespace

import numpy as np
from scipy.stats import zscore

from .base import (
    Args,
    Plot,
    ExpressionLoader,
    register_expression
)
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    gaussian_filter1d_with_nan,
    seq_dynamics_trends,
)
from utils.i18n import _, _l
from utils.wavtool import extract_wav_rms


@register_expression
class DynLoader(ExpressionLoader):
    expression_name = "dyn"
    expression_info = _l("Dynamics (curve)")
    args = SimpleNamespace(
        trim_silence    = Args(name="trim_silence", type=bool , default=True, help=_l("**Trim silence** from the leading and trailing edges of the audio before extracting expression")),  # noqa: E501
        align_radius    = Args(name="align_radius", type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        smoothness      = Args(name="smoothness"  , type=int  , default=2   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler          = Args(name="scaler"      , type=float, default=1.5 , help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
    )
    plots = SimpleNamespace(
        expression  = Plot(tag=expression_info    , title=expression_info    , x_label=_l("Tick")    , y_label=expression_name, legends=[expression_name]            ),  # noqa: E501
        raw_rms     = Plot(tag=_l("raw_rms")      , title=_l("Raw RMS")      , x_label=_l("Time (s)"), y_label=_l("RMS")      , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
        aligned_rms = Plot(tag=_l("aligned_rms")  , title=_l("Aligned RMS")  , x_label=_l("Tick")    , y_label=_l("RMS")      , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
    )

    def get_expression(
        self,
        trim_silence = args.trim_silence.default,
        align_radius = args.align_radius.default,
        smoothness   = args.smoothness  .default,
        scaler       = args.scaler      .default,
    ):
        self.logger.info(_("Extracting expression..."))

        # Extract RMS features from WAV files
        utau_time, utau_rms, utau_features = get_wav_features(
            wav_path=self.utau_path, mask_silence=trim_silence
        )
        ref_time, ref_rms, ref_features = get_wav_features(
            wav_path=self.ref_path, mask_silence=trim_silence
        )

        # Align all sequences to a common MIDI tick time base.
        # Features from the UTAU WAV are the reference; Ref. WAV features are the query.
        dyn_tick, (time_aligned_ref_rms, *_unused), (time_unified_utau_rms, *_unused) = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_rms, *ref_features),
            reference_time=utau_time,
            references=(utau_rms, *utau_features),
            align_radius=align_radius,
        )

        # Mask positions where UTAU is silent (NaN)
        time_aligned_ref_rms[np.isnan(time_unified_utau_rms)] = np.nan

        # Generate expression curve
        dyn_val = get_experssion_dynamics(time_aligned_ref_rms, smoothness, scaler)

        # Collect plots
        self.collect_plot(self.plots.expression,  (dyn_tick, dyn_val))
        self.collect_plot(self.plots.raw_rms,     (ref_time,  ref_rms), (utau_time, utau_rms))
        self.collect_plot(self.plots.aligned_rms, (dyn_tick, time_aligned_ref_rms), (dyn_tick,  time_unified_utau_rms))

        self.expression_tick, self.expression_val = dyn_tick, dyn_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def get_wav_features(wav_path, mask_silence=True):
    feature_times = []
    feature_vals  = []

    # Extract RMS feature
    rms_time, rms = extract_wav_rms(wav_path, mask_silence=mask_silence)
    feature_times += [rms_time]
    feature_vals  += [rms]

    # Extract RMS dynamics and trends
    rms_dynamics_trends = seq_dynamics_trends(rms)
    feature_times += [rms_time] * len(rms_dynamics_trends)
    feature_vals  += list(rms_dynamics_trends)

    # Unify time and features
    wav_time, (wav_rms, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_rms, wav_features


def get_experssion_dynamics(time_aligned_rms, smoothness=2, scaler=1.0):
    base_scaler = 10.0
    smoothed_dyn = gaussian_filter1d_with_nan(
        base_scaler * zscore(time_aligned_rms, nan_policy='omit'),
        sigma=smoothness,
    )
    return scaler * smoothed_dyn
