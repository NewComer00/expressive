from types import SimpleNamespace

import numpy as np
from scipy.stats import zscore
from skimage.filters import threshold_otsu

from .base import (
    Args,
    Plot,
    ExpressionLoader,
    register_expression,
)
from utils.wavtool import (
    extract_wav_rms,
    extract_wav_breath_voice,
)
from utils.seqtool import (
    unify_sequence_time,
    align_sequence_tick,
    seq_dynamics_trends,
    seq_spline_smoothing,
    gaussian_filter1d_with_nan,
)
from utils.i18n import _, _l
from utils.log import StreamToLogger


@register_expression
class VoicLoader(ExpressionLoader):
    expression_name = "voic"
    expression_info = _l("Voicing (curve)")
    args = SimpleNamespace(
        align_radius     = Args(name="align_radius"    , type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        smoothness       = Args(name="smoothness"      , type=int  , default=4   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        dynamic_range    = Args(name="dynamic_range"   , type=float, default=1.0 , help=_l("**Dynamic range** of the expression curve. Values >1 amplify the variation, =1 keeps original range, <1 compresses it")),  # noqa: E501
        bias             = Args(name="bias"            , type=int  , default=-10   , help=_l("**Bias** offset added to the expression curve. Positive values shift the curve upward; negative values shift it downward")),  # noqa: E501
        spline_smoothing = Args(name="spline_smoothing", type=bool , default=True, help=_l("Perform **spline smoothing** on the final expression curve for extra smoothness")),  # noqa: E501
    )
    plots = SimpleNamespace(
        expression          = Plot(tag=expression_info           , title=expression_info           , x_label=_l("Tick")    , y_label=expression_name   , legends=[expression_name]            ),  # noqa: E501
        raw_voice_index     = Plot(tag=_l("raw_voice_index")     , title=_l("Raw Voice Index")     , x_label=_l("Time (s)"), y_label=_l("Voice Index") , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
        aligned_voice_index = Plot(tag=_l("aligned_voice_index") , title=_l("Aligned Voice Index") , x_label=_l("Tick")    , y_label=_l("Voice Index") , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
    )

    def get_expression(
        self,
        align_radius     = args.align_radius    .default,
        smoothness       = args.smoothness      .default,
        dynamic_range    = args.dynamic_range   .default,
        bias             = args.bias            .default,
        spline_smoothing = args.spline_smoothing.default,
    ):
        self.logger.info(_("Extracting expression..."))

        with StreamToLogger(self.logger, tee=False):
            utau_time, utau_bi, utau_vi, utau_features = \
                get_wav_features(wav_path=self.utau_path)
            ref_time, ref_bi, ref_vi, ref_features = \
                get_wav_features(wav_path=self.ref_path)

        (
            voic_tick,
            (time_aligned_ref_bi, time_aligned_ref_vi, *_unused),
            (time_unified_utau_bi, time_unified_utau_vi, *_unused),
        ) = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_bi, ref_vi, *ref_features),
            reference_time=utau_time,
            references=(utau_bi, utau_vi, *utau_features),
            align_radius=align_radius,
        )

        voic_val = get_expression_voicing(time_aligned_ref_vi, smoothness, dynamic_range, bias)

        if spline_smoothing:
            voic_val = seq_spline_smoothing(voic_tick, voic_val, nan_policy='preserve_head_tail')

        self.collect_plot(self.plots.expression,          (voic_tick, voic_val))
        self.collect_plot(self.plots.raw_voice_index,     (ref_time, ref_vi),  (utau_time, utau_vi))
        self.collect_plot(self.plots.aligned_voice_index, (voic_tick, time_aligned_ref_vi), (voic_tick, time_unified_utau_vi))

        self.expression_tick, self.expression_val = voic_tick, voic_val
        self.logger.info(_("Expression extraction complete."))
        return self.expression_tick, self.expression_val


def get_wav_features(wav_path):
    feature_times = []
    feature_vals  = []

    time, bi, vi = extract_wav_breath_voice(wav_path)
    feature_times += [time, time]
    feature_vals  += [bi, vi]

    bi_trends = seq_dynamics_trends(bi)
    feature_times += [time] * len(bi_trends)
    feature_vals  += list(bi_trends)

    rms_time, rms = extract_wav_rms(wav_path, mask_silence=True)
    feature_times += [rms_time]
    feature_vals  += [rms]

    rms_trends = seq_dynamics_trends(rms)
    feature_times += [rms_time] * len(rms_trends)
    feature_vals  += list(rms_trends)

    wav_time, (wav_bi, wav_vi, *wav_features) = unify_sequence_time(
        seq_times=feature_times, seq_vals=feature_vals
    )
    return wav_time, wav_bi, wav_vi, wav_features


def get_expression_voicing(voice_index, smoothness=4, dynamic_range=1.0, bias=-10):
    base_scaler = 10.0
    base_bias = 100.0
    voice_index = voice_index.copy()

    valid = np.isfinite(voice_index)
    if not valid.any():
        return np.zeros_like(voice_index)

    thresh = threshold_otsu(voice_index[valid])
    voice_index[~valid | (voice_index < thresh)] = np.nan

    smoothed_voic = gaussian_filter1d_with_nan(
        base_bias + dynamic_range * base_scaler * zscore(voice_index, nan_policy='omit'),
        sigma=smoothness,
    )
    return smoothed_voic + bias
