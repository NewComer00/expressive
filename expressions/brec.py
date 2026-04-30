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
class BrecLoader(ExpressionLoader):
    expression_name = "brec"
    expression_info = _l("Breathiness (curve)")
    args = SimpleNamespace(
        align_radius     = Args(name="align_radius"    , type=int  , default=1   , help=_l("**Radius** for the FastDTW alignment algorithm; larger values allow more flexible alignment but increase computation time")),  # noqa: E501
        smoothness       = Args(name="smoothness"      , type=int  , default=4   , help=_l("Controls the **smoothness** of the expression curve using Gaussian filtering. Higher values produce smoother curves but may lose fine detail")),  # noqa: E501
        scaler           = Args(name="scaler"          , type=float, default=1.0 , help=_l("**Scaling factor** applied to the expression curve. Values >1 amplify the expression, =1 keeps original intensity, <1 reduces it")),  # noqa: E501
        bias             = Args(name="bias"            , type=int  , default=10   , help=_l("**Bias** offset added to the expression curve. Positive values shift the curve upward; negative values shift it downward")),  # noqa: E501
        spline_smoothing = Args(name="spline_smoothing", type=bool , default=True, help=_l("Perform **spline smoothing** on the final expression curve for extra smoothness")),  # noqa: E501
    )
    plots = SimpleNamespace(
        expression           = Plot(tag=expression_info            , title=expression_info            , x_label=_l("Tick")    , y_label=expression_name    , legends=[expression_name]            ),  # noqa: E501
        raw_breath_index     = Plot(tag=_l("raw_breath_index")     , title=_l("Raw Breath Index")     , x_label=_l("Time (s)"), y_label=_l("Breath Index")  , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
        aligned_breath_index = Plot(tag=_l("aligned_breath_index") , title=_l("Aligned Breath Index") , x_label=_l("Tick")    , y_label=_l("Breath Index")  , legends=[_l("Reference"), _l("UTAU")]),  # noqa: E501
    )

    def get_expression(
        self,
        align_radius     = args.align_radius    .default,
        smoothness       = args.smoothness      .default,
        scaler           = args.scaler          .default,
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
            brec_tick,
            (time_aligned_ref_bi, time_aligned_ref_vi, *_unused),
            (time_unified_utau_bi, time_unified_utau_vi, *_unused),
        ) = align_sequence_tick(
            query_time=ref_time,
            queries=(ref_bi, ref_vi, *ref_features),
            reference_time=utau_time,
            references=(utau_bi, utau_vi, *utau_features),
            align_radius=align_radius,
        )

        brec_val = get_expression_breathiness(time_aligned_ref_bi, time_aligned_ref_vi, smoothness, scaler, bias)

        if spline_smoothing:
            brec_val = seq_spline_smoothing(brec_tick, brec_val, nan_policy='preserve_head_tail')

        self.collect_plot(self.plots.expression,           (brec_tick, brec_val))
        self.collect_plot(self.plots.raw_breath_index,     (ref_time, ref_bi),  (utau_time, utau_bi))
        self.collect_plot(self.plots.aligned_breath_index, (brec_tick, time_aligned_ref_bi), (brec_tick, time_unified_utau_bi))

        self.expression_tick, self.expression_val = brec_tick, brec_val
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


def get_expression_breathiness(breath_index, voice_index, smoothness=4, scaler=1.0, bias=10):
    base_scaler = 10.0
    breath_index = breath_index.copy()

    valid = np.isfinite(voice_index)
    if valid.any():
        thresh = threshold_otsu(voice_index[valid])
        breath_index[~valid | (voice_index < thresh)] = np.nan
    else:
        breath_index[:] = np.nan

    smoothed_brec = gaussian_filter1d_with_nan(
        base_scaler * zscore(breath_index, nan_policy='omit'),
        sigma=smoothness,
    )
    return scaler * smoothed_brec + bias
