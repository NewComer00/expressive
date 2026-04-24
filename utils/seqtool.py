from itertools import accumulate
from typing import Callable

import numpy as np
from fastdtw import fastdtw  # type: ignore
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore


# ---------------------------------------------------------------------------
# Tick converter registry
# ---------------------------------------------------------------------------
# Defaults raise clearly if to_ticks=True is used before registration.
# Call set_tick_converters() once at startup to wire up tempo-map-aware ones,
# e.g.:
#     axis = editor.build_time_axis()
#     set_tick_converters(axis.seconds_to_ticks, axis.ticks_to_seconds)

def _default_time_to_ticks(time: np.ndarray, unique: bool = True) -> np.ndarray:
    raise RuntimeError(
        "No tick converter registered. "
        "Call set_tick_converters() before using to_ticks=True."
    )

def _default_ticks_to_time(ticks: np.ndarray) -> np.ndarray:
    raise RuntimeError(
        "No tick converter registered. "
        "Call set_tick_converters() before using to_ticks=True."
    )

_time_to_ticks_fn: Callable = _default_time_to_ticks
_ticks_to_time_fn: Callable = _default_ticks_to_time


def set_tick_converters(
    time_to_ticks_fn: Callable[[np.ndarray], np.ndarray],
    ticks_to_time_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Register tempo-map-aware tick converters for this module.

    Must be called before any function that uses ``to_ticks=True``.

    Args:
        time_to_ticks_fn:   ``(times: ndarray) -> ticks: ndarray``
        ticks_to_time_fn:   ``(ticks: ndarray) -> times: ndarray``
    """
    global _time_to_ticks_fn, _ticks_to_time_fn
    _time_to_ticks_fn = time_to_ticks_fn
    _ticks_to_time_fn = ticks_to_time_fn


def reset_tick_converters() -> None:
    """Restore the default (error-raising) tick converters."""
    global _time_to_ticks_fn, _ticks_to_time_fn
    _time_to_ticks_fn = _default_time_to_ticks
    _ticks_to_time_fn = _default_ticks_to_time


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def sequence_interval_intersection(seqs):
    """Find the intersection of multiple sequences.

    This function finds the intersection of multiple sequences, returning a sorted list of unique values.

    Args:
        seqs (list of list): List of sequences to intersect.

    Returns:
        list: Sorted list of unique values in the intersection.

    Example:
        seqs = [[0, 1, 2, 3], [1., 1.1, 2., 4., 5.]]
        result = [1., 1.1, 2., 3.]
    """
    min_val = max(min(s) for s in seqs)  # Highest lower bound
    max_val = min(max(s) for s in seqs)  # Lowest upper bound
    return [x for x in np.unique(np.concatenate(seqs)) if min_val <= x <= max_val]


def sequence_interval_union(seqs):
    """Find the union of multiple sequences.

    This function finds the union of multiple sequences, returning a sorted list of unique values.

    Args:
        seqs (list of list): List of sequences to unite.

    Returns:
        list: Sorted list of unique values in the union.

    Example:
        seqs = [[0, 1, 2, 3], [1., 1.1, 2., 4., 5.]]
        result = [0., 1., 1.1, 2., 3., 4., 5.]
    """
    return np.unique(np.concatenate(seqs)).tolist()


def unify_sequence_time(seq_times, seq_vals, to_ticks=False):
    """Unify multiple sequences to a common time base.

    Aligns multiple sequences to a common time base by interpolating values.
    When ``to_ticks=True``, uses the converters registered via
    :func:`set_tick_converters`.

    Args:
        seq_times (list of array-like): List of time sequences. Shape: (n_sequences, n_time_points).
        seq_vals (list of array-like):  List of value sequences. Shape: (n_sequences, n_time_points).
        to_ticks (bool, optional):      Convert unified time to MIDI ticks. Defaults to ``False``.

    Returns:
        tuple: ``(unified_time, unified_seqs)`` where

        - **unified_time** (*ndarray*): Unified time points (seconds or ticks).
        - **unified_seqs** (*tuple of ndarray*): Interpolated sequences.
    """
    unified_seq_time = np.asarray(sequence_interval_union(seq_times))

    if not to_ticks:
        unified_seq_time = np.unique(unified_seq_time)
        unified_seqs_val = [
            interp1d(st, sv, fill_value=np.nan, bounds_error=False)(unified_seq_time)  # type: ignore
            for (st, sv) in zip(seq_times, seq_vals, strict=False)
        ]
        return unified_seq_time, tuple(unified_seqs_val)

    unified_seq_ticks = np.unique(_time_to_ticks_fn(unified_seq_time))
    time_mapping = _ticks_to_time_fn(unified_seq_ticks)
    unified_seqs_val = [
        interp1d(st, sv, fill_value=np.nan, bounds_error=False)(time_mapping)  # type: ignore
        for (st, sv) in zip(seq_times, seq_vals, strict=False)
    ]
    return unified_seq_ticks, tuple(unified_seqs_val)


def gaussian_filter1d_with_nan(seq, sigma, **kwargs):
    """Apply a 1D Gaussian filter to a sequence while handling NaN values.

    This function applies Gaussian smoothing to a sequence, ignoring NaN values to prevent distortion.

    Args:
        seq (numpy.ndarray): Input sequence with possible NaN values.
        sigma (float): Standard deviation for Gaussian kernel.
        **kwargs: Additional arguments for scipy.ndimage.gaussian_filter1d.

    Returns:
        numpy.ndarray: Smoothed sequence with NaN handling.
    """
    # https://stackoverflow.com/a/36307291
    if sigma > 0:
        (v := seq.copy())[np.isnan(seq)] = 0
        vv = gaussian_filter1d(v, sigma, **kwargs)
        (w := np.ones(len(seq)))[np.isnan(seq)] = 0
        ww = gaussian_filter1d(w, sigma, **kwargs)
        with np.errstate(invalid="ignore"):
            return np.divide(vv, ww)
    else:
        return seq


def seq_spline_smoothing(seq_time, seq_val, lam=None, nan_policy='preserve_all'):
    """Smooth a sequence using an adaptive smoothing spline.

    Args:
        seq_time (numpy.ndarray): Time values for the sequence.
        seq_val (numpy.ndarray):  Sequence values to smooth.
        lam (float or None):      Smoothing parameter. None selects automatically via GCV.
                                  Higher values produce smoother results.
        nan_policy (str):         How to handle NaNs in the output. One of:
            - 'no_nan':            NaNs are excluded from fitting; output is fully predicted
                                   with no NaNs.
            - 'preserve_all':      NaN positions are excluded from fitting and restored in
                                   the output.
            - 'preserve_head_tail': Only leading and trailing NaNs are restored; interior
                                   NaNs are filled by the spline.

    Returns:
        numpy.ndarray: Smoothed sequence values, same length as seq_time.

    Raises:
        ValueError: If nan_policy is not recognized.

    Example:
        >>> seq_smoothing(time, val)                                    # auto smoothness, preserve all NaNs
        >>> seq_smoothing(time, val, lam=0.1)                           # manual smoothness
        >>> seq_smoothing(time, val, nan_policy='no_nan')               # fully predicted, no NaNs
        >>> seq_smoothing(time, val, nan_policy='preserve_head_tail')   # only boundary NaNs restored
    """
    nan_policies = {'no_nan', 'preserve_all', 'preserve_head_tail'}
    if nan_policy not in nan_policies:
        raise ValueError(f"Unknown nan_policy: {nan_policy!r}. Choose from: {nan_policies}.")

    seq_val  = np.asarray(seq_val,  dtype=float)
    seq_time = np.asarray(seq_time, dtype=float)
    nan_mask = np.isnan(seq_val)

    valid_time = seq_time[~nan_mask]
    valid_val  = seq_val[~nan_mask]

    result = make_smoothing_spline(valid_time, valid_val, lam=lam)(seq_time)

    if nan_policy == 'preserve_all':
        result[nan_mask] = np.nan

    elif nan_policy == 'preserve_head_tail':
        first_valid    = np.argmax(~nan_mask)
        last_valid     = len(nan_mask) - np.argmax(~nan_mask[::-1]) - 1
        head_tail_mask = nan_mask.copy()
        head_tail_mask[first_valid:last_valid + 1] = False
        result[head_tail_mask] = np.nan

    return result


def align_sequence_tick(
    query_time, queries, reference_time, references, align_radius=1
):
    """Align sequences to a common MIDI tick time base using dynamic time warping.

    Requires tick converters to be registered via :func:`set_tick_converters`.

    Args:
        query_time (numpy.ndarray):     Time values for the query sequences.
        queries (tuple):                Query sequences to align.
        reference_time (numpy.ndarray): Time values for the reference sequences.
        references (tuple):             Reference sequences to align.
        align_radius (int, optional):   DTW radius. Defaults to 1.

    Returns:
        tuple: (unified_tick, aligned_queries, unified_references), where:
            - unified_tick (numpy.ndarray):    Unified MIDI tick time base. Shape: (n_time_points).
            - aligned_queries (tuple):         Aligned query sequences. Shape: (n_sequences, n_time_points).
            - unified_references (tuple):      Unified reference sequences. Shape: (n_sequences, n_time_points).
    """
    query_times     = [query_time]     * len(queries)
    reference_times = [reference_time] * len(references)

    # Unify time and sequences
    unified_tick, seqs = unify_sequence_time(
        (*query_times, *reference_times),
        (*queries, *references),
        to_ticks=True,
    )
    unified_queries    = list(seqs)[: len(queries)]
    unified_references = list(seqs)[len(queries):]

    # Align sequences using dynamic time warping
    qs_nonan = np.nan_to_num(zscore(unified_queries, axis=1, nan_policy="omit"))
    rs_nonan = np.nan_to_num(zscore(unified_references, axis=1, nan_policy="omit"))
    _, path = fastdtw(
        list(map(tuple, zip(*qs_nonan,  strict=False))),
        list(map(tuple, zip(*rs_nonan,  strict=False))),
        radius=align_radius,
    )

    # Align queries to reference time
    path = np.array(path)
    aligned_queries = []
    for q in unified_queries:
        aligned_tick = np.interp(path[:, 1], np.arange(len(unified_tick)), unified_tick)
        aligned_seq  = np.interp(path[:, 0], np.arange(len(q)), q)
        interp_seq   = interp1d(aligned_tick, aligned_seq, fill_value=np.nan, bounds_error=False)  # type: ignore
        aligned_queries.append(interp_seq(unified_tick))

    return unified_tick, tuple(aligned_queries), tuple(unified_references)


def seq_dynamics_trends(seq, n_order=3):
    """Extract dynamic and trend features from a sequence.
    This function computes the gradients and cumulative sums of a sequence.
    Args:
        seq (numpy.ndarray): Input sequence. Shape: (n_time_points,).
        n_order (int, optional): Order of features to extract. Defaults to 3.

    Returns:
        numpy.ndarray: Extracted features, including gradients and cumulative sums. Shape: (2 * n_order, n_time_points).
    """
    # Extract dynamic features (order 1 to order n)
    seq_grads = list(accumulate([seq] * (n_order + 1), lambda x, _: np.gradient(x)))
    seq_grads = np.vstack(seq_grads[1:])

    # Extract trend features (order 1 to order n)
    seq_trends = list(accumulate([seq] * (n_order + 1), lambda x, _: np.nancumsum(x)))
    seq_trends = np.vstack(seq_trends[1:])
    return np.vstack([seq_grads, seq_trends])


def seq_rcr(seq):
    """Compute the relative change rate (RCR) of a sequence.
    This function calculates the relative change rate of a sequence, which is useful for analyzing dynamics.
    Args:
        seq (numpy.ndarray): Input sequence. Shape: (n_time_points,).

    Returns:
        numpy.ndarray: Relative change rate of the sequence. Shape: (n_time_points,).
    """
    epsilon = 1e-6
    rcr_raw = np.abs(np.diff(seq)) / (seq[:-1] + epsilon)
    rcr = np.insert(rcr_raw, 0, rcr_raw[0])
    return rcr
