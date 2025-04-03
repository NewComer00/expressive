import numpy as np
from fastdtw import fastdtw # type: ignore
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def time_to_ticks(time, tempo, ppqn=480):
    """Convert time in seconds to MIDI ticks.

    Args:
        time (float or array-like): Time values in seconds.
        tempo (float): Tempo in beats per minute (BPM).
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.

    Returns:
        numpy.ndarray: Corresponding MIDI tick values.
    """
    return (np.array(time) * tempo * ppqn) / 60


def ticks_to_time(ticks, tempo, ppqn=480):
    """Convert MIDI ticks to time in seconds.

    Args:
        ticks (int or array-like): MIDI tick values.
        tempo (float): Tempo in beats per minute (BPM).
        ppqn (int, optional): Pulses per quarter note. Defaults to 480.

    Returns:
        numpy.ndarray: Corresponding time values in seconds.
    """
    return (np.array(ticks) * 60) / (tempo * ppqn)


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


def unify_sequence_time(seq_times, seq_vals, to_ticks=False, tempo=120, ppqn=480):
    """Unify multiple sequences to a common time base.

    This function aligns multiple sequences to a common time base by interpolating values.

    Args:
        seq_times (list of array-like): List of time sequences. Shape: (n_sequences, n_time_points).
        seq_vals (list of array-like): List of value sequences. Shape: (n_sequences, n_time_points).
        to_ticks (bool, optional): Whether to convert time to MIDI ticks. Defaults to False.
        tempo (float, optional): Tempo in beats per minute (BPM). Defaults to 120.
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.

    Returns:
        tuple: (unified_time, unified_seqs), where:
            - unified_time (numpy.ndarray): Unified time points. Shape: (n_time_points).
            - unified_seqs (tuple): Unified sequences. Shape: (n_sequences, n_time_points).
    """
    unified_seq_time = np.array(sequence_interval_intersection(seq_times))
    if not to_ticks:
        unified_seq_time = np.unique(unified_seq_time)
        unified_seqs_val = [
            interp1d(st, sv, fill_value="extrapolate")(unified_seq_time) # type: ignore
            for (st, sv) in zip(seq_times, seq_vals)
        ]
        return unified_seq_time, tuple(unified_seqs_val)

    else:
        unified_seq_ticks = time_to_ticks(unified_seq_time, tempo, ppqn)
        unified_seq_ticks = np.unique(np.round(unified_seq_ticks).astype(int))

        time_mapping = ticks_to_time(unified_seq_ticks, tempo, ppqn)
        unified_seqs_val = [
            interp1d(st, sv, fill_value="extrapolate")(time_mapping) # type: ignore
            for (st, sv) in zip(seq_times, seq_vals)
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


def align_sequence_tick(
    query_time, queries, reference_time, references, tempo=120, ppqn=480, align_radius=1
):
    """Align sequences to a common MIDI tick time base.

    This function aligns sequences to a common MIDI tick time base using dynamic time warping.

    Args:
        query_time (numpy.ndarray): Time values for the query sequences. Shape: (n_time_points).
        queries (tuple): Query sequences to align. Shape: (n_sequences, n_time_points).
        reference_time (numpy.ndarray): Time values for the reference sequences. Shape: (n_time_points).
        references (tuple): Reference sequences to align. Shape: (n_sequences, n_time_points).
        tempo (float, optional): Tempo in beats per minute (BPM). Defaults to 120.
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.
        align_radius (int, optional): Radius for dynamic time warping. Defaults to 1.

    Returns:
        tuple: (unified_tick, aligned_queries, unified_references), where:
            - unified_tick (numpy.ndarray): Unified MIDI tick time base. Shape: (n_time_points).
            - aligned_queries (tuple): Aligned query sequences. Shape: (n_sequences, n_time_points).
            - unified_references (tuple): Unified reference sequences. Shape: (n_sequences, n_time_points).
    """
    query_times = [query_time] * len(queries)
    reference_times = [reference_time] * len(references)

    # Unify time and sequences
    unified_tick, seqs = unify_sequence_time(
        (*query_times, *reference_times),
        (*queries, *references),
        to_ticks=True,
        tempo=tempo,
        ppqn=ppqn,
    )
    unified_queries = list(seqs)[: len(queries)]
    unified_references = list(seqs)[len(queries) :]

    # Align sequences using dynamic time warping
    qs_nonan = np.nan_to_num(zscore(unified_queries, axis=1, nan_policy="omit"))
    rs_nonan = np.nan_to_num(zscore(unified_references, axis=1, nan_policy="omit"))
    distance, path = fastdtw(
        list(map(tuple, zip(*qs_nonan))),
        list(map(tuple, zip(*rs_nonan))),
        radius=align_radius,
    )

    # Align queries to reference time
    path = np.array(path)
    aligned_queries = []
    for q in unified_queries:
        aligned_tick = np.interp(path[:, 1], np.arange(len(unified_tick)), unified_tick)
        aligned_seq = np.interp(path[:, 0], np.arange(len(q)), q)
        interp_seq = interp1d(aligned_tick, aligned_seq, fill_value="extrapolate") # type: ignore
        aligned_queries.append(interp_seq(unified_tick))

    return unified_tick, tuple(aligned_queries), tuple(unified_references)
