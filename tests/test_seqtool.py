import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from utils.seqtool import (
    set_tick_converters,
    reset_tick_converters,
    sequence_interval_intersection,
    sequence_interval_union,
    unify_sequence_time,
    gaussian_filter1d_with_nan,
    align_sequence_tick,
    seq_dynamics_trends,
    seq_rcr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tick_converters(tempo=120, ppqn=480):
    """Return a (time_to_ticks, ticks_to_time) pair for a constant tempo map."""
    ticks_per_second = tempo / 60 * ppqn

    def time_to_ticks(times: np.ndarray, unique: bool = True) -> np.ndarray:
        ticks = np.round(np.asarray(times, dtype=float) * ticks_per_second).astype(int)
        return np.unique(ticks) if unique else ticks

    def ticks_to_time(ticks: np.ndarray) -> np.ndarray:
        return np.asarray(ticks, dtype=float) / ticks_per_second

    return time_to_ticks, ticks_to_time


@pytest.fixture()
def tick_converters_120bpm():
    """Register 120 BPM / 480 PPQN converters, then reset after the test."""
    t2t, t2s = _make_tick_converters(tempo=120, ppqn=480)
    set_tick_converters(t2t, t2s)
    yield t2t, t2s
    reset_tick_converters()


# ---------------------------------------------------------------------------
# Tick converter registry
# ---------------------------------------------------------------------------

class TestTickConverterRegistry:
    """Test the set_tick_converters / reset_tick_converters machinery."""

    def test_raises_before_registration(self):
        reset_tick_converters()
        with pytest.raises(RuntimeError, match="No tick converter registered"):
            unify_sequence_time(
                [np.array([0.0, 1.0])],
                [np.array([0.0, 1.0])],
                to_ticks=True,
            )

    def test_set_and_reset(self):
        t2t, t2s = _make_tick_converters()
        set_tick_converters(t2t, t2s)
        # Should not raise
        unify_sequence_time(
            [np.array([0.0, 1.0])],
            [np.array([0.0, 1.0])],
            to_ticks=True,
        )
        reset_tick_converters()
        with pytest.raises(RuntimeError):
            unify_sequence_time(
                [np.array([0.0, 1.0])],
                [np.array([0.0, 1.0])],
                to_ticks=True,
            )

    def test_converter_values(self):
        """Registered converters should produce correct tick values."""
        t2t, t2s = _make_tick_converters(tempo=120, ppqn=480)
        set_tick_converters(t2t, t2s)

        # 120 BPM, 480 PPQN → 960 ticks/second
        assert t2t(np.array([1.0]), unique=False)[0] == 960
        assert t2s(np.array([960]))[0] == pytest.approx(1.0)

        reset_tick_converters()

    @pytest.mark.parametrize("tempo,ppqn,time,expected_ticks", [
        (120, 480, 1.0,  960),
        ( 60, 480, 1.0,  480),
        (240, 480, 1.0, 1920),
        (120, 960, 1.0, 1920),
        (120, 480, 0.5,  480),
        (120, 480, 2.0, 1920),
        (120, 480, 0.0,    0),
    ])
    def test_converter_parametrized(self, tempo, ppqn, time, expected_ticks):
        t2t, _ = _make_tick_converters(tempo=tempo, ppqn=ppqn)
        result = t2t(np.array([time]), unique=False)
        assert result[0] == expected_ticks

    def test_converter_roundtrip(self):
        t2t, t2s = _make_tick_converters(tempo=120, ppqn=480)
        original = np.array([0.5, 1.0, 1.5, 2.0])
        ticks = t2t(original, unique=False)
        recovered = t2s(ticks)
        assert_array_almost_equal(original, recovered)

    def test_converter_roundtrip_precision(self):
        """Round-trip error must stay within half a tick duration."""
        t2t, t2s = _make_tick_converters(tempo=120, ppqn=480)
        original = np.linspace(0, 10, 1000)
        ticks = t2t(original, unique=False)
        recovered = t2s(ticks)
        tick_duration = 60 / (120 * 480)
        max_error = tick_duration / 2
        assert np.all(np.abs(original - recovered) <= max_error + 1e-12)

    def test_converter_unique_deduplicates(self):
        t2t, _ = _make_tick_converters(tempo=120, ppqn=480)
        times = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
        result = t2t(times, unique=True)
        assert_array_equal(result, np.array([0, 960, 1920]))

    def test_converter_zero(self):
        t2t, _ = _make_tick_converters(tempo=120, ppqn=480)
        assert t2t(np.array([0.0]), unique=False)[0] == 0

    def test_converter_negative(self):
        t2t, _ = _make_tick_converters(tempo=120, ppqn=480)
        assert t2t(np.array([-1.0]), unique=False)[0] == -960


# ---------------------------------------------------------------------------
# Sequence interval operations
# ---------------------------------------------------------------------------

class TestSequenceOperations:
    """Test sequence_interval_intersection and sequence_interval_union."""

    def test_intersection_basic(self):
        seqs = [[0, 1, 2, 3], [1.0, 1.1, 2.0, 4.0, 5.0]]
        result = sequence_interval_intersection(seqs)
        assert result == [1.0, 1.1, 2.0, 3.0]

    def test_intersection_no_overlap(self):
        seqs = [[0, 1, 2], [5, 6, 7]]
        result = sequence_interval_intersection(seqs)
        assert result == []

    def test_intersection_complete_overlap(self):
        seqs = [[1, 2, 3], [1, 2, 3]]
        result = sequence_interval_intersection(seqs)
        assert result == [1, 2, 3]

    @pytest.mark.parametrize("seq1,seq2,expected_len", [
        ([0, 1, 2],       [1, 2, 3],       2),
        ([0, 1, 2],       [5, 6, 7],       0),
        ([0, 1, 2, 3, 4], [2, 3, 4, 5],    3),
        ([1, 2, 3],       [1, 2, 3],       3),
    ])
    def test_intersection_parametrized(self, seq1, seq2, expected_len):
        result = sequence_interval_intersection([seq1, seq2])
        assert len(result) == expected_len

    def test_union_basic(self):
        seqs = [[0, 1, 2, 3], [1.0, 1.1, 2.0, 4.0, 5.0]]
        result = sequence_interval_union(seqs)
        assert result == [0.0, 1.0, 1.1, 2.0, 3.0, 4.0, 5.0]

    def test_union_deduplicates(self):
        seqs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        result = sequence_interval_union(seqs)
        assert result == [1, 2, 3, 4, 5]

    def test_union_sorted(self):
        seqs = [[5, 3, 1], [4, 2, 0]]
        result = sequence_interval_union(seqs)
        assert result == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# unify_sequence_time
# ---------------------------------------------------------------------------

class TestUnifySequenceTime:
    """Test unify_sequence_time in both seconds and ticks mode."""

    def test_basic_same_length(self):
        seq_times = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        seq_vals  = [np.array([10., 20., 30.]), np.array([15., 25., 35.])]
        unified_time, (v0, v1) = unify_sequence_time(seq_times, seq_vals)
        assert len(v0) == len(v1) == len(unified_time)

    def test_different_lengths(self):
        seq_times = [np.array([0.0, 1.0, 2.0]),
                     np.array([0.0, 0.5, 1.0, 1.5, 2.0])]
        seq_vals  = [np.array([10., 20., 30.]),
                     np.array([15., 17., 22., 27., 32.])]
        unified_time, (v0, v1) = unify_sequence_time(seq_times, seq_vals)
        assert len(v0) == len(v1) == len(unified_time)

    def test_to_ticks_output_dtype(self, tick_converters_120bpm):
        seq_times = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        seq_vals  = [np.array([10., 20., 30.]), np.array([15., 25., 35.])]
        unified_ticks, (v0, v1) = unify_sequence_time(seq_times, seq_vals, to_ticks=True)
        assert np.issubdtype(unified_ticks.dtype, np.integer)
        assert len(v0) == len(v1) == len(unified_ticks)

    def test_to_ticks_values(self, tick_converters_120bpm):
        """Unified ticks should match the registered converter's output."""
        seq_times = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        seq_vals  = [np.array([0., 1., 2.]), np.array([0., 1., 2.])]
        unified_ticks, _ = unify_sequence_time(seq_times, seq_vals, to_ticks=True)
        # 120 BPM, 480 PPQN → 0 s=0 ticks, 1 s=960, 2 s=1920
        assert_array_equal(unified_ticks, np.array([0, 960, 1920]))

    def test_to_ticks_requires_registration(self):
        reset_tick_converters()
        with pytest.raises(RuntimeError):
            unify_sequence_time(
                [np.array([0.0, 1.0])],
                [np.array([0.0, 1.0])],
                to_ticks=True,
            )


# ---------------------------------------------------------------------------
# gaussian_filter1d_with_nan
# ---------------------------------------------------------------------------

class TestGaussianFilter:
    """Test gaussian_filter1d_with_nan."""

    def test_no_nan(self):
        seq = np.array([1., 2., 3., 4., 5.])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)
        assert result.shape == seq.shape
        assert not np.any(np.isnan(result))
        assert np.var(result) < np.var(seq)

    def test_with_nan(self):
        seq = np.array([1., 2., np.nan, 4., 5.])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)
        assert result.shape == seq.shape
        for i in [0, 1, 3, 4]:
            assert not np.isnan(result[i])

    def test_zero_sigma_identity(self):
        seq = np.array([1., 2., 3., 4., 5.])
        result = gaussian_filter1d_with_nan(seq, sigma=0)
        assert_array_equal(result, seq)

    def test_all_nan(self):
        seq = np.array([np.nan, np.nan, np.nan])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)
        assert np.all(np.isnan(result))

    def test_preserves_mean(self):
        np.random.seed(42)
        seq = np.random.randn(100) + 10
        result = gaussian_filter1d_with_nan(seq, sigma=2.0)
        assert abs(np.mean(result) - np.mean(seq)) < 0.5

    @pytest.mark.parametrize("sigma,should_smooth", [
        (0,   False),
        (0.5, True),
        (1.0, True),
        (2.0, True),
    ])
    def test_smoothing_levels(self, sigma, should_smooth):
        seq = np.array([1., 5., 2., 6., 3.])
        result = gaussian_filter1d_with_nan(seq, sigma)
        if should_smooth:
            assert np.var(result) < np.var(seq)
        else:
            assert_array_almost_equal(result, seq)


# ---------------------------------------------------------------------------
# seq_dynamics_trends
# ---------------------------------------------------------------------------

class TestSeqDynamicsTrends:
    """Test seq_dynamics_trends."""

    def test_output_shape(self):
        seq = np.array([1., 2., 3., 4., 5.])
        result = seq_dynamics_trends(seq, n_order=3)
        assert result.shape == (6, len(seq))  # 2 * n_order rows

    def test_constant_sequence_zero_gradient(self):
        seq = np.array([5., 5., 5., 5., 5.])
        result = seq_dynamics_trends(seq, n_order=2)
        assert np.allclose(result[0], 0, atol=1e-10)

    def test_linear_sequence_unit_gradient(self):
        seq = np.array([1., 2., 3., 4., 5.])
        result = seq_dynamics_trends(seq, n_order=2)
        assert np.allclose(result[0], 1.0, atol=0.1)

    @pytest.mark.parametrize("n_order", [1, 2, 3, 4])
    def test_various_orders(self, n_order):
        seq = np.array([1., 2., 3., 4., 5.])
        result = seq_dynamics_trends(seq, n_order=n_order)
        assert result.shape[0] == 2 * n_order


# ---------------------------------------------------------------------------
# seq_rcr
# ---------------------------------------------------------------------------

class TestSeqRCR:
    """Test seq_rcr (relative change rate)."""

    def test_output_shape(self):
        seq = np.array([1., 2., 4., 8.])
        result = seq_rcr(seq)
        assert result.shape == seq.shape

    def test_first_value_duplicated(self):
        seq = np.array([1., 2., 4., 8.])
        result = seq_rcr(seq)
        assert result[0] == result[1]

    def test_constant_near_zero(self):
        seq = np.array([5., 5., 5., 5.])
        result = seq_rcr(seq)
        assert np.all(result < 0.01)

    def test_zero_values_no_nan_inf(self):
        seq = np.array([0., 1., 2.])
        result = seq_rcr(seq)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_negative_values_no_nan_inf(self):
        seq = np.array([-1., -2., -3.])
        result = seq_rcr(seq)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_very_small_values_no_nan_inf(self):
        seq = np.array([1e-10, 2e-10, 3e-10])
        result = seq_rcr(seq)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ---------------------------------------------------------------------------
# align_sequence_tick
# ---------------------------------------------------------------------------

class TestAlignSequenceTick:
    """Test align_sequence_tick (DTW-based alignment)."""

    @pytest.mark.slow
    def test_basic_alignment(self, tick_converters_120bpm):
        query_time = np.linspace(0, 5, 50)
        ref_time   = np.linspace(0, 5, 50)
        query_seq  = np.sin(2 * np.pi * query_time)
        ref_seq    = np.sin(2 * np.pi * ref_time)

        unified_tick, aligned_queries, unified_refs = align_sequence_tick(
            query_time, (query_seq,),
            ref_time,   (ref_seq,),
            align_radius=1,
        )

        assert len(aligned_queries) == 1
        assert len(unified_refs)    == 1
        assert len(aligned_queries[0]) == len(unified_tick)
        assert len(unified_refs[0])    == len(unified_tick)

    @pytest.mark.slow
    def test_multiple_features(self, tick_converters_120bpm):
        query_time = np.linspace(0, 5, 50)
        ref_time   = np.linspace(0, 5, 50)
        qs = (np.sin(2 * np.pi * query_time), np.cos(2 * np.pi * query_time))
        rs = (np.sin(2 * np.pi * ref_time),   np.cos(2 * np.pi * ref_time))

        unified_tick, aligned_queries, unified_refs = align_sequence_tick(
            query_time, qs, ref_time, rs, align_radius=1,
        )

        assert len(aligned_queries) == 2
        assert len(unified_refs)    == 2
        for aq in aligned_queries:
            assert len(aq) == len(unified_tick)
        for ur in unified_refs:
            assert len(ur) == len(unified_tick)

    @pytest.mark.slow
    def test_requires_registration(self):
        reset_tick_converters()
        query_time = np.linspace(0, 2, 20)
        ref_time   = np.linspace(0, 2, 20)
        with pytest.raises(RuntimeError, match="No tick converter registered"):
            align_sequence_tick(
                query_time, (np.ones(20),),
                ref_time,   (np.ones(20),),
            )
