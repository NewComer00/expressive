import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from utils.seqtool import (
    time_to_ticks,
    ticks_to_time,
    sequence_interval_intersection,
    sequence_interval_union,
    unify_sequence_time,
    gaussian_filter1d_with_nan,
    align_sequence_tick,
    seq_dynamics_trends,
    seq_rcr,
)


class TestTimeConversion:
    """Test time and MIDI tick conversion functions"""

    def test_time_to_ticks_basic(self):
        """Test basic time to tick conversion"""
        # 120 BPM, 480 PPQN
        # 1 second = 120/60 * 480 = 960 ticks
        result = time_to_ticks(1.0, tempo=120, ppqn=480)
        assert result == 960

    def test_time_to_ticks_array(self):
        """Test array input"""
        times = [0, 0.5, 1.0, 2.0]
        result = time_to_ticks(times, tempo=120, ppqn=480)
        expected = np.array([0, 480, 960, 1920])
        assert_array_equal(result, expected)

    def test_ticks_to_time_basic(self):
        """Test basic tick to time conversion"""
        result = ticks_to_time(960, tempo=120, ppqn=480)
        assert result == 1.0

    def test_time_tick_roundtrip(self):
        """Test roundtrip conversion consistency"""
        original_time = np.array([0.5, 1.0, 1.5, 2.0])
        ticks = time_to_ticks(original_time, tempo=120, ppqn=480)
        recovered_time = ticks_to_time(ticks, tempo=120, ppqn=480)
        assert_array_almost_equal(original_time, recovered_time)

    @pytest.mark.parametrize("time,tempo,ppqn,expected", [
        (1.0, 120, 480, 960),      # Standard case
        (0.5, 120, 480, 480),      # Half second
        (2.0, 120, 480, 1920),     # Two seconds
        (1.0, 60, 480, 480),       # Slower tempo
        (1.0, 240, 480, 1920),     # Faster tempo
        (1.0, 120, 960, 1920),     # Higher resolution
    ])
    def test_time_to_ticks_parametrized(self, time, tempo, ppqn, expected):
        """Test time_to_ticks with multiple parameter combinations"""
        result = time_to_ticks(time, tempo, ppqn)
        assert result == expected

    def test_time_to_ticks_zero(self):
        """Test with zero time"""
        result = time_to_ticks(0, tempo=120, ppqn=480)
        assert result == 0

    def test_time_to_ticks_negative(self):
        """Test with negative time"""
        result = time_to_ticks(-1.0, tempo=120, ppqn=480)
        assert result == -960


class TestSequenceOperations:
    """Test sequence operation functions"""

    def test_sequence_interval_intersection_basic(self):
        """Test sequence intersection - basic case"""
        seqs = [[0, 1, 2, 3], [1.0, 1.1, 2.0, 4.0, 5.0]]
        result = sequence_interval_intersection(seqs)
        # Intersection range is [1.0, 3.0]
        expected = [1.0, 1.1, 2.0, 3.0]
        assert result == expected

    def test_sequence_interval_intersection_no_overlap(self):
        """Test sequences with no overlap"""
        seqs = [[0, 1, 2], [5, 6, 7]]
        result = sequence_interval_intersection(seqs)
        assert result == []

    def test_sequence_interval_intersection_complete_overlap(self):
        """Test sequences with complete overlap"""
        seqs = [[1, 2, 3], [1, 2, 3]]
        result = sequence_interval_intersection(seqs)
        assert result == [1, 2, 3]

    @pytest.mark.parametrize("seq1,seq2,expected_len", [
        ([0, 1, 2], [1, 2, 3], 2),           # Overlap: [1, 2]
        ([0, 1, 2], [5, 6, 7], 0),           # No overlap
        ([0, 1, 2, 3, 4], [2, 3, 4, 5], 3),  # Overlap: [2, 3, 4]
        ([1, 2, 3], [1, 2, 3], 3),           # Complete overlap
    ])
    def test_sequence_intersection_parametrized(self, seq1, seq2, expected_len):
        """Test sequence intersection with various inputs"""
        result = sequence_interval_intersection([seq1, seq2])
        assert len(result) == expected_len

    def test_sequence_interval_union_basic(self):
        """Test sequence union - basic case"""
        seqs = [[0, 1, 2, 3], [1.0, 1.1, 2.0, 4.0, 5.0]]
        result = sequence_interval_union(seqs)
        expected = [0.0, 1.0, 1.1, 2.0, 3.0, 4.0, 5.0]
        assert result == expected

    def test_sequence_interval_union_duplicates(self):
        """Test union removes duplicates"""
        seqs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        result = sequence_interval_union(seqs)
        expected = [1, 2, 3, 4, 5]
        assert result == expected

    def test_sequence_interval_union_sorted(self):
        """Test union returns sorted result"""
        seqs = [[5, 3, 1], [4, 2, 0]]
        result = sequence_interval_union(seqs)
        expected = [0, 1, 2, 3, 4, 5]
        assert result == expected


class TestUnifySequenceTime:
    """Test sequence time unification function"""

    def test_unify_sequence_time_basic(self):
        """Test basic sequence unification"""
        seq_times = [
            np.array([0, 1, 2]),
            np.array([0, 1, 2])
        ]
        seq_vals = [
            np.array([10, 20, 30]),
            np.array([15, 25, 35])
        ]

        unified_time, unified_vals = unify_sequence_time(
            seq_times, seq_vals, to_ticks=False
        )

        # All unified sequences should have same length
        assert len(unified_vals[0]) == len(unified_vals[1])
        assert len(unified_vals[0]) == len(unified_time)

    def test_unify_sequence_time_different_lengths(self):
        """Test unifying sequences with different lengths"""
        seq_times = [
            np.array([0, 1, 2]),
            np.array([0, 0.5, 1, 1.5, 2])
        ]
        seq_vals = [
            np.array([10, 20, 30]),
            np.array([15, 17, 22, 27, 32])
        ]

        unified_time, unified_vals = unify_sequence_time(
            seq_times, seq_vals, to_ticks=False
        )

        # All unified sequences should have same length
        assert len(unified_vals[0]) == len(unified_vals[1])
        assert len(unified_vals[0]) == len(unified_time)

    def test_unify_sequence_time_to_ticks(self):
        """Test unification with tick conversion"""
        seq_times = [
            np.array([0, 1, 2]),
            np.array([0, 1, 2])
        ]
        seq_vals = [
            np.array([10, 20, 30]),
            np.array([15, 25, 35])
        ]

        unified_ticks, unified_vals = unify_sequence_time(
            seq_times, seq_vals, to_ticks=True, tempo=120, ppqn=480
        )

        # Result should be in ticks (integers)
        assert unified_ticks.dtype == np.int64 or unified_ticks.dtype == np.int32
        assert len(unified_vals[0]) == len(unified_vals[1])


class TestGaussianFilter:
    """Test Gaussian filter with NaN handling"""

    def test_gaussian_filter_no_nan(self):
        """Test without NaN values"""
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)

        assert result.shape == seq.shape
        assert not np.any(np.isnan(result))
        # Result should be smoothed
        assert np.var(result) < np.var(seq)

    def test_gaussian_filter_with_nan(self):
        """Test with NaN values"""
        seq = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)

        assert result.shape == seq.shape
        # Non-NaN positions should have reasonable values
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    def test_gaussian_filter_zero_sigma(self):
        """Test with sigma=0 (no filtering)"""
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gaussian_filter1d_with_nan(seq, sigma=0)

        # Should return original sequence
        assert_array_equal(result, seq)

    def test_gaussian_filter_all_nan(self):
        """Test with all NaN values"""
        seq = np.array([np.nan, np.nan, np.nan])
        result = gaussian_filter1d_with_nan(seq, sigma=1.0)

        # Result should also be all NaN
        assert np.all(np.isnan(result))

    @pytest.mark.parametrize("sigma,should_smooth", [
        (0, False),    # No smoothing
        (0.5, True),   # Light smoothing
        (1.0, True),   # Medium smoothing
        (2.0, True),   # Heavy smoothing
    ])
    def test_gaussian_filter_smoothing_levels(self, sigma, should_smooth):
        """Test different smoothing levels"""
        # Create a noisy signal
        seq = np.array([1.0, 5.0, 2.0, 6.0, 3.0])
        result = gaussian_filter1d_with_nan(seq, sigma)

        if should_smooth:
            # Result should be smoother (less variance)
            assert np.var(result) < np.var(seq)
        else:
            # No smoothing, should be identical
            assert_array_almost_equal(result, seq)


class TestSeqDynamicsTrends:
    """Test dynamics and trends extraction"""

    def test_seq_dynamics_trends_basic(self):
        """Test basic dynamics trends extraction"""
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = seq_dynamics_trends(seq, n_order=3)

        # Should return 2 * n_order features
        assert result.shape[0] == 6  # 2 * 3
        assert result.shape[1] == len(seq)

    def test_seq_dynamics_trends_constant(self):
        """Test with constant sequence"""
        seq = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = seq_dynamics_trends(seq, n_order=2)

        # Gradients should be near zero
        assert np.allclose(result[0], 0, atol=1e-10)

    def test_seq_dynamics_trends_linear(self):
        """Test with linear sequence"""
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = seq_dynamics_trends(seq, n_order=2)

        # First gradient should be constant (all 1s)
        assert np.allclose(result[0], 1.0, atol=0.1)

    def test_seq_dynamics_trends_different_orders(self):
        """Test with different orders"""
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        for n_order in [1, 2, 3, 4]:
            result = seq_dynamics_trends(seq, n_order=n_order)
            assert result.shape[0] == 2 * n_order


class TestSeqRCR:
    """Test relative change rate calculation"""

    def test_seq_rcr_basic(self):
        """Test basic RCR calculation"""
        seq = np.array([1.0, 2.0, 4.0, 8.0])
        result = seq_rcr(seq)

        assert result.shape == seq.shape
        # First value should be duplicated
        assert result[0] == result[1]

    def test_seq_rcr_constant(self):
        """Test with constant sequence"""
        seq = np.array([5.0, 5.0, 5.0, 5.0])
        result = seq_rcr(seq)

        # RCR should be near zero for constant sequence
        assert np.all(result < 0.01)

    def test_seq_rcr_zero_values(self):
        """Test with zero values (should handle epsilon)"""
        seq = np.array([0.0, 1.0, 2.0])
        result = seq_rcr(seq)

        # Should not produce inf or nan
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_seq_rcr_negative_values(self):
        """Test with negative values"""
        seq = np.array([-1.0, -2.0, -3.0])
        result = seq_rcr(seq)

        # Should handle negative values
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestAlignSequenceTick:
    """Test sequence alignment with DTW"""

    @pytest.mark.slow
    def test_align_sequence_tick_basic(self):
        """Test basic sequence alignment"""
        # Create two similar sequences with slight time shift
        query_time = np.linspace(0, 5, 50)
        reference_time = np.linspace(0, 5, 50)

        query_seq = np.sin(2 * np.pi * query_time)
        reference_seq = np.sin(2 * np.pi * reference_time)

        unified_tick, aligned_queries, unified_refs = align_sequence_tick(
            query_time,
            (query_seq,),
            reference_time,
            (reference_seq,),
            tempo=120,
            ppqn=480,
            align_radius=1
        )

        # Check output shapes
        assert len(aligned_queries) == 1
        assert len(unified_refs) == 1
        assert len(aligned_queries[0]) == len(unified_tick)
        assert len(unified_refs[0]) == len(unified_tick)

    @pytest.mark.slow
    def test_align_sequence_tick_multiple_features(self):
        """Test alignment with multiple features"""
        query_time = np.linspace(0, 5, 50)
        reference_time = np.linspace(0, 5, 50)

        # Multiple features
        query_seq1 = np.sin(2 * np.pi * query_time)
        query_seq2 = np.cos(2 * np.pi * query_time)
        reference_seq1 = np.sin(2 * np.pi * reference_time)
        reference_seq2 = np.cos(2 * np.pi * reference_time)

        unified_tick, aligned_queries, unified_refs = align_sequence_tick(
            query_time,
            (query_seq1, query_seq2),
            reference_time,
            (reference_seq1, reference_seq2),
            tempo=120,
            ppqn=480,
            align_radius=1
        )

        # Check output shapes
        assert len(aligned_queries) == 2
        assert len(unified_refs) == 2


class TestNumericalStability:
    """Test numerical stability and precision"""

    def test_time_tick_roundtrip_precision(self):
        """Test precision in roundtrip conversion"""
        original_times = np.linspace(0, 10, 1000)
        ticks = time_to_ticks(original_times, tempo=120, ppqn=480)
        recovered_times = ticks_to_time(ticks, tempo=120, ppqn=480)

        # Should be very close
        assert_array_almost_equal(original_times, recovered_times, decimal=10)

    def test_gaussian_filter_preserves_mean(self):
        """Test that Gaussian filter approximately preserves mean"""
        np.random.seed(42)
        seq = np.random.randn(100) + 10
        result = gaussian_filter1d_with_nan(seq, sigma=2.0)

        # Mean should be approximately preserved
        assert abs(np.mean(result) - np.mean(seq)) < 0.5

    def test_seq_rcr_with_very_small_values(self):
        """Test RCR with very small values"""
        seq = np.array([1e-10, 2e-10, 3e-10])
        result = seq_rcr(seq)

        # Should not produce inf or nan
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
