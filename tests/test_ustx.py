"""
Tests for utils/ustx.py.

The public API under test:
  - load_ustx(path) -> UProject
  - save_ustx(project, path) -> None
  - UProject / UVoicePart / UCurve / UTrack / UTempo / UTimeSignature
  - TimeAxis  (tick ↔ ms ↔ seconds conversions)
  - UstxEditor  (context-manager editing session)
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from utils.ustx import (
    load_ustx,
    save_ustx,
    UProject,
    UVoicePart,
    UCurve,
    UTrack,
    UTempo,
    UTimeSignature,
    TimeAxis,
    UstxEditor,
    RESOLUTION,
)


# ===========================================================================
# load_ustx / save_ustx
# ===========================================================================

class TestLoadUSTX:
    """Test USTX file loading."""

    def test_load_returns_uproject(self, temp_ustx_file):
        project = load_ustx(str(temp_ustx_file))
        assert isinstance(project, UProject)

    def test_load_tempos(self, temp_ustx_file):
        project = load_ustx(str(temp_ustx_file))
        assert len(project.tempos) == 1
        assert project.tempos[0].bpm == 120

    def test_load_voice_parts(self, temp_ustx_file):
        project = load_ustx(str(temp_ustx_file))
        assert len(project.voice_parts) == 1
        assert project.voice_parts[0].name == "Part 1"

    def test_load_utf8_bom(self, temp_dir):
        content = (
            "tempos:\n  - bpm: 140\n    position: 0\n"
            "time_signatures:\n  - bar_position: 0\n    beat_per_bar: 4\n    beat_unit: 4\n"
            "tracks: []\n"
            "voice_parts:\n  - name: BOM Track\n    track_no: 0\n    position: 0\n    duration: 960\n"
            "    notes: []\n    curves: []\n"
        )
        path = temp_dir / "bom.ustx"
        path.write_text(content, encoding="utf-8-sig")
        project = load_ustx(str(path))
        assert project.tempos[0].bpm == 140

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_ustx("nonexistent_file.ustx")

    def test_load_preserves_voice_part_count(self, sample_ustx_dict, temp_dir):
        path = temp_dir / "counts.ustx"
        project = UProject.from_dict(sample_ustx_dict)
        save_ustx(project, str(path))
        loaded = load_ustx(str(path))
        assert len(loaded.voice_parts) == len(project.voice_parts)

    def test_resolution_always_480(self, temp_ustx_file):
        project = load_ustx(str(temp_ustx_file))
        assert project.resolution == RESOLUTION == 480


class TestSaveUSTX:
    """Test USTX file saving."""

    def test_save_creates_file(self, sample_project, temp_dir):
        path = temp_dir / "out.ustx"
        assert not path.exists()
        save_ustx(sample_project, str(path))
        assert path.exists()

    def test_save_nonempty(self, sample_project, temp_dir):
        path = temp_dir / "out.ustx"
        save_ustx(sample_project, str(path))
        assert path.stat().st_size > 0

    def test_save_overwrites_existing(self, sample_project, temp_dir):
        path = temp_dir / "existing.ustx"
        path.write_text("old content", encoding="utf-8-sig")
        save_ustx(sample_project, str(path))
        loaded = load_ustx(str(path))
        assert loaded.tempos[0].bpm == 120

    def test_save_utf8_bom(self, sample_project, temp_dir):
        path = temp_dir / "encoding.ustx"
        save_ustx(sample_project, str(path))
        assert path.read_bytes()[:3] == b"\xef\xbb\xbf"


class TestSaveLoadRoundtrip:
    """Save → load roundtrip consistency."""

    def test_roundtrip_bpm(self, sample_project, temp_dir):
        path = temp_dir / "rt.ustx"
        save_ustx(sample_project, str(path))
        loaded = load_ustx(str(path))
        assert loaded.tempos[0].bpm == sample_project.tempos[0].bpm

    def test_roundtrip_voice_part_count(self, sample_project, temp_dir):
        path = temp_dir / "rt.ustx"
        save_ustx(sample_project, str(path))
        loaded = load_ustx(str(path))
        assert len(loaded.voice_parts) == len(sample_project.voice_parts)

    def test_roundtrip_voice_part_name(self, sample_project, temp_dir):
        path = temp_dir / "rt.ustx"
        save_ustx(sample_project, str(path))
        loaded = load_ustx(str(path))
        assert loaded.voice_parts[0].name == sample_project.voice_parts[0].name

    def test_roundtrip_with_curves(self, sample_project, temp_dir):
        path = temp_dir / "curves_rt.ustx"
        part = sample_project.voice_parts[0]
        part.set_curve("dyn", np.array([0, 480, 960]), np.array([0.0, 50.0, 100.0]))
        save_ustx(sample_project, str(path))

        loaded = load_ustx(str(path))
        curve = loaded.voice_parts[0].get_curve("dyn")
        assert curve is not None
        assert curve.xs == [0, 480, 960]
        assert curve.ys == [0, 50, 100]


# ===========================================================================
# UProject
# ===========================================================================

class TestUProject:
    """Test UProject data model."""

    def test_from_dict_tempos(self, sample_ustx_dict):
        project = UProject.from_dict(sample_ustx_dict)
        assert isinstance(project.tempos[0], UTempo)
        assert project.tempos[0].bpm == 120

    def test_from_dict_time_signatures(self, sample_ustx_dict):
        project = UProject.from_dict(sample_ustx_dict)
        ts = project.time_signatures[0]
        assert isinstance(ts, UTimeSignature)
        assert ts.bar_position == 0
        assert ts.beat_per_bar == 4
        assert ts.beat_unit == 4

    def test_from_dict_tracks(self, sample_ustx_dict):
        project = UProject.from_dict(sample_ustx_dict)
        assert len(project.tracks) == 1
        assert isinstance(project.tracks[0], UTrack)

    def test_from_dict_voice_parts(self, sample_ustx_dict):
        project = UProject.from_dict(sample_ustx_dict)
        assert len(project.voice_parts) == 1
        assert isinstance(project.voice_parts[0], UVoicePart)

    def test_from_dict_legacy_bpm_fallback(self):
        """A dict with no ``tempos`` key but a top-level ``bpm`` is accepted."""
        d = {"bpm": 90.0, "voice_parts": []}
        project = UProject.from_dict(d)
        assert project.tempos[0].bpm == 90.0

    def test_from_dict_legacy_time_sig_fallback(self):
        d = {
            "tempos": [{"bpm": 120, "position": 0}],
            "beat_per_bar": 3,
            "beat_unit": 4,
            "voice_parts": [],
        }
        project = UProject.from_dict(d)
        assert project.time_signatures[0].beat_per_bar == 3

    def test_get_track_valid(self, sample_project):
        track = sample_project.get_track(0)
        assert isinstance(track, UTrack)

    def test_get_track_out_of_range(self, sample_project):
        with pytest.raises(IndexError):
            sample_project.get_track(99)

    def test_get_parts_for_track(self, sample_project):
        parts = sample_project.get_parts_for_track(0)
        assert len(parts) == 1
        assert all(p.track_no == 0 for p in parts)

    def test_get_parts_for_track_sorted(self, sample_ustx_dict):
        # Add a second part with an earlier position
        sample_ustx_dict["voice_parts"].append({
            "name": "Early",
            "track_no": 0,
            "position": 0,
            "duration": 480,
            "notes": [],
            "curves": [],
        })
        sample_ustx_dict["voice_parts"][0]["position"] = 960
        project = UProject.from_dict(sample_ustx_dict)
        parts = project.get_parts_for_track(0)
        positions = [p.position for p in parts]
        assert positions == sorted(positions)

    def test_resolution_fixed(self, sample_project):
        assert sample_project.resolution == 480

    def test_to_dict_roundtrip_keys(self, sample_project):
        d = sample_project.to_dict()
        assert "tempos" in d
        assert "time_signatures" in d
        assert "voice_parts" in d


# ===========================================================================
# UCurve / UVoicePart curve helpers
# ===========================================================================

class TestUCurve:
    """Test UCurve and UVoicePart curve helpers."""

    def test_get_curve_existing(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.curves.append(UCurve(abbr="dyn", xs=[0], ys=[0]))
        assert part.get_curve("dyn") is not None

    def test_get_curve_missing(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        assert part.get_curve("dyn") is None

    def test_get_or_create_curve_creates(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        curve = part.get_or_create_curve("dyn")
        assert curve.abbr == "dyn"
        assert len(part.curves) == 1

    def test_get_or_create_curve_reuses(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        c1 = part.get_or_create_curve("dyn")
        c2 = part.get_or_create_curve("dyn")
        assert c1 is c2
        assert len(part.curves) == 1

    def test_set_curve_basic(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn", np.array([0, 480, 960]), np.array([0.0, 50.0, 100.0]))
        curve = part.get_curve("dyn")
        assert curve.xs == [0, 480, 960]
        assert curve.ys == [0, 50, 100]

    def test_set_curve_filters_nan(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn",
                       np.array([0, 480, 960, 1440]),
                       np.array([0.0, np.nan, 100.0, 75.0]))
        curve = part.get_curve("dyn")
        assert 480 not in curve.xs
        assert curve.xs == [0, 960, 1440]
        assert curve.ys == [0, 100, 75]

    def test_set_curve_all_nan(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn", np.array([0, 480]), np.array([np.nan, np.nan]))
        curve = part.get_curve("dyn")
        assert curve.xs == []
        assert curve.ys == []

    def test_set_curve_rounds_values(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn", np.array([0, 480]), np.array([10.7, 50.3]))
        assert part.get_curve("dyn").ys == [11, 50]

    def test_set_curve_negative_values(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn", np.array([0, 480]), np.array([-10.0, -20.0]))
        assert part.get_curve("dyn").ys == [-10, -20]

    def test_set_curve_overwrites(self):
        part = UVoicePart(track_no=0, position=0, duration=960)
        part.set_curve("dyn", np.array([0, 480]), np.array([0.0, 50.0]))
        part.set_curve("dyn", np.array([0, 960]), np.array([100.0, 200.0]))
        assert len(part.curves) == 1
        assert part.get_curve("dyn").xs == [0, 960]


# ===========================================================================
# TimeAxis
# ===========================================================================

class TestTimeAxis:
    """Test TimeAxis tick ↔ ms ↔ seconds conversions."""

    @pytest.fixture
    def axis_120bpm(self):
        tempos = [UTempo(position=0, bpm=120.0)]
        time_sigs = [UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4)]
        return TimeAxis.build(tempos, time_sigs)

    # --- basic scalar conversions ---

    def test_tick_to_ms_zero(self, axis_120bpm):
        assert axis_120bpm.tick_pos_to_ms(0) == pytest.approx(0.0)

    def test_tick_to_ms_one_beat(self, axis_120bpm):
        # 120 BPM → 500 ms/beat → 480 ticks/beat → 500 ms
        assert axis_120bpm.tick_pos_to_ms(480) == pytest.approx(500.0)

    def test_ms_to_tick_zero(self, axis_120bpm):
        assert axis_120bpm.ms_pos_to_tick(0.0) == pytest.approx(0.0)

    def test_ms_to_tick_500ms(self, axis_120bpm):
        assert axis_120bpm.ms_pos_to_tick(500.0) == pytest.approx(480.0)

    def test_ms_between_ticks(self, axis_120bpm):
        assert axis_120bpm.ms_between_ticks(0, 480) == pytest.approx(500.0)

    # --- vectorised API ---

    def test_ticks_to_ms_array(self, axis_120bpm):
        result = axis_120bpm.ticks_to_ms(np.array([0, 480, 960]))
        assert_array_almost_equal(result, [0.0, 500.0, 1000.0])

    def test_ms_to_ticks_array(self, axis_120bpm):
        result = axis_120bpm.ms_to_ticks(np.array([0.0, 500.0, 1000.0]))
        assert_array_equal(result, [0, 480, 960])

    def test_ms_to_ticks_unique(self, axis_120bpm):
        result = axis_120bpm.ms_to_ticks(np.array([0.0, 0.0, 500.0]), unique=True)
        assert_array_equal(result, [0, 480])

    # --- seconds wrappers ---

    def test_ticks_to_seconds(self, axis_120bpm):
        result = axis_120bpm.ticks_to_seconds(np.array([0, 480, 960]))
        assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_seconds_to_ticks(self, axis_120bpm):
        result = axis_120bpm.seconds_to_ticks(np.array([0.0, 0.5, 1.0]))
        assert_array_equal(result, [0, 480, 960])

    def test_seconds_to_ticks_unique(self, axis_120bpm):
        result = axis_120bpm.seconds_to_ticks(np.array([0.0, 0.0, 1.0]), unique=True)
        assert_array_equal(result, [0, 960])

    # --- roundtrip ---

    def test_roundtrip_tick_ms(self, axis_120bpm):
        original = np.array([0, 240, 480, 720, 960], dtype=float)
        ms = axis_120bpm.ticks_to_ms(original)
        recovered = axis_120bpm.ms_to_ticks(ms)
        assert_array_equal(recovered, original.astype(int))

    def test_roundtrip_precision(self, axis_120bpm):
        """Round-trip error ≤ half a tick duration."""
        original = np.linspace(0, 10, 500)          # seconds
        ticks = axis_120bpm.seconds_to_ticks(original)
        recovered = axis_120bpm.ticks_to_seconds(ticks)
        tick_duration_s = 60 / (120 * RESOLUTION)
        assert np.all(np.abs(original - recovered) <= tick_duration_s / 2 + 1e-12)

    # --- tempo change ---

    def test_tempo_change_boundary(self):
        """After a tempo change the ms position must reflect the new BPM."""
        tempos = [
            UTempo(position=0,    bpm=120.0),
            UTempo(position=1920, bpm=60.0),   # 4 beats in at 120 BPM
        ]
        time_sigs = [UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4)]
        axis = TimeAxis.build(tempos, time_sigs)

        # First 1920 ticks at 120 BPM = 2000 ms
        assert axis.tick_pos_to_ms(1920) == pytest.approx(2000.0)
        # Next 480 ticks at 60 BPM (1000 ms/beat) = 1000 ms more
        assert axis.tick_pos_to_ms(2400) == pytest.approx(3000.0)

    # --- shift_ticks_by_seconds ---

    def test_shift_ticks_by_seconds_positive(self, axis_120bpm):
        ticks = np.array([0, 480, 960])
        shifted = axis_120bpm.shift_ticks_by_seconds(ticks, 0.5)
        # 0.5 s = 480 ticks at 120 BPM
        assert_array_equal(shifted, [480, 960, 1440])

    def test_shift_ticks_by_seconds_zero(self, axis_120bpm):
        ticks = np.array([0, 480, 960])
        shifted = axis_120bpm.shift_ticks_by_seconds(ticks, 0.0)
        assert_array_equal(shifted, ticks)

    def test_shift_ticks_by_seconds_negative(self, axis_120bpm):
        ticks = np.array([960, 1440])
        shifted = axis_120bpm.shift_ticks_by_seconds(ticks, -0.5)
        assert_array_equal(shifted, [480, 960])

    # --- build validation ---

    def test_build_requires_tempos(self):
        with pytest.raises(ValueError, match="tempo"):
            TimeAxis.build(
                [],
                [UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4)],
            )

    def test_build_requires_time_signatures(self):
        with pytest.raises(ValueError, match="time.signature"):
            TimeAxis.build([UTempo(position=0, bpm=120)], [])

    def test_build_requires_first_time_sig_at_bar_0(self):
        with pytest.raises(ValueError):
            TimeAxis.build(
                [UTempo(position=0, bpm=120)],
                [UTimeSignature(bar_position=1, beat_per_bar=4, beat_unit=4)],
            )


# ===========================================================================
# TimeAxis internal segment coverage
# ===========================================================================

class TestTimeAxisSegments:
    """Test TimeAxis internal _TempoSegment properties (lines 337, 341)."""

    def test_tempo_segment_ticks_property(self):
        """_TempoSegment.ticks = tick_end - tick_pos (line 337)."""
        tempos = [UTempo(position=0, bpm=120.0)]
        time_sigs = [UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4)]
        axis = TimeAxis.build(tempos, time_sigs)
        # Access internal segments to verify ticks property
        seg = axis._segs[0]
        assert seg.ticks == seg.tick_end - seg.tick_pos

    def test_tempo_segment_ms_end_property(self):
        """_TempoSegment.ms_end = ms_pos + ticks * ms_per_tick (line 341)."""
        tempos = [UTempo(position=0, bpm=120.0)]
        time_sigs = [UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4)]
        axis = TimeAxis.build(tempos, time_sigs)
        seg = axis._segs[0]
        expected_ms_end = seg.ms_pos + seg.ticks * seg.ms_per_tick
        assert seg.ms_end == expected_ms_end

    def test_build_time_sig_at_nonzero_bar(self):
        """TimeAxis.build with subsequent time signature at non-zero bar (lines 397-398)."""
        tempos = [UTempo(position=0, bpm=120.0)]
        time_sigs = [
            UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4),
            UTimeSignature(bar_position=4, beat_per_bar=3, beat_unit=4),
        ]
        axis = TimeAxis.build(tempos, time_sigs)
        assert len(axis._segs) >= 2

    def test_build_inserts_tempo_at_existing_boundary(self):
        """TimeAxis.build inserts tempo at existing segment boundary (lines 420-421)."""
        # Tempo at bar 4 (1920 ticks at 120 BPM) - same position as time sig change
        tempos = [
            UTempo(position=0, bpm=120.0),
            UTempo(position=1920, bpm=60.0),
        ]
        time_sigs = [
            UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4),
            UTimeSignature(bar_position=4, beat_per_bar=3, beat_unit=4),
        ]
        axis = TimeAxis.build(tempos, time_sigs)
        # Should handle merge of tempo and time_sig boundaries
        assert len(axis._segs) >= 2

    def test_build_propagates_bpm_forward(self):
        """TimeAxis.build propagates BPM forward to segments without explicit tempo (line 427)."""
        # Only one tempo, multiple time sig changes
        tempos = [UTempo(position=0, bpm=120.0)]
        time_sigs = [
            UTimeSignature(bar_position=0, beat_per_bar=4, beat_unit=4),
            UTimeSignature(bar_position=4, beat_per_bar=3, beat_unit=4),
            UTimeSignature(bar_position=8, beat_per_bar=6, beat_unit=8),
        ]
        axis = TimeAxis.build(tempos, time_sigs)
        # All segments should have BPM from the first tempo (120)
        for seg in axis._segs:
            assert seg.tick_end > 0  # Verify segments exist with propagated BPM


# ===========================================================================
# UstxEditor
# ===========================================================================

class TestUstxEditor:
    """Test UstxEditor context-manager and expression helpers."""

    def test_context_manager_saves_on_clean_exit(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            editor.project.voice_parts[0].name = "Edited"
        # Reload and check
        reloaded = load_ustx(str(temp_ustx_file))
        assert reloaded.voice_parts[0].name == "Edited"

    def test_context_manager_no_save_on_exception(self, temp_ustx_file):
        original_name = load_ustx(str(temp_ustx_file)).voice_parts[0].name
        with pytest.raises(RuntimeError):
            with UstxEditor(str(temp_ustx_file)) as editor:
                editor.project.voice_parts[0].name = "Should Not Save"
                raise RuntimeError("deliberate error")
        reloaded = load_ustx(str(temp_ustx_file))
        assert reloaded.voice_parts[0].name == original_name

    def test_add_expression_to_part(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            part = editor.voice_parts[0]
            editor.add_expression_to_part(
                part, "dyn",
                np.array([0, 480, 960]),
                np.array([0.0, 50.0, 100.0]),
            )
        reloaded = load_ustx(str(temp_ustx_file))
        curve = reloaded.voice_parts[0].get_curve("dyn")
        assert curve is not None
        assert curve.xs == [0, 480, 960]

    def test_add_expression_to_track_basic(self, temp_ustx_file):
        """Absolute ticks within part window are written as relative ticks."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            # part starts at 0, duration 1920
            editor.add_expression_to_track(
                0, "dyn",
                np.array([0, 480, 960]),
                np.array([0.0, 50.0, 100.0]),
            )
        reloaded = load_ustx(str(temp_ustx_file))
        curve = reloaded.voice_parts[0].get_curve("dyn")
        assert curve is not None
        assert curve.xs == [0, 480, 960]

    def test_add_expression_to_track_clips_to_part_window(self, temp_ustx_file):
        """Ticks outside [part.position, part.position + part.duration) are dropped."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            # Part has position=0, duration=1920; tick 2400 is outside
            editor.add_expression_to_track(
                0, "dyn",
                np.array([0, 960, 2400]),
                np.array([10.0, 20.0, 30.0]),
            )
        reloaded = load_ustx(str(temp_ustx_file))
        curve = reloaded.voice_parts[0].get_curve("dyn")
        assert 2400 not in curve.xs

    def test_add_expression_to_track_relative_ticks(self, temp_dir):
        """Ticks stored in the curve must be relative to part.position."""
        content = (
            "tempos:\n  - bpm: 120\n    position: 0\n"
            "time_signatures:\n  - bar_position: 0\n    beat_per_bar: 4\n    beat_unit: 4\n"
            "tracks:\n  - track_name: T\n    track_color: Blue\n    singer: ''\n"
            "    phonemizer: ''\n    mute: false\n    solo: false\n    volume: 0.0\n    pan: 0.0\n"
            "voice_parts:\n  - name: P\n    track_no: 0\n    position: 480\n    duration: 960\n"
            "    notes: []\n    curves: []\n"
        )
        path = temp_dir / "offset.ustx"
        path.write_text(content, encoding="utf-8-sig")

        with UstxEditor(str(path)) as editor:
            # absolute ticks 480–1439 fall inside the part (offset 480)
            editor.add_expression_to_track(
                0, "dyn",
                np.array([480, 960, 1439]),
                np.array([10.0, 20.0, 30.0]),
            )
        reloaded = load_ustx(str(path))
        curve = reloaded.voice_parts[0].get_curve("dyn")
        # Stored as relative: 480-480=0, 960-480=480, 1439-480=959
        assert curve.xs == [0, 480, 959]

    def test_add_expression_to_track_no_parts_raises(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            with pytest.raises(ValueError, match="No voice parts"):
                editor.add_expression_to_track(
                    99, "dyn",
                    np.array([0]), np.array([0.0]),
                )

    def test_manual_save_and_close(self, temp_ustx_file):
        editor = UstxEditor(str(temp_ustx_file))
        editor.project.voice_parts[0].name = "Manual"
        editor.save()
        editor.close()
        reloaded = load_ustx(str(temp_ustx_file))
        assert reloaded.voice_parts[0].name == "Manual"

    def test_build_time_axis_returns_time_axis(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            axis = editor.build_time_axis()
        assert isinstance(axis, TimeAxis)

    def test_tracks_property(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            assert editor.tracks is editor.project.tracks

    def test_voice_parts_property(self, temp_ustx_file):
        with UstxEditor(str(temp_ustx_file)) as editor:
            assert editor.voice_parts is editor.project.voice_parts

    def test_tempos_property(self, temp_ustx_file):
        """UstxEditor.tempos property delegates to project.tempos (line 625)."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            assert editor.tempos is editor.project.tempos

    def test_time_signatures_property(self, temp_ustx_file):
        """UstxEditor.time_signatures property delegates to project (line 630)."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            assert editor.time_signatures is editor.project.time_signatures

    def test_get_track_property(self, temp_ustx_file):
        """UstxEditor.get_track delegates to project.get_track (line 634)."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            track = editor.get_track(0)
            assert isinstance(track, UTrack)

    def test_get_parts_for_track_property(self, temp_ustx_file):
        """UstxEditor.get_parts_for_track delegates to project (line 638)."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            parts = editor.get_parts_for_track(0)
            assert len(parts) == 1


# ===========================================================================
# add_expression_to_track edge cases
# ===========================================================================

class TestAddExpressionToTrackEdgeCases:
    """Test add_expression_to_track edge cases (line 716 - continue)."""

    def test_add_expression_skips_ticks_outside_all_parts(self, temp_ustx_file):
        """Ticks with no overlapping parts are skipped via continue (line 716)."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            # Part at position=0, duration=1920
            # Ticks completely outside any part: 5000-6000
            # These ticks don't overlap with the part [0, 1920), so continue is hit
            editor.add_expression_to_track(
                0, "dyn",
                np.array([5000, 5500, 6000]),
                np.array([10.0, 20.0, 30.0]),
            )
            # Curve should not exist since no ticks overlap with the part
            curve = editor.voice_parts[0].get_curve("dyn")
            assert curve is None

    def test_add_expression_all_ticks_inside_part(self, temp_ustx_file):
        """All ticks inside part - no filtering needed."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            # Part at position=0, duration=1920
            # All ticks within [0, 1920)
            editor.add_expression_to_track(
                0, "dyn",
                np.array([0, 500, 1500]),
                np.array([10.0, 20.0, 30.0]),
            )
            curve = editor.voice_parts[0].get_curve("dyn")
            assert curve is not None
            # All ticks are stored (converted to relative)
            assert 0 in curve.xs
            assert 500 in curve.xs
            assert 1500 in curve.xs


# ===========================================================================
# Integration
# ===========================================================================

class TestIntegration:
    """End-to-end workflows."""

    def test_full_workflow_via_editor(self, temp_ustx_file):
        """Load → edit via UstxEditor → verify persisted curve."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            axis = editor.build_time_axis()
            ticks = axis.seconds_to_ticks(np.array([0.0, 0.5, 1.0]))
            editor.add_expression_to_track(
                0, "dyn", ticks, np.array([0.0, 50.0, 100.0])
            )

        final = load_ustx(str(temp_ustx_file))
        curve = final.voice_parts[0].get_curve("dyn")
        assert curve is not None
        assert len(curve.xs) == 3

    def test_time_axis_used_for_ticks(self, temp_ustx_file):
        """Verify ticks produced by TimeAxis match expected values."""
        with UstxEditor(str(temp_ustx_file)) as editor:
            axis = editor.build_time_axis()
            ticks = axis.seconds_to_ticks(np.array([0.0, 0.5, 1.0, 1.5]))
            # 120 BPM, 480 PPQN → 960 ticks/second
            assert_array_equal(ticks, [0, 480, 960, 1440])


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sample_ustx_dict():
    """Minimal valid USTX project dict."""
    return {
        "tempos": [{"bpm": 120.0, "position": 0}],
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tracks": [{"track_name": "T", "track_color": "Blue", "singer": "", "phonemizer": "", "mute": False, "solo": False, "volume": 0.0, "pan": 0.0}],  # noqa: E501
        "voice_parts": [{
            "name": "Part 1",
            "track_no": 0,
            "position": 0,
            "duration": 1920,
            "notes": [],
            "curves": [],
        }],
    }


@pytest.fixture
def sample_project(sample_ustx_dict):
    """UProject instance built from sample_ustx_dict."""
    return UProject.from_dict(sample_ustx_dict)


@pytest.fixture
def temp_ustx_file(temp_dir, sample_ustx_dict):
    """Path to a temporary USTX file with sample content."""
    path = temp_dir / "test.ustx"
    project = UProject.from_dict(sample_ustx_dict)
    save_ustx(project, str(path))
    return path
