"""
ustx.py — OpenUtau USTX file I/O and editing utilities.

Data structures mirror the OpenUtau C# models (UProject, UTempo,
UTimeSignature, UVoicePart, UCurve) as found in:
  OpenUtau.Core/Ustx/UProject.cs
  OpenUtau.Core/Ustx/UTrack.cs
  OpenUtau.Core/Ustx/UPart.cs
  OpenUtau.Core/Ustx/UCurve.cs
  OpenUtau.Core/Util/TimeAxis.cs

TimeAxis provides tick ↔ millisecond conversion that respects all tempo and
time-signature changes, matching the BuildSegments / TickPosToMsPos logic in
OpenUtau.Core/Util/TimeAxis.cs.

Notes on the format:
  - ``resolution`` is always 480 ppqn (hardcoded in UProject.cs); it is
    *not* stored in the YAML file.
  - TimeAxis works in **milliseconds** internally (ms_per_tick = 60000 / (bpm * resolution)).
  - YAML keys use snake_case (e.g. ``voice_parts``, ``track_no``,
    ``beat_per_bar``, ``bar_position``, ``time_signatures``).
  - Legacy top-level ``bpm`` / ``beat_per_bar`` / ``beat_unit`` fields exist for
    files predating ustx v0.6 but are marked [Obsolete] in C#.
"""

from __future__ import annotations

import bisect
import logging
from typing import Optional
from dataclasses import dataclass, field

import oyaml
import numpy as np
from filelock import FileLock
from yamlcore import CoreLoader

log = logging.getLogger(__name__)

RESOLUTION = 480          # pulses per quarter note — hardcoded in UProject.cs
MS_PER_MIN = 60_000.0     # milliseconds per minute


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UTempo:
    """A tempo event. ``position`` is a tick offset from the project start."""
    position: int
    bpm: float

    @classmethod
    def from_dict(cls, d: dict) -> "UTempo":
        return cls(position=int(d["position"]), bpm=float(d["bpm"]))

    def to_dict(self) -> dict:
        return {"position": self.position, "bpm": self.bpm}


@dataclass
class UTimeSignature:
    """
    A time-signature event.

    ``bar_position`` is a 0-based bar index.
    """
    bar_position: int
    beat_per_bar: int
    beat_unit: int

    @classmethod
    def from_dict(cls, d: dict) -> "UTimeSignature":
        return cls(
            bar_position=int(d["bar_position"]),
            beat_per_bar=int(d["beat_per_bar"]),
            beat_unit=int(d["beat_unit"]),
        )

    def to_dict(self) -> dict:
        return {
            "bar_position": self.bar_position,
            "beat_per_bar": self.beat_per_bar,
            "beat_unit": self.beat_unit,
        }


@dataclass
class UCurve:
    """Expression curve inside a voice part (xs = ticks, ys = integer values)."""
    abbr: str
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "UCurve":
        return cls(
            abbr=str(d["abbr"]),
            xs=list(d.get("xs", [])),
            ys=list(d.get("ys", [])),
        )

    def to_dict(self) -> dict:
        return {"xs": self.xs, "ys": self.ys, "abbr": self.abbr}


@dataclass
class UTrack:
    """
    A track (UTrack in C#).  Holds singer / phonemizer / renderer metadata.
    Does *not* store notes or curves — those live in UVoicePart.

    ``track_no`` is the 0-based index of this track in ``UProject.tracks``,
    set after loading (mirrors ``TrackNo = project.tracks.IndexOf(this)``).
    """
    track_no: int = 0           # populated by UProject.from_dict
    track_name: str = "New Track"
    track_color: str = "Blue"
    singer: str = ""
    phonemizer: str = ""
    mute: bool = False
    solo: bool = False
    volume: float = 0.0
    pan: float = 0.0
    _raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict, track_no: int) -> "UTrack":
        return cls(
            track_no=track_no,
            track_name=str(d.get("track_name", "New Track")),
            track_color=str(d.get("track_color", "Blue")),
            singer=str(d.get("singer", "") or ""),
            phonemizer=str(d.get("phonemizer", "") or ""),
            mute=bool(d.get("mute", False)),
            solo=bool(d.get("solo", False)),
            volume=float(d.get("volume", 0.0)),
            pan=float(d.get("pan", 0.0)),
            _raw=d,
        )

    def to_dict(self) -> dict:
        out = dict(self._raw)
        out["track_name"] = self.track_name
        out["track_color"] = self.track_color
        out["singer"] = self.singer
        out["phonemizer"] = self.phonemizer
        out["mute"] = self.mute
        out["solo"] = self.solo
        out["volume"] = self.volume
        out["pan"] = self.pan
        return out


@dataclass
class UVoicePart:
    """
    A voice part (UVoicePart in C#).

    ``track_no`` is 0-based.
    ``position`` is the tick offset of the part start within the project.
    """
    track_no: int
    position: int
    duration: int
    name: str = ""
    curves: list[UCurve] = field(default_factory=list)
    # Preserves all unrecognised YAML keys for lossless round-trip
    _raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> "UVoicePart":
        return cls(
            track_no=int(d.get("track_no", 0)),
            position=int(d.get("position", 0)),
            duration=int(d.get("duration", 0)),
            name=str(d.get("name", "")),
            curves=[UCurve.from_dict(c) for c in d.get("curves", [])],
            _raw=d,
        )

    def to_dict(self) -> dict:
        out = dict(self._raw)
        out["track_no"] = self.track_no
        out["position"] = self.position
        out["duration"] = self.duration
        out["name"] = self.name
        out["curves"] = [c.to_dict() for c in self.curves]
        return out

    # ------------------------------------------------------------------
    # Curve helpers
    # ------------------------------------------------------------------

    def get_curve(self, abbr: str) -> Optional[UCurve]:
        for c in self.curves:
            if c.abbr == abbr:
                return c
        return None

    def get_or_create_curve(self, abbr: str) -> UCurve:
        curve = self.get_curve(abbr)
        if curve is None:
            curve = UCurve(abbr=abbr)
            self.curves.append(curve)
        return curve

    def set_curve(
        self,
        abbr: str,
        ticks: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Overwrite the xs/ys of *abbr* from numpy arrays, skipping NaN frames.

        Args:
            abbr:   Expression abbreviation, e.g. ``"dyn"``.
            ticks:  1-D integer array of tick positions.
            values: 1-D float array of curve values; NaN entries are dropped.
        """
        mask = ~np.isnan(values)
        curve = self.get_or_create_curve(abbr)
        curve.xs = ticks[mask].astype(int).tolist()
        curve.ys = np.round(values[mask]).astype(int).tolist()


@dataclass
class UProject:
    """
    Top-level USTX project.

    ``resolution`` is always 480 (hardcoded in C#); it is not read from or
    written to the YAML file.

    ``voice_parts`` is a flat list ordered as they appear under the YAML key
    ``voiceParts``.
    """
    tempos: list[UTempo]
    time_signatures: list[UTimeSignature]
    tracks: list[UTrack]
    voice_parts: list[UVoicePart]
    _raw: dict = field(default_factory=dict, repr=False, compare=False)

    resolution: int = field(default=RESOLUTION, init=False)

    @classmethod
    def from_dict(cls, d: dict) -> "UProject":
        tempos = [UTempo.from_dict(t) for t in d.get("tempos", [])]
        if not tempos:
            tempos = [UTempo(position=0, bpm=float(d.get("bpm", 120.0)))]

        time_sigs = [UTimeSignature.from_dict(ts) for ts in d.get("time_signatures", [])]
        if not time_sigs:
            time_sigs = [
                UTimeSignature(
                    bar_position=0,
                    beat_per_bar=int(d.get("beat_per_bar", 4)),
                    beat_unit=int(d.get("beat_unit", 4)),
                )
            ]

        tracks = [UTrack.from_dict(t, i) for i, t in enumerate(d.get("tracks", []))]
        voice_parts = [UVoicePart.from_dict(vp) for vp in d.get("voice_parts", [])]

        return cls(
            tempos=tempos,
            time_signatures=time_sigs,
            tracks=tracks,
            voice_parts=voice_parts,
            _raw=d,
        )

    def to_dict(self) -> dict:
        out = dict(self._raw)
        out["tempos"] = [t.to_dict() for t in self.tempos]
        out["time_signatures"] = [ts.to_dict() for ts in self.time_signatures]
        out["tracks"] = [t.to_dict() for t in self.tracks]
        out["voice_parts"] = [vp.to_dict() for vp in self.voice_parts]
        return out

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_track(self, track_no: int) -> UTrack:
        """Return the track at 0-based index *track_no*.

        Raises:
            IndexError: if *track_no* is out of range.
        """
        if track_no < 0 or track_no >= len(self.tracks):
            raise IndexError(
                f"track_no {track_no} is out of range "
                f"(project has {len(self.tracks)} track(s))."
            )
        return self.tracks[track_no]

    def get_parts_for_track(self, track_no: int) -> list[UVoicePart]:
        """Return all voice parts whose ``track_no`` matches *track_no* (0-based).

        A track can own multiple parts (segments); this returns them all,
        sorted by position.
        """
        return sorted(
            [vp for vp in self.voice_parts if vp.track_no == track_no],
            key=lambda p: p.position,
        )

    def build_time_axis(self) -> "TimeAxis":
        """Build and return a :class:`TimeAxis` for this project."""
        return TimeAxis.build(self.tempos, self.time_signatures)


# ---------------------------------------------------------------------------
# TimeAxis  (mirrors OpenUtau.Core/Util/TimeAxis.cs → BuildSegments)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TempoSegment:
    tick_pos: int
    tick_end: int           # exclusive upper bound
    bpm: float
    ms_pos: float           # absolute ms at segment start
    ms_per_tick: float
    ticks_per_ms: float

    @property
    def ticks(self) -> int:
        return self.tick_end - self.tick_pos

    @property
    def ms_end(self) -> float:
        return self.ms_pos + self.ticks * self.ms_per_tick


class TimeAxis:
    """
    Piecewise tick ↔ millisecond converter that faithfully replicates the
    ``BuildSegments`` / ``TickPosToMsPos`` / ``MsPosToTickPos`` logic from
    ``OpenUtau.Core/Util/TimeAxis.cs``.

    The C# implementation merges time-signature segment boundaries with tempo
    events before computing absolute millisecond offsets.  We replicate that
    merge here so segment boundaries are identical.

    Usage::

        axis = project.build_time_axis()
        ms   = axis.ticks_to_ms(ticks_array)
        tick = axis.ms_to_ticks(ms_array)

    ``seconds_to_ticks`` / ``ticks_to_seconds`` wrappers are also provided.
    """

    def __init__(self, segments: list[_TempoSegment]) -> None:
        self._segs = segments                          # sorted by tick_pos
        self._tick_starts = [s.tick_pos for s in segments]
        self._ms_starts   = [s.ms_pos   for s in segments]

    # ------------------------------------------------------------------
    # Factory — replicates BuildSegments
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        tempos: list[UTempo],
        time_signatures: list[UTimeSignature],
        resolution: int = RESOLUTION,
    ) -> "TimeAxis":
        """Build a TimeAxis from project tempo and time-signature lists."""
        if not tempos:
            raise ValueError("At least one tempo event is required.")
        if not time_signatures:
            raise ValueError("At least one time-signature event is required.")

        sorted_ts  = sorted(time_signatures, key=lambda ts: ts.bar_position)
        sorted_bpm = sorted(tempos, key=lambda t: t.position)

        # --- step 1: compute the tick position of each time-signature change ---
        ts_tick: list[int] = []
        ticks_per_bar: list[int] = []
        for i, ts in enumerate(sorted_ts):
            if i == 0:
                if ts.bar_position != 0:
                    raise ValueError("First time signature must be at bar 0.")
                ts_tick.append(0)
            else:
                prev_bar = sorted_ts[i - 1].bar_position
                ts_tick.append(
                    ts_tick[-1]
                    + ticks_per_bar[-1] * (ts.bar_position - prev_bar)
                )
            ticks_per_bar.append(
                resolution * 4 * ts.beat_per_bar // ts.beat_unit
            )

        # --- step 2: merge time-sig boundary ticks with tempo-event ticks ---
        # Replicates the C# loop that inserts / updates TempoSegments.
        seg_ticks: list[int]   = list(ts_tick)
        seg_bpms:  list[float] = [0.0] * len(ts_tick)

        for tempo in sorted_bpm:
            p = tempo.position
            idx = bisect.bisect_left(seg_ticks, p)
            if idx < len(seg_ticks) and seg_ticks[idx] == p:
                seg_bpms[idx] = tempo.bpm
            elif idx == len(seg_ticks):
                seg_ticks.append(p)
                seg_bpms.append(tempo.bpm)
            else:
                seg_ticks.insert(idx, p)
                seg_bpms.insert(idx, tempo.bpm)

        # Propagate BPM forward into time-sig boundary segments that have
        # no explicit tempo event (they inherit the preceding tempo).
        for i in range(1, len(seg_bpms)):
            if seg_bpms[i] == 0.0:
                seg_bpms[i] = seg_bpms[i - 1]

        # --- step 3: compute absolute ms_pos for each segment ---
        n = len(seg_ticks)
        ms_pos_arr = [0.0] * n
        for i in range(1, n):
            dt = seg_ticks[i] - seg_ticks[i - 1]
            ms_per_tick_prev = MS_PER_MIN / (seg_bpms[i - 1] * resolution)
            ms_pos_arr[i] = ms_pos_arr[i - 1] + dt * ms_per_tick_prev

        # --- step 4: build immutable _TempoSegment objects ---
        segments: list[_TempoSegment] = []
        for i in range(n):
            bpm = seg_bpms[i]
            ms_per_tick = MS_PER_MIN / (bpm * resolution)
            tick_end = seg_ticks[i + 1] if i + 1 < n else 2 ** 31 - 1
            segments.append(
                _TempoSegment(
                    tick_pos=seg_ticks[i],
                    tick_end=tick_end,
                    bpm=bpm,
                    ms_pos=ms_pos_arr[i],
                    ms_per_tick=ms_per_tick,
                    ticks_per_ms=1.0 / ms_per_tick,
                )
            )
        return cls(segments)

    # ------------------------------------------------------------------
    # Scalar converters (mirror TickPosToMsPos / MsPosToTickPos in C#)
    # ------------------------------------------------------------------

    def _seg_at_tick(self, tick: float) -> _TempoSegment:
        idx = bisect.bisect_right(self._tick_starts, tick) - 1
        return self._segs[max(idx, 0)]

    def _seg_at_ms(self, ms: float) -> _TempoSegment:
        idx = bisect.bisect_right(self._ms_starts, ms) - 1
        return self._segs[max(idx, 0)]

    def tick_pos_to_ms(self, tick: float) -> float:
        """Convert a tick position to milliseconds (mirrors TickPosToMsPos)."""
        seg = self._seg_at_tick(tick)
        return seg.ms_pos + seg.ms_per_tick * (tick - seg.tick_pos)

    def ms_pos_to_tick(self, ms: float) -> float:
        """Convert a ms position to (non-integer) ticks (mirrors MsPosToNonExactTickPos)."""
        seg = self._seg_at_ms(ms)
        return seg.tick_pos + (ms - seg.ms_pos) * seg.ticks_per_ms

    def ms_between_ticks(self, tick_start: float, tick_end: float) -> float:
        """Duration in ms between two tick positions (mirrors MsBetweenTickPos)."""
        return self.tick_pos_to_ms(tick_end) - self.tick_pos_to_ms(tick_start)

    # ------------------------------------------------------------------
    # Vectorised numpy API
    # ------------------------------------------------------------------

    def ticks_to_ms(self, ticks: np.ndarray | float) -> np.ndarray:
        """Convert tick values to milliseconds (vectorised)."""
        return np.vectorize(self.tick_pos_to_ms)(np.asarray(ticks, dtype=float))

    def ms_to_ticks(
        self,
        ms: np.ndarray | float,
        *,
        unique: bool = False,
    ) -> np.ndarray:
        """Convert millisecond positions to integer ticks (vectorised).

        Args:
            ms:     Millisecond positions.
            unique: Return sorted deduplicated ticks when ``True``.
        """
        ticks = np.round(
            np.vectorize(self.ms_pos_to_tick)(np.asarray(ms, dtype=float))
        ).astype(int)
        return np.unique(ticks) if unique else ticks

    def ticks_to_seconds(self, ticks: np.ndarray | float) -> np.ndarray:
        """Convenience wrapper: ticks → seconds."""
        return self.ticks_to_ms(ticks) / 1000.0

    def seconds_to_ticks(
        self,
        times: np.ndarray | float,
        *,
        unique: bool = False,
    ) -> np.ndarray:
        """Convenience wrapper: seconds → integer ticks."""
        return self.ms_to_ticks(np.asarray(times, dtype=float) * 1000.0, unique=unique)

    def shift_ticks_by_seconds(
        self,
        ticks: np.ndarray,
        offset_seconds: float,
    ) -> np.ndarray:
        """Shift tick positions by *offset_seconds* seconds.

        Unlike ``ticks + seconds_to_ticks(offset)``, this correctly handles
        tempo changes: each tick is converted back to seconds, shifted, then
        re-converted to ticks — so the shift is always measured in real time,
        not in a fixed-tempo approximation.

        Args:
            ticks:          1-D integer array of tick positions.
            offset_seconds: Time shift in seconds (positive = delay).

        Returns:
            Shifted integer tick positions.
        """
        times = self.ticks_to_seconds(np.asarray(ticks, dtype=float))
        return self.seconds_to_ticks(times + offset_seconds)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_ustx(ustx_path: str) -> UProject:
    """Parse a USTX file and return a :class:`UProject`.

    Args:
        ustx_path: Path to the ``.ustx`` file.
    """
    with open(ustx_path, "r", encoding="utf-8-sig") as fh:
        raw = oyaml.load(fh.read(), CoreLoader)
    project = UProject.from_dict(raw)
    log.debug(
        "Loaded USTX from %s  (%d voice part(s), %d tempo(s))",
        ustx_path, len(project.voice_parts), len(project.tempos),
    )
    return project


def save_ustx(project: UProject, ustx_path: str) -> None:
    """Serialise *project* back to a USTX file, preserving key order.

    Args:
        project:   The project to save.
        ustx_path: Destination path.
    """
    output = oyaml.dump(project.to_dict(), Dumper=oyaml.Dumper, allow_unicode=True)
    with open(ustx_path, "w+", encoding="utf-8-sig") as fh:
        fh.write(output)
    log.debug("Saved USTX to %s", ustx_path)


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

class UstxEditor:
    """
    RAII wrapper that holds an exclusive file lock for the duration of an
    editing session and exposes the parsed :class:`UProject`.

    Preferred usage — context manager (auto-saves on clean exit)::

        with UstxEditor("song.ustx") as editor:
            axis  = editor.build_time_axis()
            ticks = axis.seconds_to_ticks(times_array)
            # write relative-tick data to one part:
            editor.add_expression_to_part(editor.voice_parts[0], "dyn", ticks, values)
            # write absolute-tick data across every part on a track:
            editor.add_expression_to_track(0, "pitd", abs_ticks, pitd_values)

    Manual usage::

        editor = UstxEditor("song.ustx")
        ...
        editor.save()
        editor.close()
    """

    def __init__(self, ustx_path: str) -> None:
        self.ustx_path = ustx_path
        self._lock = FileLock(ustx_path + ".lock", thread_local=False, is_singleton=True)
        self._lock.acquire()
        self.project: UProject = load_ustx(ustx_path)

    # ------------------------------------------------------------------
    # Project-level properties
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> list[UTrack]:
        """All tracks in the project (0-based)."""
        return self.project.tracks

    @property
    def voice_parts(self) -> list[UVoicePart]:
        """All voice parts in the project, in file order."""
        return self.project.voice_parts

    @property
    def tempos(self) -> list[UTempo]:
        """Tempo map of the project."""
        return self.project.tempos

    @property
    def time_signatures(self) -> list[UTimeSignature]:
        """Time-signature map of the project."""
        return self.project.time_signatures

    def get_track(self, track_no: int) -> UTrack:
        """Return the track at 0-based index *track_no*."""
        return self.project.get_track(track_no)

    def get_parts_for_track(self, track_no: int) -> list[UVoicePart]:
        """Return all voice parts for *track_no* (0-based), sorted by position."""
        return self.project.get_parts_for_track(track_no)

    def build_time_axis(self) -> TimeAxis:
        """Build and return a :class:`TimeAxis` for the project's tempo map."""
        return self.project.build_time_axis()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def __enter__(self) -> "UstxEditor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None:
                self.save()
        finally:
            self.close()
        return False

    def save(self) -> None:
        """Write the project back to disk."""
        save_ustx(self.project, self.ustx_path)

    def close(self) -> None:
        """Release the file lock without saving."""
        self._lock.release()

    def add_expression_to_part(
        self,
        part: UVoicePart,
        expression_name: str,
        expression_ticks: np.ndarray,
        expression_values: np.ndarray,
    ) -> None:
        """Overwrite an expression curve on a specific voice part.

        Args:
            part:               The :class:`UVoicePart` to edit.
            expression_name:    Curve abbreviation, e.g. ``"dyn"``.
            expression_ticks:   1-D integer array of tick positions (relative to part start).
            expression_values:  1-D float array of values (NaN entries are skipped).
        """
        part.set_curve(expression_name, expression_ticks, expression_values)

    def add_expression_to_track(
        self,
        track_no: int,
        expression_name: str,
        expression_ticks: np.ndarray,
        expression_values: np.ndarray,
    ) -> None:
        """Overwrite an expression curve across all voice parts on a track.

        Tick positions in *expression_ticks* are absolute project ticks.
        Each part receives the slice of the curve that falls within its own
        ``[position, position + duration)`` window, re-expressed as ticks
        relative to the part start (matching how OpenUtau stores curves).

        Args:
            track_no:           0-based track index (matches ``UVoicePart.track_no``).
            expression_name:    Curve abbreviation, e.g. ``"dyn"``.
            expression_ticks:   1-D integer array of *absolute* project tick positions.
            expression_values:  1-D float array of values (NaN entries are skipped).
        """
        parts = self.project.get_parts_for_track(track_no)
        if not parts:
            raise ValueError(f"No voice parts found for track_no {track_no}.")

        curve_set = False
        ticks = np.asarray(expression_ticks, dtype=int)
        values = np.asarray(expression_values, dtype=float)

        for part in parts:
            part_start = part.position
            part_end   = part.position + part.duration
            mask = (ticks >= part_start) & (ticks < part_end)
            if not mask.any():
                continue
            relative_ticks = ticks[mask] - part_start
            part.set_curve(expression_name, relative_ticks, values[mask])
            curve_set = True

        if not curve_set:
            log.warning(
                "No expression points fit inside any part on track %d (0-based) for curve %s.",
                track_no,
                expression_name,
            )
