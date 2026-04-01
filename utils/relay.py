"""
relay — lightweight data transport between processes.

    import relay
    relay.set_relay_dir("/tmp/my_relay")   # call once before any Collector/Deliverer

Collector (producer side):
    collector = Collector(source="dyn@0")
    collector.register(tag="rms_overview", data=fig)
    # flush on __del__, explicit .flush(), or used as context manager

Deliverer (consumer side):
    deliverer = Deliverer()

    # all sources, all tags
    async for source, tag, data in deliverer.deliver():
        data.show()

    # glob sources, filtered tags
    async for source, tag, data in deliverer.deliver("dyn@*", tags="rms_*"):
        data.show()

    # ignore everything already on disk — only yield future arrivals
    async for source, tag, data in Deliverer(skip_existing=True).deliver("dyn@*"):
        data.show()

--- Directory layout (after Collector.flush) ---

    <output_dir>/
      <source_hash>/               # sha256[:24] of source string
        .sidecar.json              # {"source": "dyn@0", "tag": null}
        .done                      # completion sentinel
        <tag_hash>.pkl             # pickled payload
        <tag_hash>.sidecar.json   # {"source": "dyn@0", "tag": "rms_overview"}

--- Demo ---

Producer process:

    import relay
    import plotly.graph_objects as go
    from relay import Collector

    relay.set_relay_dir("/tmp/my_relay")

    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9], name="y=x²"))
    fig.update_layout(title="Demo")

    collector = Collector(source="demo@0")
    collector.register(tag="scatter", data=fig)
    # flush happens automatically when collector is destroyed

Consumer process:

    import asyncio
    import relay
    from relay import Deliverer

    relay.set_relay_dir("/tmp/my_relay")

    async def main():
        async for source, tag, data in Deliverer().deliver("dyn@*"):
            print(f"{source} / {tag}")
            data.show()

    asyncio.run(main())
"""

from __future__ import annotations

import re
import json
import pickle
import shutil
import atexit
import fnmatch
import hashlib
import logging
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, AsyncIterator, ClassVar

from watchfiles import awatch

log = logging.getLogger(__name__)

# Shared output directory — used by all Collector and Deliverer instances.
# Override at startup with set_relay_dir() before constructing any instances.
_OUTPUT_DIR         = Path(tempfile.gettempdir()) / "_relay_watch" / "relay"
_DEFAULT_OUTPUT_DIR = _OUTPUT_DIR  # sentinel — compared by identity in _warn_if_default_dir
_default_dir_warned = False        # set to True on first warning, or by set_relay_dir()


def set_relay_dir(path: Path | str) -> None:
    """
    Set the module-level output directory used by all Collector and Deliverer
    instances.

    Call once at process startup, before constructing any Collector or
    Deliverer.  Changing the directory after instances have been created has
    no effect on already-constructed objects.

    The path is stored as given.  A stable parent directory (at least one
    level up) must exist or be creatable, because Deliverer watches the
    parent to survive rm/recreate of the relay dir itself.

        import relay
        relay.set_relay_dir("/data/my_relay")
    """
    global _OUTPUT_DIR, _default_dir_warned
    _OUTPUT_DIR         = Path(path)
    _default_dir_warned = True  # explicit call — suppress the default-dir warning
    log.debug("relay output dir set to %s", _OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Process-level Deliverer reference count + atexit cleanup.
# ---------------------------------------------------------------------------
_deliverer_count = 0


def _warn_if_default_dir() -> None:
    """
    Emit a warning the first time a Collector or Deliverer is constructed
    without a prior set_relay_dir() call.  Fires at most once per process.
    """
    global _default_dir_warned
    if _OUTPUT_DIR is _DEFAULT_OUTPUT_DIR and not _default_dir_warned:
        _default_dir_warned = True
        warnings.warn(
            f"relay: output directory has not been set explicitly; "
            f"using default tempdir {_OUTPUT_DIR!r}.  "
            f"Call relay.set_relay_dir(...) at process startup to silence this.",
            stacklevel=3,
        )


def _make_cleanup(output_dir: Path):
    """
    Return an atexit callback closed over output_dir and the shared refcount.

    Registered at Deliverer construction time; unregistered by Deliverer.close().
    If the process exits with live Deliverers still open, this fires, decrements
    the refcount, and removes the relay dir when the count reaches zero.

    It is *not* called by Deliverer.close() — close() calls _decrement() instead
    so that the filesystem is never touched during normal mid-program teardown.
    This means any number of Deliverers can be created and closed sequentially
    within the same process without wiping the relay dir between them.
    """
    def _atexit() -> None:
        global _deliverer_count
        _deliverer_count -= 1
        if _deliverer_count <= 0:
            try:
                shutil.rmtree(output_dir)
                log.info("Cleaned up %s", output_dir)
            except FileNotFoundError:
                pass  # already gone — not an error
            except Exception:
                log.exception("Error during relay cleanup")
    return _atexit


def _decrement() -> None:
    """
    Decrement the global Deliverer refcount without touching the filesystem.
    Called by Deliverer.close() for mid-program teardown.
    """
    global _deliverer_count
    _deliverer_count -= 1
    log.debug("Deliverer refcount now %d", _deliverer_count)


# How often (in watch cycles) to prune _seen of dead sources.
_PRUNE_EVERY_N_CYCLES = 20


def _hash(name: str) -> str:
    """
    Return a 24-char lowercase hex digest of name.

    sha256[:24] = 96 bits of entropy — collision probability is negligible
    for any realistic tag/source namespace.
    """
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def _read_sidecar(path: Path) -> tuple[str, str | None] | None:
    """
    Read a sidecar JSON file and return (source, tag).

    The source-level sidecar has tag=null; per-tag sidecars have both set.
    Returns None if the file is missing or malformed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["source"], data["tag"]
    except Exception:
        return None


def history_cleanup() -> None:
    """
    Remove all contents under _OUTPUT_DIR immediately, while preserving the
    directory itself.

    Safe to call at any time — if the directory does not exist the call is a
    no-op.  Any in-progress deliver() generators will stop seeing new entries
    until a Collector flushes again and recreates the directory.
    """
    try:
        for entry in _OUTPUT_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        log.info("Cleaned up %s", _OUTPUT_DIR)
    except FileNotFoundError:
        pass  # already gone — not an error
    except Exception:
        log.exception("Error during relay.history_cleanup")


def _as_patterns(val: str | list[str] | None) -> list[str]:
    """Normalize sources/tags argument to a list of glob patterns."""
    if val is None:
        return ["*"]
    return [val] if isinstance(val, str) else list(val)


class Collector:
    def __init__(self, source: str) -> None:
        _warn_if_default_dir()
        self.source   = source
        self._entries: dict[str, Any] = {}
        self._flushed = False

    def register(self, tag: str, data: Any) -> None:
        """Register any picklable object under the given tag."""
        if self._flushed:
            raise RuntimeError(
                f"Collector for source={self.source!r} has already been flushed."
            )
        self._entries[tag] = data
        log.debug("Registered tag=%r for source=%r", tag, self.source)

    def flush(self) -> None:
        """
        Write all entries atomically and mark the source as done. Idempotent.

        Uses a staging directory that is renamed into place so that a consumer
        can never observe a partially-written source, even if the producer
        crashes mid-flush.

        Directory layout after flush:

            <output_dir>/
              <source_hash>/               # sha256[:24] of source string
                .sidecar.json              # {"source": "...", "tag": null}
                .done                      # completion sentinel
                <tag_hash>.pkl             # pickled payload
                <tag_hash>.sidecar.json   # {"source": "...", "tag": "..."}
        """
        if self._flushed:
            return

        source_hash = _hash(self.source)
        staging_dir = _OUTPUT_DIR / f".staging_{source_hash}"
        final_dir   = _OUTPUT_DIR / source_hash

        # Clean up any previous staging remnant or final directory.
        for d in (staging_dir, final_dir):
            if d.exists():
                shutil.rmtree(d)

        staging_dir.mkdir(parents=True)

        # Source-level sidecar — source name only, tag is null.
        (staging_dir / ".sidecar.json").write_text(
            json.dumps({"source": self.source, "tag": None}),
            encoding="utf-8",
        )

        for tag, data in self._entries.items():
            tag_hash = _hash(tag)

            # Pickled payload.
            pkl_path = staging_dir / f"{tag_hash}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            # Per-tag sidecar — both source and tag for self-contained lookup.
            (staging_dir / f"{tag_hash}.sidecar.json").write_text(
                json.dumps({"source": self.source, "tag": tag}),
                encoding="utf-8",
            )

            log.debug("Wrote tag=%r → %s", tag, tag_hash)

        # Atomic rename: guaranteed on POSIX; best-effort on Windows
        # (os.replace is atomic within the same volume).
        staging_dir.rename(final_dir)
        (final_dir / ".done").touch()

        log.info("Flushed source=%r (%d entries)", self.source, len(self._entries))
        self._flushed = True

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            log.exception("Error during Collector.__del__ flush for source=%r", self.source)

    def __enter__(self)    -> "Collector": return self
    def __exit__(self, *_) -> None: self.flush()


class Deliverer:
    def __init__(self, *, skip_existing: bool = False) -> None:
        """
        skip_existing: if True, entries already present in _OUTPUT_DIR at the
            time deliver() is called are silently skipped.  Only entries that
            arrive after deliver() starts will be yielded.  Scoped to the same
            source/tag patterns passed to deliver(), so unrelated entries are
            never added to _seen.
        """
        global _deliverer_count
        _warn_if_default_dir()
        _deliverer_count += 1

        self._seen:          set[tuple[str, str]] = set()
        self._cycle:         int                  = 0
        self._skip_existing: bool                 = skip_existing
        self._closed:        bool                 = False

        # _atexit_fn fires at process exit (if close() was never called) and
        # deletes the relay dir once the last Deliverer exits.
        # close() calls the module-level _decrement() instead — refcount only,
        # no filesystem work — so sequential Deliverer usage never wipes the dir.
        self._atexit_fn = _make_cleanup(_OUTPUT_DIR)
        atexit.register(self._atexit_fn)

    def close(self) -> None:
        """
        Release this Deliverer's reference.  Idempotent — safe to call multiple
        times.

        Unregisters the atexit handler and decrements the refcount.  The relay
        dir is never deleted here; deletion is deferred to process exit via
        atexit for any Deliverer that remains open until shutdown.
        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_fn)  # won't fire at exit — already closed
        _decrement()                         # refcount only; no rmtree

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            log.exception("Error during Deliverer.__del__")

    def __enter__(self) -> "Deliverer": return self
    def __exit__(self, *_) -> None: self.close()

    def _read_source(self, source_dir: Path) -> str | None:
        """Read the source name from the directory-level sidecar."""
        result = _read_sidecar(source_dir / ".sidecar.json")
        return result[0] if result else None

    def _snapshot_existing(
        self,
        source_patterns: list[str],
        tag_patterns:    list[str],
    ) -> None:
        """
        Pre-populate _seen with every (source, tag) pair that currently exists
        on disk and matches the given patterns, so _scan skips them on the
        first pass.  Does not deserialize any payload data.

        Called before _OUTPUT_DIR.mkdir() in deliver(), so the directory may
        not exist yet — in that case there is nothing to snapshot.
        """
        if not _OUTPUT_DIR.exists():
            return
        before = len(self._seen)
        for source_dir in _OUTPUT_DIR.iterdir():
            if source_dir.name.startswith(".staging_"):
                continue
            if not (source_dir / ".done").exists():
                continue
            source = self._read_source(source_dir)
            if source is None:
                continue
            if not any(fnmatch.fnmatch(source, p) for p in source_patterns):
                continue
            for sidecar_path in source_dir.glob("*.sidecar.json"):
                result = _read_sidecar(sidecar_path)
                if result is None:
                    continue
                _, tag = result
                if tag is None:
                    continue
                if not any(fnmatch.fnmatch(tag, p) for p in tag_patterns):
                    continue
                self._seen.add((source, tag))
        log.debug(
            "skip_existing: pre-seeded _seen with %d matching existing entries",
            len(self._seen) - before,
        )

    def _prune_seen(self) -> None:
        """
        Remove entries from _seen whose source directory no longer exists.
        Prevents unbounded growth for long-running deliverers.
        """
        if not _OUTPUT_DIR.exists():
            self._seen.clear()
            return
        live_sources: set[str] = set()
        for p in _OUTPUT_DIR.iterdir():
            if p.is_dir() and not p.name.startswith(".staging_"):
                source = self._read_source(p)
                if source is not None:
                    live_sources.add(source)
        before = len(self._seen)
        self._seen = {(src, tag) for src, tag in self._seen if src in live_sources}
        pruned = before - len(self._seen)
        if pruned:
            log.debug("Pruned %d stale entries from _seen", pruned)

    def _scan(
        self,
        source_patterns: list[str],
        tag_patterns:    list[str],
    ) -> list[tuple[str, str, Any]]:
        """
        Scan the output directory once and return a list of (source, tag, data)
        tuples for all new matching entries. Deserialization errors are logged
        and skipped — they do not propagate to the caller.
        """
        results: list[tuple[str, str, Any]] = []

        if not _OUTPUT_DIR.exists():
            return results

        for source_dir in sorted(_OUTPUT_DIR.iterdir()):
            # Skip staging directories — they are not yet complete.
            if source_dir.name.startswith(".staging_"):
                continue
            if not (source_dir / ".done").exists():
                continue

            source = self._read_source(source_dir)
            if source is None:
                log.warning("Missing or malformed source sidecar in %s — skipping", source_dir)
                continue
            if not any(fnmatch.fnmatch(source, p) for p in source_patterns):
                continue

            for pkl_path in sorted(source_dir.glob("*.pkl")):
                sidecar_path = pkl_path.with_suffix(".sidecar.json")
                result = _read_sidecar(sidecar_path)
                if result is None:
                    log.warning("Missing or malformed sidecar for %s — skipping", pkl_path)
                    continue
                _, tag = result
                if tag is None:
                    log.warning("Null tag in sidecar for %s — skipping", pkl_path)
                    continue
                if not any(fnmatch.fnmatch(tag, p) for p in tag_patterns):
                    continue
                if (source, tag) in self._seen:
                    continue

                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    log.warning(
                        "Failed to deserialize tag=%r from source=%r — skipping",
                        tag, source,
                    )
                    # Mark as seen so we don't retry a permanently broken file.
                    self._seen.add((source, tag))
                    continue

                self._seen.add((source, tag))
                log.debug("Yielding source=%r tag=%r", source, tag)
                results.append((source, tag, data))

        return results

    async def deliver(
        self,
        sources: str | list[str] | None = None,
        tags:    str | list[str] | None = None,
    ) -> AsyncIterator[tuple[str, str, Any]]:
        """
        Async generator — yields (source, tag, data) whenever a new matching
        entry appears via watchfiles (near-zero latency). Caller controls termination.

        sources: glob pattern(s) to match source names. Default: "*" (all).
        tags:    glob pattern(s) to match tag names.   Default: "*" (all).

        Patterns are matched against the original source/tag strings as
        registered, e.g. Collector(source="DynLoader@0") requires
        deliver("DynLoader@*").

        The watcher targets the *parent* of _OUTPUT_DIR so that deletion and
        recreation of the relay dir itself does not kill the watch.
        """
        source_patterns = _as_patterns(sources)
        tag_patterns    = _as_patterns(tags)

        if self._skip_existing:
            # Snapshot before mkdir so the window between "directory created"
            # and "snapshot taken" cannot swallow entries from a racing producer.
            self._snapshot_existing(source_patterns, tag_patterns)

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Initial pass — yield anything already on disk.
        for item in self._scan(source_patterns, tag_patterns):
            yield item

        # Watch the *parent* directory so the watch survives rm/recreate of
        # _OUTPUT_DIR itself.  The parent is created here if it doesn't exist
        # (e.g. first run after set_relay_dir pointed at a fresh path).
        watch_dir = _OUTPUT_DIR.parent
        watch_dir.mkdir(parents=True, exist_ok=True)

        async for changes in awatch(watch_dir):
            # Only react to changes that touch our relay dir subtree.
            relay_str = str(_OUTPUT_DIR)
            relevant = any(
                str(changed_path).startswith(relay_str)
                for _, changed_path in changes
            )
            if not relevant:
                continue

            # If the relay dir was just deleted, clear stale _seen entries and
            # wait — the next Collector.flush() will recreate it and fire again.
            if not _OUTPUT_DIR.exists():
                self._seen.clear()
                log.debug("relay dir removed — cleared _seen, waiting for recreation")
                continue

            self._cycle += 1
            if self._cycle % _PRUNE_EVERY_N_CYCLES == 0:
                self._prune_seen()

            for item in self._scan(source_patterns, tag_patterns):
                yield item


@dataclass
class ExpressionLoaderCollectorNaming:
    PATTERN:         ClassVar[str]         = r'{expression}#{id}@{timestamp}'
    GLOB:            ClassVar[str]         = r'*#*@*'
    REGEX:           ClassVar[re.Pattern]  = re.compile(
        r'^(?P<expression>\w+)#(?P<id>\d+)@'
        r'(?P<timestamp>\d{4}\d{2}\d{2}T'
        r'\d{2}\d{2}\d{2}\.\d{6})$'
    )
    _TIMESTAMP_FMT:  ClassVar[str]         = '%Y%m%dT%H%M%S.%f'

    @classmethod
    def make(cls, expression: str, id: int) -> str:
        ts = datetime.now().strftime(cls._TIMESTAMP_FMT)
        return cls.PATTERN.format(expression=expression, id=id, timestamp=ts)

    @classmethod
    def parse(cls, s: str) -> tuple[str, int, datetime]:
        m = cls.REGEX.fullmatch(s)
        if m is None:
            raise ValueError(f'Invalid collector name: {s!r}')
        d = m.groupdict()
        return (
            d['expression'],
            int(d['id']),
            datetime.strptime(d['timestamp'], cls._TIMESTAMP_FMT),
        )


if __name__ == "__main__":  # pragma: no cover
    import asyncio
    import logging

    import plotly.graph_objects as go

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("example")


    # ---------------------------------------------------------------------------
    # Producer — simulates two loaders flushing at different times
    # ---------------------------------------------------------------------------

    async def producer():
        await asyncio.sleep(0.5)

        log.info("Flushing dyn@0")
        with Collector(source="dyn@0") as col:
            col.register(tag="rms_overview", data=go.Figure(
                go.Scatter(x=[1, 2, 3], y=[1, 4, 9], name="UTAU RMS"),
            ).update_layout(title="dyn@0 — RMS Overview"))
            col.register(tag="aligned", data=go.Figure(
                go.Scatter(x=[1, 2, 3], y=[0.9, 3.8, 8.7], name="Aligned Ref RMS"),
            ).update_layout(title="dyn@0 — Aligned"))

        await asyncio.sleep(1.0)

        log.info("Flushing dyn@1")
        with Collector(source="dyn@1") as col:
            col.register(tag="rms_overview", data=go.Figure(
                go.Scatter(x=[1, 2, 3], y=[2, 5, 10], name="UTAU RMS"),
            ).update_layout(title="dyn@1 — RMS Overview"))


    # ---------------------------------------------------------------------------
    # Consumers — two independent deliverers running concurrently
    # ---------------------------------------------------------------------------

    async def consumer(name: str, sources: str, tags: str) -> None:
        deliverer = Deliverer()
        async for source, tag, data in deliverer.deliver(sources, tags=tags):
            log.info(
                "[%s] Received source=%r tag=%r title=%r",
                name, source, tag, data.layout.title.text,
            )
            data.show()


    # ---------------------------------------------------------------------------
    # Main
    # ---------------------------------------------------------------------------

    async def main():
        set_relay_dir(Path(tempfile.gettempdir()) / "_relay_watch" / "relay")
        await asyncio.gather(
            producer(),
            consumer("overview", "dyn@*", tags="rms_overview"),
            consumer("all",      "dyn@*", tags="*"),
        )

    asyncio.run(main())
