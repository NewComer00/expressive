"""
Tests for utils/relay.py

Run with:
    pytest test_relay.py -v
"""

import asyncio
import json
import pickle
import unittest.mock
import warnings

import pytest

from utils.relay import (
    Collector,
    Deliverer,
    _as_patterns,
    _hash,
    _PRUNE_EVERY_N_CYCLES,
    _read_sidecar,
    _make_cleanup,
    _decrement,
)


# ---------------------------------------------------------------------------
# _as_patterns helper
# ---------------------------------------------------------------------------

class TestAsPatterns:
    def test_none_returns_wildcard(self):
        assert _as_patterns(None) == ["*"]

    def test_str_returns_list(self):
        assert _as_patterns("dyn@*") == ["dyn@*"]

    def test_list_passes_through(self):
        assert _as_patterns(["dyn@0", "dyn@1"]) == ["dyn@0", "dyn@1"]


# ---------------------------------------------------------------------------
# _read_sidecar
# ---------------------------------------------------------------------------

class TestReadSidecar:
    def test_returns_source_and_tag(self, relay_dir):
        relay_dir.mkdir(parents=True)
        sidecar = relay_dir / ".sidecar.json"
        sidecar.write_text(json.dumps({"source": "test@0", "tag": "plot1"}), encoding="utf-8")
        result = _read_sidecar(sidecar)
        assert result == ("test@0", "plot1")

    def test_returns_none_for_missing_file(self, relay_dir):
        result = _read_sidecar(relay_dir / "nonexistent.json")
        assert result is None

    def test_returns_none_for_malformed_json(self, relay_dir):
        relay_dir.mkdir(parents=True)
        sidecar = relay_dir / ".sidecar.json"
        sidecar.write_text("not valid json{", encoding="utf-8")
        result = _read_sidecar(sidecar)
        assert result is None

    def test_returns_none_when_missing_source_key(self, relay_dir):
        relay_dir.mkdir(parents=True)
        sidecar = relay_dir / ".sidecar.json"
        sidecar.write_text(json.dumps({"tag": "plot1"}), encoding="utf-8")
        result = _read_sidecar(sidecar)
        assert result is None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TestCollector:
    def test_register_and_flush(self, relay_dir):
        col = Collector(source="test@0")
        col.register(tag="plot1", data={"x": [1, 2, 3]})
        col.flush()

        source_dir = relay_dir / _hash("test@0")
        assert (source_dir / ".done").exists()
        assert (source_dir / f"{_hash('plot1')}.pkl").exists()

    def test_flush_idempotent(self, relay_dir):
        col = Collector(source="test@0")
        col.register(tag="plot1", data={"x": 1})
        col.flush()
        col.flush()  # should not raise

    def test_register_after_flush_raises(self, relay_dir):
        col = Collector(source="test@0")
        col.flush()
        with pytest.raises(RuntimeError, match="already been flushed"):
            col.register(tag="plot1", data={})

    def test_context_manager_flushes(self, relay_dir):
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"x": 1})
        assert (relay_dir / _hash("test@0") / ".done").exists()

    def test_stale_data_cleared_on_flush(self, relay_dir):
        with Collector(source="test@0") as col:
            col.register(tag="old_plot", data={"x": 1})

        with Collector(source="test@0") as col:
            col.register(tag="new_plot", data={"x": 2})

        source_dir = relay_dir / _hash("test@0")
        pkl_files = list(source_dir.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_data_pickled_correctly(self, relay_dir):
        payload = {"x": [1, 2, 3], "y": [4, 5, 6]}
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data=payload)

        path = relay_dir / _hash("test@0") / f"{_hash('plot1')}.pkl"
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == payload

    def test_no_done_sentinel_during_staging(self, relay_dir):
        """A source directory must not be visible to consumers until rename completes."""
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"x": 1})

        staging = relay_dir / f".staging_{_hash('test@0')}"
        final   = relay_dir / _hash("test@0")
        assert not staging.exists()
        assert final.exists()
        assert (final / ".done").exists()


# ---------------------------------------------------------------------------
# _warn_if_default_dir
# ---------------------------------------------------------------------------

class TestWarnIfDefaultDir:
    def test_warns_once_on_first_collector(self, monkeypatch, tmp_path):
        """_warn_if_default_dir must emit a warning when using the default dir."""
        import utils.relay as relay_module

        monkeypatch.setattr(relay_module, "_OUTPUT_DIR", relay_module._DEFAULT_OUTPUT_DIR)
        monkeypatch.setattr(relay_module, "_default_dir_warned", False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Collector(source="test@0")
            assert len(w) == 1
            assert "output directory has not been set explicitly" in str(w[0].message)

    def test_warns_once_on_first_deliverer(self, monkeypatch, tmp_path):
        """_warn_if_default_dir must emit a warning when using the default dir for Deliverer."""
        import utils.relay as relay_module

        monkeypatch.setattr(relay_module, "_OUTPUT_DIR", relay_module._DEFAULT_OUTPUT_DIR)
        monkeypatch.setattr(relay_module, "_default_dir_warned", False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Deliverer()
            assert len(w) == 1
            assert "output directory has not been set explicitly" in str(w[0].message)

    def test_no_warning_after_set_relay_dir(self, monkeypatch, tmp_path):
        """set_relay_dir must suppress the default-dir warning."""
        import utils.relay as relay_module

        monkeypatch.setattr(relay_module, "_default_dir_warned", False)
        relay_module.set_relay_dir(tmp_path / "custom")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Collector(source="test@0")
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Deliverer
# ---------------------------------------------------------------------------

class TestDeliverer:
    def test_deliver_all(self, relay_dir):
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})
            col.register(tag="plot2", data={"v": 2})

        results = asyncio.run(collect_delivered(Deliverer()))
        assert {(s, t) for s, t, _ in results} == {("test@0", "plot1"), ("test@0", "plot2")}

    def test_deliver_source_glob(self, relay_dir):
        with Collector(source="dyn@0") as col:
            col.register(tag="plot1", data={"v": 1})
        with Collector(source="other@0") as col:
            col.register(tag="plot1", data={"v": 2})

        results = asyncio.run(collect_delivered(Deliverer(), sources="dyn@*"))
        assert {s for s, _, _ in results} == {"dyn@0"}

    def test_deliver_tag_glob(self, relay_dir):
        with Collector(source="test@0") as col:
            col.register(tag="rms_overview", data={"v": 1})
            col.register(tag="aligned",      data={"v": 2})
            col.register(tag="zscore",       data={"v": 3})

        results = asyncio.run(collect_delivered(
            Deliverer(), sources="test@*", tags="rms_*"
        ))
        assert {t for _, t, _ in results} == {"rms_overview"}

    def test_no_duplicate_delivery(self, relay_dir):
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})

        deliverer = Deliverer()
        r1 = asyncio.run(collect_delivered(deliverer))
        r2 = asyncio.run(collect_delivered(deliverer))  # same deliverer, _seen persists
        assert len(r1) == 1
        assert len(r2) == 0

    def test_skip_existing_suppresses_preexisting_entries(self, relay_dir):
        """skip_existing=True must not yield entries already on disk when deliver() is called."""
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})
            col.register(tag="plot2", data={"v": 2})

        results = asyncio.run(collect_delivered(Deliverer(skip_existing=True)))
        assert results == []

    def test_skip_existing_still_yields_new_arrivals(self, relay_dir):
        """skip_existing=True must still yield entries that arrive after deliver() starts."""
        with Collector(source="old@0") as col:
            col.register(tag="plot1", data={"v": 1})

        async def run():
            results = []

            async def flush_after_delay():
                await asyncio.sleep(0.1)
                with Collector(source="new@0") as col:
                    col.register(tag="plot1", data={"v": 2})

            async def consume():
                async for source, tag, data in Deliverer(skip_existing=True).deliver():
                    results.append((source, tag, data))
                    return  # stop after first new item

            await asyncio.gather(
                flush_after_delay(),
                asyncio.wait_for(consume(), timeout=5.0),
            )
            return results

        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0][:2] == ("new@0", "plot1")

    def test_skip_existing_scoped_to_patterns(self, relay_dir):
        """skip_existing must only suppress entries matching the deliver() patterns."""
        with Collector(source="dyn@0") as col:
            col.register(tag="rms", data={"v": 1})
        with Collector(source="other@0") as col:
            col.register(tag="rms", data={"v": 2})

        # Each Deliverer is explicitly closed after use so the relay dir is
        # never deleted mid-test (close() only decrements; rmtree is atexit-only).
        d1 = Deliverer(skip_existing=True)
        r1 = asyncio.run(collect_delivered(d1, sources="dyn@*"))
        d1.close()
        assert r1 == []

        d2 = Deliverer(skip_existing=True)
        r2 = asyncio.run(collect_delivered(d2, sources="other@*"))
        d2.close()
        assert r2 == []

        d3 = Deliverer()
        r3 = asyncio.run(collect_delivered(d3))
        d3.close()
        assert {(s, t) for s, t, _ in r3} == {("dyn@0", "rms"), ("other@0", "rms")}

    def test_skip_existing_no_op_when_dir_absent(self, relay_dir):
        """_snapshot_existing must return early when _OUTPUT_DIR does not exist."""
        d = Deliverer(skip_existing=True)
        results = asyncio.run(collect_delivered(d))
        assert results == []
        assert d._seen == set()

    def test_skip_existing_ignores_staging_dirs(self, relay_dir):
        """_snapshot_existing must skip .staging_ directories."""
        staging = relay_dir / f".staging_{_hash('test@0')}"
        staging.mkdir(parents=True)
        (staging / ".done").touch()
        (staging / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))

        d = Deliverer(skip_existing=True)
        asyncio.run(collect_delivered(d))
        assert d._seen == set()

    def test_skip_existing_ignores_dirs_without_done(self, relay_dir):
        """_snapshot_existing must skip source dirs that have no .done sentinel."""
        incomplete = relay_dir / _hash("test@0")
        incomplete.mkdir(parents=True)
        (incomplete / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))

        d = Deliverer(skip_existing=True)
        asyncio.run(collect_delivered(d))
        assert d._seen == set()

    def test_skip_existing_skips_missing_source_sidecar(self, relay_dir, monkeypatch):
        """_snapshot_existing must skip dirs with missing/malformed source sidecar."""
        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))
        # No .sidecar.json — _read_sidecar returns None

        d = Deliverer(skip_existing=True)
        d._snapshot_existing(["*"], ["*"])
        assert d._seen == set()

    def test_skip_existing_skips_malformed_tag_sidecar(self, relay_dir):
        """_snapshot_existing must skip entries with missing/malformed tag sidecar."""
        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / ".sidecar.json").write_text(
            json.dumps({"source": "test@0", "tag": None}), encoding="utf-8"
        )
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))
        # No tag sidecar — _read_sidecar returns None

        d = Deliverer(skip_existing=True)
        d._snapshot_existing(["*"], ["*"])
        assert d._seen == set()

    def test_skip_existing_respects_tag_pattern(self, relay_dir):
        """_snapshot_existing must not add tags that do not match tag_patterns."""
        with Collector(source="test@0") as col:
            col.register(tag="rms_overview", data={"v": 1})
            col.register(tag="aligned",      data={"v": 2})

        d = Deliverer(skip_existing=True)
        asyncio.run(collect_delivered(d, tags="rms_*"))
        assert ("test@0", "rms_overview") in d._seen
        assert ("test@0", "aligned") not in d._seen

    def test_data_delivered_correctly(self, relay_dir):
        payload = {"x": [1, 2], "y": [3, 4]}
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data=payload)

        results = asyncio.run(collect_delivered(Deliverer()))
        assert results[0][2] == payload

    def test_corrupt_pkl_skipped(self, relay_dir):
        """A corrupt .pkl file should be skipped, not crash the consumer."""
        with Collector(source="test@0") as col:
            col.register(tag="good", data={"v": 1})

        source_dir = relay_dir / _hash("test@0")
        bad_path = source_dir / f"{_hash('good')}.pkl"
        bad_path.write_bytes(b"not valid pickle data")

        results = asyncio.run(collect_delivered(Deliverer()))
        assert results == []

    def test_scan_skips_missing_source_sidecar(self, relay_dir, monkeypatch):
        """_scan must skip dirs with missing source sidecar and log warning."""
        import utils.relay as relay_module

        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))
        # No .sidecar.json

        d = Deliverer()
        with unittest.mock.patch.object(relay_module.log, "warning") as mock_warn:
            results = d._scan(["*"], ["*"])
            assert results == []
            assert mock_warn.called

    def test_scan_skips_missing_tag_sidecar(self, relay_dir):
        """_scan must skip entries with missing tag sidecar and log warning."""
        import utils.relay as relay_module

        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / ".sidecar.json").write_text(
            json.dumps({"source": "test@0", "tag": None}), encoding="utf-8"
        )
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))
        # No tag sidecar

        d = Deliverer()
        with unittest.mock.patch.object(relay_module.log, "warning") as mock_warn:
            results = d._scan(["*"], ["*"])
            assert results == []
            assert mock_warn.called

    def test_scan_skips_null_tag_in_sidecar(self, relay_dir):
        """_scan must skip entries with null tag and log warning."""
        import utils.relay as relay_module

        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / ".sidecar.json").write_text(
            json.dumps({"source": "test@0", "tag": None}), encoding="utf-8"
        )
        (source_dir / f"{_hash('plot1')}.sidecar.json").write_text(
            json.dumps({"source": "test@0", "tag": None}), encoding="utf-8"
        )
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))

        d = Deliverer()
        with unittest.mock.patch.object(relay_module.log, "warning") as mock_warn:
            results = d._scan(["*"], ["*"])
            assert results == []
            assert mock_warn.called

    def test_multiple_deliverers_independent_seen(self, relay_dir):
        """Two deliverers must each receive all matching entries independently."""
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})

        d1 = Deliverer()
        d2 = Deliverer()
        r1 = asyncio.run(collect_delivered(d1))
        r2 = asyncio.run(collect_delivered(d2))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_close_is_idempotent(self, relay_dir):
        """Calling close() multiple times must not raise or double-decrement."""
        import utils.relay as relay_module

        d = Deliverer()
        count_before = relay_module._deliverer_count
        d.close()
        d.close()
        d.close()
        assert relay_module._deliverer_count == count_before - 1

    def test_close_does_not_delete_relay_dir(self, relay_dir):
        """close() must never delete the relay dir — that is atexit's job."""
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})

        d = Deliverer()
        asyncio.run(collect_delivered(d))
        d.close()
        assert relay_dir.exists(), "relay dir must survive an explicit close()"

    def test_sequential_deliverers_do_not_wipe_relay_dir(self, relay_dir):
        """Creating and closing multiple Deliverers in sequence must not delete the dir."""
        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})

        for _ in range(3):
            d = Deliverer()
            asyncio.run(collect_delivered(d))
            d.close()

        assert relay_dir.exists()
        # A fresh Deliverer must still find the data.
        results = asyncio.run(collect_delivered(Deliverer()))
        assert len(results) == 1

    def test_cleanup_filenotfounderror_is_silent(self, relay_dir):
        """atexit cleanup of a non-existent dir must not log an error."""
        import utils.relay as relay_module

        atexit_fn = _make_cleanup(relay_dir)
        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            atexit_fn()  # relay_dir was never created — must not raise
            mock_exc.assert_not_called()

    def test_cleanup_unexpected_error_is_logged(self, relay_dir, monkeypatch):
        """An unexpected rmtree error in the atexit callback must be logged."""
        import utils.relay as relay_module

        relay_dir.mkdir(parents=True, exist_ok=True)

        def boom(path):
            raise PermissionError("locked")

        monkeypatch.setattr("shutil.rmtree", boom)
        atexit_fn = _make_cleanup(relay_dir)
        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            atexit_fn()
            mock_exc.assert_called_once()

    def test_decrement_reduces_count_without_rmtree(self, relay_dir, monkeypatch):
        """_decrement() must lower the refcount but never call shutil.rmtree."""
        import utils.relay as relay_module
        import shutil

        relay_dir.mkdir(parents=True, exist_ok=True)
        relay_module._deliverer_count = 1

        with unittest.mock.patch.object(shutil, "rmtree") as mock_rm:
            _decrement()
            mock_rm.assert_not_called()

        assert relay_module._deliverer_count == 0

    def test_collector_del_exception_logged(self, relay_dir, monkeypatch):
        """Collector.__del__ must log, not propagate, a flush exception."""
        import utils.relay as relay_module

        col = Collector(source="test@0")
        col.register(tag="plot1", data={"v": 1})

        def bad_flush():
            raise RuntimeError("disk full")
        col.flush = bad_flush

        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            col.__del__()
            mock_exc.assert_called_once()

    def test_prune_seen_clears_when_dir_missing(self, relay_dir):
        """_prune_seen should clear _seen entirely when _OUTPUT_DIR is gone."""
        deliverer = Deliverer()
        deliverer._seen = {("test@0", "plot1"), ("test@0", "plot2")}
        deliverer._prune_seen()
        assert deliverer._seen == set()

    def test_prune_seen_removes_dead_sources(self, relay_dir):
        """_prune_seen should drop entries whose source dir no longer exists."""
        live_dir = relay_dir / _hash("live@0")
        live_dir.mkdir(parents=True)
        (live_dir / ".sidecar.json").write_text(
            json.dumps({"source": "live@0", "tag": None}), encoding="utf-8"
        )

        deliverer = Deliverer()
        deliverer._seen = {("live@0", "plot1"), ("dead@0", "plot1")}
        deliverer._prune_seen()
        assert deliverer._seen == {("live@0", "plot1")}

    def test_scan_returns_empty_when_dir_missing(self, relay_dir):
        """_scan should return [] when _OUTPUT_DIR does not exist."""
        deliverer = Deliverer()
        assert deliverer._scan(["*"], ["*"]) == []

    def test_scan_skips_staging_dirs(self, relay_dir):
        """_scan should ignore .staging_ directories."""
        staging = relay_dir / f".staging_{_hash('test@0')}"
        staging.mkdir(parents=True)
        (staging / ".done").touch()

        results = Deliverer()._scan(["*"], ["*"])
        assert results == []

    def test_scan_skips_dirs_without_done(self, relay_dir):
        """_scan should ignore source dirs that have no .done sentinel."""
        incomplete = relay_dir / _hash("test@0")
        incomplete.mkdir(parents=True)
        (incomplete / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))
        results = Deliverer()._scan(["*"], ["*"])
        assert results == []

    def test_deliver_via_awatch(self, relay_dir):
        """Data flushed after deliver() starts must be picked up via awatch."""
        async def run():
            results = []

            async def flush_after_delay():
                await asyncio.sleep(0.1)
                with Collector(source="late@0") as col:
                    col.register(tag="plot1", data={"v": 42})

            async def consume():
                d = Deliverer()
                d._cycle = _PRUNE_EVERY_N_CYCLES - 1
                async for source, tag, data in d.deliver():
                    results.append((source, tag, data))
                    return

            await asyncio.gather(
                flush_after_delay(),
                asyncio.wait_for(consume(), timeout=5.0),
            )
            return results

        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0][:2] == ("late@0", "plot1")
        assert results[0][2] == {"v": 42}

    def test_atexit_cleanup_on_last_deliverer_exit(self, relay_dir):
        """
        The atexit callback must delete the relay dir only when the last
        outstanding Deliverer's atexit fn fires.  Two sequential calls to the
        atexit fn (simulating two un-closed Deliverers exiting at process end)
        must keep the dir alive until the refcount reaches zero.
        """
        import utils.relay as relay_module

        relay_dir.mkdir(parents=True, exist_ok=True)

        # Simulate two Deliverers that were never explicitly closed — their
        # atexit callbacks fire at shutdown.
        relay_module._deliverer_count = 2
        fn1 = _make_cleanup(relay_dir)
        fn2 = _make_cleanup(relay_dir)

        fn1()
        assert relay_dir.exists(), "dir should survive while a second Deliverer is live"

        fn2()
        assert not relay_dir.exists(), "dir should be gone after last Deliverer exits"


# ---------------------------------------------------------------------------
# Deliverer.__del__ exception path (lines 365-366)
# ---------------------------------------------------------------------------

    def test_del_exception_is_logged(self, relay_dir, monkeypatch):
        """Deliverer.__del__ must log, not propagate, an exception from close()."""
        import utils.relay as relay_module

        d = Deliverer()

        def bad_close():
            raise RuntimeError("boom")
        d.close = bad_close

        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            d.__del__()
            mock_exc.assert_called_once()

    # ---------------------------------------------------------------------------
    # _snapshot_existing — malformed tag sidecar returns None (line 405)
    # ---------------------------------------------------------------------------

    def test_snapshot_existing_skips_unreadable_tag_sidecar(self, relay_dir):
        """
        _snapshot_existing must skip (continue) when _read_sidecar returns None
        for a tag sidecar file — i.e. the file exists but is malformed JSON.
        """
        source_dir = relay_dir / _hash("test@0")
        source_dir.mkdir(parents=True)
        (source_dir / ".done").touch()
        (source_dir / ".sidecar.json").write_text(
            json.dumps({"source": "test@0", "tag": None}), encoding="utf-8"
        )
        # Write a tag sidecar that is valid enough to be found by glob but
        # contains malformed JSON so _read_sidecar returns None.
        (source_dir / f"{_hash('plot1')}.sidecar.json").write_bytes(b"{{bad json")
        (source_dir / f"{_hash('plot1')}.pkl").write_bytes(pickle.dumps({"v": 1}))

        d = Deliverer(skip_existing=True)
        d._snapshot_existing(["*"], ["*"])
        # Malformed sidecar → result is None → continue; nothing added to _seen.
        assert d._seen == set()

    # ---------------------------------------------------------------------------
    # deliver() awatch paths (lines 546, 551-553)
    # ---------------------------------------------------------------------------

    def test_deliver_ignores_unrelated_watch_events(self, relay_dir, monkeypatch):
        """
        deliver() must skip awatch change-sets where no path falls under the
        relay dir subtree (line 546 — the `continue` on `not relevant`).

        Strategy: mock awatch to emit one unrelated event followed by one
        relevant event, then assert only the relevant flush is yielded.
        """
        import utils.relay as relay_module

        with Collector(source="real@0") as col:
            col.register(tag="plot1", data={"v": 1})

        # The unrelated path is a sibling of the relay dir — same parent, but
        # does not start with str(relay_dir).
        unrelated_path = relay_dir.parent / "unrelated" / "noise.txt"
        relevant_path  = relay_dir / "somefile"

        change_sets = [
            # First batch: only the unrelated path — must be skipped entirely.
            {(1, unrelated_path)},
            # Second batch: relay-dir path — must trigger _scan.
            {(1, relevant_path)},
        ]

        async def fake_awatch(_dir, **kwargs):
            for cs in change_sets:
                yield cs

        monkeypatch.setattr(relay_module, "awatch", fake_awatch)

        async def run():
            results = []
            async for source, tag, data in Deliverer().deliver():
                results.append((source, tag, data))
            return results

        results = asyncio.run(run())
        # Exactly one item from the real flush; the unrelated event yielded nothing.
        assert len(results) == 1
        assert results[0][:2] == ("real@0", "plot1")

    def test_deliver_recovers_after_relay_dir_deleted(self, relay_dir, monkeypatch):
        """
        When an awatch event arrives and _OUTPUT_DIR no longer exists, deliver()
        must clear _seen and continue watching (lines 551-553), so that a
        subsequent Collector flush is picked up after the dir is recreated.

        Strategy: mock awatch to emit three change-sets in sequence:
          1. relay dir is deleted   → deliver() hits lines 551-553 (clear + continue)
          2. relay dir is recreated → deliver() calls _scan and yields the entry
          3. (generator exhausted)  → deliver() exits the awatch loop cleanly

        We verify the *observable outcome* — that the re-flushed entry IS
        delivered — which is only possible if _seen was cleared in step 1
        (otherwise the entry would be suppressed as already-seen).
        """
        import utils.relay as relay_module
        import shutil

        # Flush initial data so _seen is non-empty after the first _scan.
        with Collector(source="revived@0") as col:
            col.register(tag="plot1", data={"v": 99})

        relevant_path = relay_dir / "somefile"

        async def fake_awatch(_dir, **kwargs):
            # Batch 1: relay dir is gone — triggers the clear+continue branch.
            shutil.rmtree(relay_dir, ignore_errors=True)
            yield {(1, relevant_path)}

            # Batch 2: relay dir recreated with the same source/tag — must be
            # delivered only if _seen was cleared in batch 1.
            with Collector(source="revived@0") as col:
                col.register(tag="plot1", data={"v": 99})
            yield {(1, relevant_path)}

        monkeypatch.setattr(relay_module, "awatch", fake_awatch)

        async def run():
            results = []
            async for source, tag, data in Deliverer().deliver():
                results.append((source, tag, data))
                return results  # stop after first delivered item

        results = asyncio.run(run())

        # If _seen was NOT cleared the entry would be suppressed and results
        # would be empty.  A non-empty result proves lines 551-553 fired.
        assert len(results) == 1
        assert results[0][:2] == ("revived@0", "plot1")


# ---------------------------------------------------------------------------
# ExpressionLoaderCollectorNaming (lines 576-577, 581-585)
# ---------------------------------------------------------------------------

class TestExpressionLoaderCollectorNaming:
    from utils.relay import ExpressionLoaderCollectorNaming as N

    def test_make_returns_correctly_formatted_string(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        result = N.make(expression="myExpr", id=7)
        expression, id_, ts = N.parse(result)
        assert expression == "myExpr"
        assert id_ == 7
        assert ts is not None

    def test_make_format_matches_pattern(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        result = N.make(expression="abc", id=0)
        assert N.REGEX.fullmatch(result) is not None

    def test_parse_returns_correct_fields(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        from datetime import datetime
        name = "myExpr#42@20240315T123456.789012"
        expression, id_, ts = N.parse(name)
        assert expression == "myExpr"
        assert id_ == 42
        assert isinstance(ts, datetime)

    def test_parse_raises_on_invalid_string(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        with pytest.raises(ValueError, match="Invalid collector name"):
            N.parse("not-valid")

    def test_parse_raises_on_wrong_timestamp_format(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        with pytest.raises(ValueError, match="Invalid collector name"):
            N.parse("expr#1@not-a-timestamp")

    def test_make_parse_roundtrip(self):
        from utils.relay import ExpressionLoaderCollectorNaming as N
        from datetime import datetime
        name = N.make(expression="loader", id=3)
        expression, id_, ts = N.parse(name)
        assert expression == "loader"
        assert id_ == 3
        assert (datetime.now() - ts).total_seconds() < 5.0


# ---------------------------------------------------------------------------
# Module-level history_cleanup()
# ---------------------------------------------------------------------------

class TestModuleHistoryCleanup:
    def test_cleanup_preserves_output_dir(self, relay_dir):
        """history_cleanup() should remove contents but keep the directory itself."""
        import utils.relay as relay_module

        relay_dir.mkdir(parents=True, exist_ok=True)
        (relay_dir / "some_file.json").write_text("{}")
        relay_module.history_cleanup()
        assert relay_dir.exists()
        assert list(relay_dir.iterdir()) == []

    def test_cleanup_removes_files_and_subdirs(self, relay_dir):
        """history_cleanup() should delete files and nested subdirectories."""
        import utils.relay as relay_module

        subdir = relay_dir / "subdir"
        subdir.mkdir(parents=True)
        (subdir / "nested.json").write_text("{}")
        (relay_dir / "top_level.json").write_text("{}")
        relay_module.history_cleanup()
        assert relay_dir.exists()
        assert list(relay_dir.iterdir()) == []

    def test_cleanup_noop_when_dir_absent(self, relay_dir):
        """history_cleanup() should not raise when the directory does not exist."""
        import utils.relay as relay_module

        assert not relay_dir.exists()
        relay_module.history_cleanup()  # must not raise

    def test_cleanup_filenotfounderror_is_silent(self, relay_dir):
        """history_cleanup() must not log an error when the directory is already gone."""
        import utils.relay as relay_module

        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            relay_module.history_cleanup()
            mock_exc.assert_not_called()

    def test_cleanup_unexpected_error_is_logged(self, relay_dir, monkeypatch):
        """history_cleanup() must log unexpected errors rather than swallowing them."""
        import utils.relay as relay_module
        import shutil

        subdir = relay_dir / "subdir"
        subdir.mkdir(parents=True)

        def boom(path):
            raise PermissionError("locked")

        monkeypatch.setattr(shutil, "rmtree", boom)
        with unittest.mock.patch.object(relay_module.log, "exception") as mock_exc:
            relay_module.history_cleanup()
            mock_exc.assert_called_once()

    def test_cleanup_entries_no_longer_delivered_after(self, relay_dir):
        """After history_cleanup(), a fresh Deliverer should find nothing to deliver."""
        import utils.relay as relay_module

        with Collector(source="test@0") as col:
            col.register(tag="plot1", data={"v": 1})

        relay_module.history_cleanup()
        results = asyncio.run(collect_delivered(Deliverer()))
        assert results == []


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

async def collect_delivered(
    deliverer: Deliverer,
    sources: "str | list[str] | None" = None,
    tags:    "str | list[str] | None" = None,
    timeout: float = 1.0,
) -> list[tuple[str, str, object]]:
    """
    Collect all items already present in _OUTPUT_DIR via the initial scan,
    then cancel.  The timeout guards against hangs if awatch blocks unexpectedly.
    """
    results = []
    try:
        await asyncio.wait_for(
            _drain(deliverer, sources, tags, results),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        pass
    return results


async def _drain(deliverer, sources, tags, results):
    async for source, tag, data in deliverer.deliver(sources, tags):
        results.append((source, tag, data))


@pytest.fixture(autouse=True)
def reset_deliverer_count():
    """Ensure _deliverer_count is reset to 0 between tests."""
    import utils.relay as relay_module
    relay_module._deliverer_count = 0
    yield
    relay_module._deliverer_count = 0


@pytest.fixture
def relay_dir(monkeypatch, tmp_path):
    """Redirect _OUTPUT_DIR to a temp path for each test."""
    import utils.relay as relay_module
    tmp_relay = tmp_path / "relay"
    monkeypatch.setattr(relay_module, "_OUTPUT_DIR", tmp_relay)
    yield tmp_relay
