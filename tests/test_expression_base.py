"""Tests for expressions/base.py"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from expressions.base import (
    EXPRESSION_LOADER_TABLE,
    Args,
    ExpressionLoader,
    get_registered_expressions,
    getExpressionLoader,
    register_expression,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_silent_wav(path: str, duration: float = 3.0, sr: int = 22050) -> None:
    sf.write(path, np.zeros(int(duration * sr), dtype=np.float32), sr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_wav_pair(tmp_path):
    """Return (ref_wav_path, utau_wav_path) as plain strings."""
    ref  = str(tmp_path / "ref.wav")
    utau = str(tmp_path / "utau.wav")
    _write_silent_wav(ref)
    _write_silent_wav(utau)
    return ref, utau


@pytest.fixture()
def temp_ustx_file(tmp_path):
    """Minimal valid USTX file (BPM 120, one voice part covering ticks 0–9600)."""
    content = (
        "tempos:\n  - bpm: 120\n    position: 0\n"
        "time_signatures:\n  - bar_position: 0\n    beat_per_bar: 4\n    beat_unit: 4\n"
        "tracks:\n  - track_name: Track 1\n    track_color: Blue\n    singer: ''\n"
        "    phonemizer: ''\n    mute: false\n    solo: false\n    volume: 0.0\n    pan: 0.0\n"
        "voice_parts:\n  - name: Track 1\n    track_no: 0\n    position: 0\n    duration: 9600\n"
        "    notes: []\n    curves: []\n"
    )
    p = tmp_path / "test.ustx"
    p.write_text(content, encoding="utf-8-sig")
    return p


@pytest.fixture()
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def clean_expression_table():
    """Isolate EXPRESSION_LOADER_TABLE for every test."""
    original = dict(EXPRESSION_LOADER_TABLE)
    EXPRESSION_LOADER_TABLE.clear()
    yield
    EXPRESSION_LOADER_TABLE.clear()
    EXPRESSION_LOADER_TABLE.update(original)


# ---------------------------------------------------------------------------
# TestArgs
# ---------------------------------------------------------------------------

class TestArgs:

    def test_args_creation(self):
        arg = Args(name="test_arg", type=int, default=10, help="Test argument")
        assert arg.name == "test_arg"
        assert arg.type is int
        assert arg.default == 10
        assert arg.help == "Test argument"

    def test_args_with_none_default(self):
        arg = Args(name="optional_arg", type=str, default=None, help="Optional argument")
        assert arg.default is None

    def test_args_different_types(self):
        assert Args("int_arg",   int,   0,     "Integer").type is int
        assert Args("float_arg", float, 0.0,   "Float").type   is float
        assert Args("str_arg",   str,   "",    "String").type  is str
        assert Args("bool_arg",  bool,  False, "Boolean").type is bool

    def test_args_choices_default_none(self):
        arg = Args(name="a", type=str, default="x", help="h")
        assert arg.choices is None

    def test_args_choices_set(self):
        arg = Args(name="a", type=str, default="x", help="h", choices=["x", "y"])
        assert arg.choices == ["x", "y"]


# ---------------------------------------------------------------------------
# TestExpressionLoader
# ---------------------------------------------------------------------------

class TestExpressionLoader:

    def test_loader_initialization(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))

        # ref_path / utau_path now point to ClampedWav temp files, not the originals
        assert loader.ref_path.endswith(".wav")
        assert loader.utau_path.endswith(".wav")
        assert loader.ustx_path == str(temp_ustx_file)
        assert loader.id > 0

    def test_loader_offset_and_duration_stored(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert isinstance(loader.ref_offset,    float)
        assert isinstance(loader.ref_duration,  float)
        assert isinstance(loader.utau_offset,   float)
        assert isinstance(loader.utau_duration, float)
        assert loader.ref_offset  == pytest.approx(0.0)
        assert loader.utau_offset == pytest.approx(0.0)
        assert loader.ref_duration  > 0
        assert loader.utau_duration > 0

    def test_loader_id_increment(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        l1 = ExpressionLoader(ref, utau, str(temp_ustx_file))
        l2 = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert l2.id == l1.id + 1

    def test_loader_has_logger(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert isinstance(loader.logger, logging.LoggerAdapter)

    def test_loader_builds_time_axis(self, temp_wav_pair, temp_ustx_file):
        """The loader must expose a working TimeAxis, not a raw tempo value."""
        from utils.ustx import TimeAxis
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert isinstance(loader.ustx_time_axis, TimeAxis)
        # 120 BPM, 480 PPQN → 960 ticks/second
        ticks = loader.ustx_time_axis.seconds_to_ticks(np.array([1.0]))
        assert ticks[0] == 960

    def test_loader_time_axis_reflects_project_tempo(self, temp_wav_pair, temp_dir):
        """TimeAxis BPM is derived from the USTX file, not hardcoded."""
        ref, utau = temp_wav_pair
        ustx_path = temp_dir / "tempo_test.ustx"
        ustx_path.write_text(
            "tempos:\n  - bpm: 140\n    position: 0\n"
            "time_signatures:\n  - bar_position: 0\n    beat_per_bar: 4\n    beat_unit: 4\n"
            "tracks: []\n"
            "voice_parts:\n  - name: T\n    track_no: 0\n    position: 0\n    duration: 9600\n"
            "    notes: []\n    curves: []\n",
            encoding="utf-8-sig",
        )
        loader = ExpressionLoader(ref, utau, str(ustx_path))
        # 140 BPM → 140/60*480 ≈ 1120 ticks/second
        ticks = loader.ustx_time_axis.seconds_to_ticks(np.array([1.0]))
        expected = round(140 / 60 * 480)
        assert ticks[0] == expected

    def test_loader_trim_start(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file), ref_start="0:01")
        assert loader.ref_offset   == pytest.approx(1.0, abs=0.01)
        assert loader.ref_duration == pytest.approx(2.0, abs=0.1)

    def test_loader_trim_end(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file), ref_end="0:02")
        assert loader.ref_offset   == pytest.approx(0.0)
        assert loader.ref_duration == pytest.approx(2.0, abs=0.1)

    def test_loader_trim_both(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(
            ref, utau, str(temp_ustx_file),
            utau_start="0:01", utau_end="0:02",
        )
        assert loader.utau_offset   == pytest.approx(1.0, abs=0.01)
        assert loader.utau_duration == pytest.approx(1.0, abs=0.1)

    def test_clamped_wav_instances_kept_alive(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert hasattr(loader, "_clamped_ref")
        assert hasattr(loader, "_clamped_utau")

    def test_get_args_dict(self):
        args_dict = ExpressionLoader.get_args_dict()
        assert isinstance(args_dict, dict)
        for key in ("ref_path", "utau_path", "ustx_path", "track_number",
                    "ref_start", "ref_end", "utau_start", "utau_end"):
            assert key in args_dict
            assert isinstance(args_dict[key], Args)

    def test_get_args_dict_types(self):
        d = ExpressionLoader.get_args_dict()
        assert d["ref_path"].type     is str
        assert d["utau_path"].type    is str
        assert d["ustx_path"].type    is str
        assert d["track_number"].type is int

    def test_get_expression_default(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        tick, val = loader.get_expression()
        assert len(tick) == 0
        assert len(val)  == 0

    def test_expression_tick_val_initialization(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        assert len(loader.expression_tick) == 0
        assert len(loader.expression_val)  == 0


# ---------------------------------------------------------------------------
# TestRegistrationMechanism
# ---------------------------------------------------------------------------

class TestRegistrationMechanism:

    def test_register_expression(self):
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        assert "test_expr" in EXPRESSION_LOADER_TABLE
        assert EXPRESSION_LOADER_TABLE["test_expr"] is TestLoader

    def test_register_multiple_expressions(self):
        @register_expression
        class L1(ExpressionLoader):
            expression_name = "expr1"

        @register_expression
        class L2(ExpressionLoader):
            expression_name = "expr2"

        assert "expr1" in EXPRESSION_LOADER_TABLE
        assert "expr2" in EXPRESSION_LOADER_TABLE
        assert len(EXPRESSION_LOADER_TABLE) == 2

    def test_register_overwrites_existing(self):
        @register_expression
        class L1(ExpressionLoader):
            expression_name = "test_expr"

        @register_expression
        class L2(ExpressionLoader):
            expression_name = "test_expr"

        assert len(EXPRESSION_LOADER_TABLE) == 1
        assert EXPRESSION_LOADER_TABLE["test_expr"] is L2

    def test_register_expression_decorator_returns_class(self):
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        assert TestLoader.expression_name == "test_expr"


# ---------------------------------------------------------------------------
# TestGetExpressionLoader
# ---------------------------------------------------------------------------

class TestGetExpressionLoader:

    def test_get_registered_loader(self):
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        assert getExpressionLoader("test_expr") is TestLoader

    def test_get_none_returns_base(self):
        assert getExpressionLoader(None) is ExpressionLoader

    def test_get_not_found_raises(self):
        with pytest.raises(ValueError, match="not registered or not supported"):
            getExpressionLoader("nonexistent_expr")

    def test_case_sensitive(self):
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        assert getExpressionLoader("test_expr") is TestLoader
        with pytest.raises(ValueError):
            getExpressionLoader("TEST_EXPR")


# ---------------------------------------------------------------------------
# TestGetRegisteredExpressions
# ---------------------------------------------------------------------------

class TestGetRegisteredExpressions:

    def test_empty(self):
        assert get_registered_expressions() == []

    def test_single(self):
        @register_expression
        class L(ExpressionLoader):
            expression_name = "test_expr"

        result = get_registered_expressions()
        assert result == ["test_expr"]

    def test_multiple(self):
        for name in ("expr1", "expr2", "expr3"):
            register_expression(type(name, (ExpressionLoader,), {"expression_name": name}))

        result = get_registered_expressions()
        assert set(result) == {"expr1", "expr2", "expr3"}
        assert len(result) == 3

    def test_returns_list(self):
        assert isinstance(get_registered_expressions(), list)


# ---------------------------------------------------------------------------
# TestLoadToUSTX
# ---------------------------------------------------------------------------

class TestLoadToUSTX:

    def test_load_to_ustx_with_data(self, temp_wav_pair, temp_ustx_file):
        from utils.ustx import load_ustx

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "dyn"

        ref, utau = temp_wav_pair
        loader = TestLoader(ref, utau, str(temp_ustx_file))
        loader.expression_tick = np.array([0, 480, 960])
        loader.expression_val  = np.array([0,  50, 100], dtype=float)

        loader.load_to_ustx(track_number=1)

        project = load_ustx(str(temp_ustx_file))
        curve = project.voice_parts[0].get_curve("dyn")
        assert curve is not None
        assert curve.abbr == "dyn"

    def test_load_to_ustx_curve_values(self, temp_wav_pair, temp_ustx_file):
        """Values written should survive a save/load roundtrip."""
        from utils.ustx import load_ustx

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "dyn"

        ref, utau = temp_wav_pair
        loader = TestLoader(ref, utau, str(temp_ustx_file))
        loader.expression_tick = np.array([0, 480, 960])
        loader.expression_val  = np.array([0.0, 50.0, 100.0])

        loader.load_to_ustx(track_number=1)

        project = load_ustx(str(temp_ustx_file))
        curve = project.voice_parts[0].get_curve("dyn")
        assert curve.ys == [0, 50, 100]

    def test_load_to_ustx_empty_data_logs_warning(self, temp_wav_pair, temp_ustx_file, caplog):
        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))

        with caplog.at_level(logging.WARNING):
            loader.load_to_ustx(track_number=1)

        messages = " ".join(r.message for r in caplog.records)
        assert any(word in messages.lower() for word in ("empty", "空"))

    def test_load_to_ustx_track_number_1based(self, temp_wav_pair, temp_ustx_file):
        """track_number=1 must map to track_no=0 inside add_expression_to_track."""
        from utils.ustx import load_ustx

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "dyn"

        ref, utau = temp_wav_pair
        loader = TestLoader(ref, utau, str(temp_ustx_file))
        loader.expression_tick = np.array([0, 480])
        loader.expression_val  = np.array([10.0, 20.0])

        loader.load_to_ustx(track_number=1)

        project = load_ustx(str(temp_ustx_file))
        assert project.voice_parts[0].get_curve("dyn") is not None


# ---------------------------------------------------------------------------
# TestCustomLoader
# ---------------------------------------------------------------------------

class TestCustomLoader:

    def test_custom_loader_with_extra_arg(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair

        @register_expression
        class CustomLoader(ExpressionLoader):
            expression_name = "custom"
            args = SimpleNamespace(
                **ExpressionLoader.args.__dict__,
                custom_param=Args("custom_param", float, 1.0, "Custom parameter"),
            )

        assert "custom" in EXPRESSION_LOADER_TABLE
        args_dict = getExpressionLoader("custom").get_args_dict()
        assert "custom_param" in args_dict
        assert args_dict["custom_param"].default == 1.0

    def test_custom_loader_override_get_expression(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair

        @register_expression
        class CustomLoader(ExpressionLoader):
            expression_name = "custom"

            def get_expression(self, *args, **kwargs):
                self.expression_tick = np.array([0, 480, 960])
                self.expression_val  = np.array([10, 20, 30])
                return self.expression_tick, self.expression_val

        loader = CustomLoader(ref, utau, str(temp_ustx_file))
        tick, val = loader.get_expression()

        assert list(tick) == [0, 480, 960]
        assert list(val)  == [10, 20, 30]


# ---------------------------------------------------------------------------
# TestExpressionLoaderIntegration
# ---------------------------------------------------------------------------

class TestExpressionLoaderIntegration:

    def test_full_loader_workflow(self, temp_wav_pair, temp_ustx_file):
        ref, utau = temp_wav_pair

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test"

            def get_expression(self, smoothness=2):
                self.expression_tick = np.array([0, 480, 960])
                self.expression_val  = np.array([0,  50, 100])
                return self.expression_tick, self.expression_val

        loader = getExpressionLoader("test")(ref, utau, str(temp_ustx_file))
        tick, val = loader.get_expression(smoothness=3)

        assert len(tick) == 3
        assert len(val)  == 3

    def test_multiple_loaders_independent(self, tmp_path, temp_ustx_file):
        """Two instances share no mutable state."""
        paths = {}
        for name in ("ref1", "utau1", "ref2", "utau2"):
            p = str(tmp_path / f"{name}.wav")
            _write_silent_wav(p)
            paths[name] = p

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test"

        l1 = TestLoader(paths["ref1"], paths["utau1"], str(temp_ustx_file))
        l2 = TestLoader(paths["ref2"], paths["utau2"], str(temp_ustx_file))

        assert l1.id != l2.id
        assert l1.ref_path  != l2.ref_path
        assert l1.utau_path != l2.utau_path

        l1.expression_tick = np.array([0, 480])
        l2.expression_tick = np.array([0, 960])
        assert l1.expression_tick[1] != l2.expression_tick[1]

    def test_temp_files_cleaned_on_del(self, temp_wav_pair, temp_ustx_file):
        """ClampedWav temp files are removed when _cleanup() is called."""
        import os

        ref, utau = temp_wav_pair
        loader = ExpressionLoader(ref, utau, str(temp_ustx_file))
        ref_tmp  = loader.ref_path
        utau_tmp = loader.utau_path

        loader._clamped_ref._cleanup()
        loader._clamped_utau._cleanup()

        assert not os.path.exists(ref_tmp)
        assert not os.path.exists(utau_tmp)

    def test_tick_converters_registered_after_init(self, temp_wav_pair, temp_ustx_file):
        """set_tick_converters must be called during __init__ so seqtool functions work."""
        from utils.seqtool import _time_to_ticks_fn, _default_time_to_ticks
        ref, utau = temp_wav_pair
        ExpressionLoader(ref, utau, str(temp_ustx_file))
        # After init the registered converter must not be the default error-raiser
        assert _time_to_ticks_fn is not _default_time_to_ticks
