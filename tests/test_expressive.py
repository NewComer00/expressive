"""Tests for expressive.py — process_expressions and setup_loggers."""

from datetime import datetime
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from expressive import process_expressions, setup_loggers


# ---------------------------------------------------------------------------
# Shared call helpers
# ---------------------------------------------------------------------------

# Default timestamp kwargs passed to every process_expressions call.
_TS = dict(ref_start=None, ref_end=None, utau_start=None, utau_end=None)

# The kwargs the loader class is instantiated with when timestamps are all None.
_LOADER_INIT_TS = dict(ref_start=None, ref_end=None, utau_start=None, utau_end=None)


class TestVersion:

    def test_version_import(self):
        from expressive import VERSION
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0
        parts = VERSION.split(".")
        assert len(parts) >= 2


class TestProcessExpressions:

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_basic(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'smoothness': Mock(default=2),
            'scaler':     Mock(default=1.0),
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[{"expression": "dyn", "smoothness": 3, "scaler": 2.0}],
            **_TS,
        )

        mock_copy.assert_called_once_with("input.ustx", "output.ustx")
        mock_get_loader.assert_called_once_with("dyn")
        mock_loader_class.assert_called_once_with(
            "ref.wav", "utau.wav", "output.ustx", **_LOADER_INIT_TS
        )
        mock_loader_instance.get_expression.assert_called_once_with(
            smoothness=3, scaler=2.0
        )
        mock_loader_instance.load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_multiple_expressions(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        mock_dyn = Mock()
        mock_dyn.get_args_dict.return_value = {'smoothness': Mock(default=2)}

        mock_pitd = Mock()
        mock_pitd.get_args_dict.return_value = {'confidence_utau': Mock(default=0.8)}

        mock_loader_classes = {
            'dyn':  Mock(return_value=mock_dyn),
            'pitd': Mock(return_value=mock_pitd),
        }
        mock_get_loader.side_effect = lambda exp: mock_loader_classes[exp]

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {"expression": "dyn",  "smoothness": 3},
                {"expression": "pitd", "confidence_utau": 0.9},
            ],
            **_TS,
        )

        assert mock_copy.call_count == 1
        assert mock_get_loader.call_count == 2
        mock_get_loader.assert_any_call("dyn")
        mock_get_loader.assert_any_call("pitd")
        mock_dyn.get_expression.assert_called_once()
        mock_dyn.load_to_ustx.assert_called_once_with(1)
        mock_pitd.get_expression.assert_called_once()
        mock_pitd.load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_invalid_type(
        self, mock_get_registered, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        with pytest.raises(ValueError, match="not supported"):
            process_expressions(
                utau_wav="utau.wav", ref_wav="ref.wav",
                ustx_input="input.ustx", ustx_output="output.ustx",
                track_number=1,
                expressions=[{"expression": "invalid_expr"}],
                **_TS,
            )

        mock_copy.assert_called_once()

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_with_defaults(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'smoothness':   Mock(default=2),
            'scaler':       Mock(default=1.0),
            'align_radius': Mock(default=1),
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[{"expression": "dyn", "smoothness": 5}],
            **_TS,
        )

        mock_loader_instance.get_expression.assert_called_once_with(
            smoothness=5, scaler=1.0, align_radius=1
        )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_empty_list(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1, expressions=[],
            **_TS,
        )

        mock_copy.assert_called_once()
        mock_get_loader.assert_not_called()

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_loader_exception(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_instance.get_expression.side_effect = RuntimeError("Audio processing failed")
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        with pytest.raises(RuntimeError, match="Audio processing failed"):
            process_expressions(
                utau_wav="utau.wav", ref_wav="ref.wav",
                ustx_input="input.ustx", ustx_output="output.ustx",
                track_number=1, expressions=[{"expression": "dyn"}],
                **_TS,
            )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_all_three_types(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        mock_instances = {}
        for expr_type in ('dyn', 'pitd', 'tenc'):
            m = Mock()
            m.get_args_dict.return_value = {}
            mock_instances[expr_type] = m

        mock_get_loader.side_effect = (
            lambda et: Mock(return_value=mock_instances[et])
        )

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {"expression": "dyn"},
                {"expression": "pitd"},
                {"expression": "tenc"},
            ],
            **_TS,
        )

        assert mock_get_loader.call_count == 3
        for expr_type in ('dyn', 'pitd', 'tenc'):
            mock_instances[expr_type].get_expression.assert_called_once()
            mock_instances[expr_type].load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_different_track_numbers(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=2, expressions=[{"expression": "dyn"}],
            **_TS,
        )

        mock_loader_instance.load_to_ustx.assert_called_with(2)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_preserves_arg_order(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'arg1': Mock(default=1),
            'arg2': Mock(default=2),
            'arg3': Mock(default=3),
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[{"expression": "dyn", "arg1": 10, "arg3": 30}],
            **_TS,
        )

        mock_loader_instance.get_expression.assert_called_once_with(
            arg1=10, arg2=2, arg3=30
        )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_timestamps_forwarded_to_loader(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        """ref_start/ref_end/utau_start/utau_end are forwarded to loader.__init__."""
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            ref_start="0:10", ref_end="1:30",
            utau_start="0:05", utau_end="1:25",
            expressions=[{"expression": "dyn"}],
        )

        mock_loader_class.assert_called_once_with(
            "ref.wav", "utau.wav", "output.ustx",
            ref_start="0:10", ref_end="1:30",
            utau_start="0:05", utau_end="1:25",
        )


class TestProcessExpressionsIntegration:

    @pytest.mark.integration
    @pytest.mark.requires_audio
    def test_process_with_real_files(self, tmp_path, has_example_files):
        if not has_example_files:
            pytest.skip("Example files not available")

        ustx_output = str(tmp_path / "output.ustx")
        process_expressions(
            utau_wav="examples/Прекрасное Далеко/utau.wav",
            ref_wav="examples/Прекрасное Далеко/reference.wav",
            ustx_input="examples/Прекрасное Далеко/project.ustx",
            ustx_output=ustx_output,
            track_number=1,
            expressions=[{"expression": "dyn", "align_radius": 1, "smoothness": 2, "scaler": 2.0}],
            **_TS,
        )

        assert Path(ustx_output).exists()
        from utils.ustx import load_ustx
        project = load_ustx(ustx_output)
        assert any(c.abbr == "dyn" for c in project.voice_parts[0].curves)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_realistic_scenario(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        mock_dyn = Mock()
        mock_dyn.get_args_dict.return_value = {
            'align_radius': Mock(default=1),
            'smoothness':   Mock(default=2),
            'scaler':       Mock(default=2.0),
        }
        mock_pitd = Mock()
        mock_pitd.get_args_dict.return_value = {
            'confidence_utau': Mock(default=0.8),
            'confidence_ref':  Mock(default=0.6),
            'align_radius':    Mock(default=1),
            'semitone_shift':  Mock(default=None),
            'smoothness':      Mock(default=2),
            'scaler':          Mock(default=2.0),
        }
        mock_tenc = Mock()
        mock_tenc.get_args_dict.return_value = {
            'align_radius': Mock(default=1),
            'smoothness':   Mock(default=2),
            'scaler':       Mock(default=2.0),
            'bias':         Mock(default=20),
        }
        mock_loaders = {
            'dyn':  Mock(return_value=mock_dyn),
            'pitd': Mock(return_value=mock_pitd),
            'tenc': Mock(return_value=mock_tenc),
        }
        mock_get_loader.side_effect = lambda exp: mock_loaders[exp]

        process_expressions(
            utau_wav="examples/test/utau.wav",
            ref_wav="examples/test/reference.wav",
            ustx_input="examples/test/project.ustx",
            ustx_output="examples/test/output.ustx",
            track_number=1,
            expressions=[
                {"expression": "dyn",  "align_radius": 1, "smoothness": 2, "scaler": 2.0},
                {"expression": "pitd", "confidence_utau": 0.8, "confidence_ref": 0.6,
                 "align_radius": 1, "semitone_shift": None, "smoothness": 2, "scaler": 2.0},
                {"expression": "tenc", "align_radius": 1, "smoothness": 2, "scaler": 2.0, "bias": 20},
            ],
            **_TS,
        )

        mock_dyn.get_expression.assert_called_once_with(
            align_radius=1, smoothness=2, scaler=2.0
        )
        mock_pitd.get_expression.assert_called_once_with(
            confidence_utau=0.8, confidence_ref=0.6,
            align_radius=1, semitone_shift=None, smoothness=2, scaler=2.0
        )
        mock_tenc.get_expression.assert_called_once_with(
            align_radius=1, smoothness=2, scaler=2.0, bias=20
        )


class TestEdgeCases:

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_with_none_values(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        mock_get_registered.return_value = ['pitd']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'semitone_shift': Mock(default=None)
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="input.ustx", ustx_output="output.ustx",
            track_number=1,
            expressions=[{"expression": "pitd", "semitone_shift": None}],
            **_TS,
        )

        mock_loader_instance.get_expression.assert_called_once_with(semitone_shift=None)

    @patch('expressive.copy')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_case_sensitive(
        self, mock_get_registered, mock_copy
    ):
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        with pytest.raises(ValueError):
            process_expressions(
                utau_wav="utau.wav", ref_wav="ref.wav",
                ustx_input="input.ustx", ustx_output="output.ustx",
                track_number=1,
                expressions=[{"expression": "DYN"}],
                **_TS,
            )


class TestSetupLoggers:
    """
    setup_loggers() configures two loggers:

      logger_app  — gets file_handler  + app_handler  (FileHandler + StreamHandler)
      logger_exp  — gets file_handler  + exp_handler  (FileHandler + StreamHandler)

    Both therefore have exactly 2 handlers.
    """

    def test_setup_loggers_creates_log_file(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert Path(log_path).exists()
            assert logger_app is not None
            assert logger_exp is not None
        assert Path(log_path).exists()
        Path(log_path).unlink()

    def test_setup_loggers_returns_correct_types(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert isinstance(logger_app, logging.Logger)
            assert isinstance(logger_exp, logging.Logger)
            assert isinstance(log_path, Path)
        Path(log_path).unlink()

    def test_setup_loggers_app_has_two_handlers(self):
        """logger_app receives file_handler + app_handler."""
        with setup_loggers() as (logger_app, _unused, log_path):
            assert len(logger_app.handlers) == 2
            types_ = {type(h).__name__ for h in logger_app.handlers}
            assert "FileHandler" in types_
            assert "StreamHandler" in types_
        Path(log_path).unlink()

    def test_setup_loggers_exp_has_two_handlers(self):
        """logger_exp receives file_handler + exp_handler (both added in setup_loggers)."""
        with setup_loggers() as (_unused, logger_exp, log_path):
            assert len(logger_exp.handlers) == 2
            types_ = {type(h).__name__ for h in logger_exp.handlers}
            assert "FileHandler" in types_
            assert "StreamHandler" in types_
        Path(log_path).unlink()

    def test_setup_loggers_sets_debug_level(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert logger_app.level == logging.DEBUG
            assert logger_exp.level == logging.DEBUG
        Path(log_path).unlink()

    def test_setup_loggers_writes_to_file(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            logger_app.info("Test message from app")
            logger_exp.debug("Test message from exp")

        log_content = Path(log_path).read_text(encoding="utf-8-sig")
        assert "Test message from app" in log_content
        assert "Test message from exp" in log_content
        Path(log_path).unlink()

    def test_setup_loggers_cleanup_on_exit(self):
        """All handlers are removed from both loggers after the context exits."""
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert len(logger_app.handlers) > 0
            assert len(logger_exp.handlers) > 0
        assert len(logger_app.handlers) == 0
        assert len(logger_exp.handlers) == 0
        Path(log_path).unlink()

    def test_setup_loggers_cleanup_on_exception(self):
        """Handlers are removed even when the body raises."""
        logger_app = logger_exp = log_path = None
        try:
            with setup_loggers() as (la, le, lp):
                logger_app, logger_exp, log_path = la, le, lp
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert len(logger_app.handlers) == 0
        assert len(logger_exp.handlers) == 0
        Path(log_path).unlink()

    def test_setup_loggers_log_file_naming(self):
        with setup_loggers() as (_, __, log_path):
            log_filename = Path(log_path).name
            assert log_filename.startswith(datetime.now().strftime("%Y%m%d_"))
            assert log_filename.endswith(".log")
        Path(log_path).unlink()

    def test_setup_loggers_writes_final_message(self):
        """The 'Logs saved to …' message is written during teardown."""
        with setup_loggers() as (_, __, log_path):
            pass
        log_content = Path(log_path).read_text(encoding="utf-8-sig")
        assert f"Logs saved to '{log_path}'" in log_content
        Path(log_path).unlink()

    def test_setup_loggers_file_encoding(self):
        with setup_loggers() as (logger_app, _, log_path):
            logger_app.info("Test with unicode: 你好世界 Привет мир")
        log_content = Path(log_path).read_text(encoding="utf-8-sig")
        assert "你好世界" in log_content
        assert "Привет мир" in log_content
        Path(log_path).unlink()


# ---------------------------------------------------------------------------
# SameFileError handling in process_expressions (lines 86-87)
# ---------------------------------------------------------------------------

class TestSameFileError:

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_same_file_error_is_silenced(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        """SameFileError raised by copy() must be caught and silently ignored."""
        from shutil import SameFileError
        mock_copy.side_effect = SameFileError
        mock_get_registered.return_value = []

        # Must not raise
        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="same.ustx", ustx_output="same.ustx",
            track_number=1, expressions=[],
            **_TS,
        )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_same_file_error_still_processes_expressions(
        self, mock_get_registered, mock_get_loader, mock_copy
    ):
        """Processing must continue normally after a SameFileError from copy()."""
        from shutil import SameFileError
        mock_copy.side_effect = SameFileError
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        process_expressions(
            utau_wav="utau.wav", ref_wav="ref.wav",
            ustx_input="same.ustx", ustx_output="same.ustx",
            track_number=1,
            expressions=[{"expression": "dyn"}],
            **_TS,
        )

        mock_loader_instance.get_expression.assert_called_once()
        mock_loader_instance.load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.get_registered_expressions')
    def test_other_copy_errors_propagate(
        self, mock_get_registered, mock_copy
    ):
        """Errors other than SameFileError raised by copy() must not be swallowed."""
        mock_copy.side_effect = OSError("disk full")
        mock_get_registered.return_value = []

        with pytest.raises(OSError, match="disk full"):
            process_expressions(
                utau_wav="utau.wav", ref_wav="ref.wav",
                ustx_input="input.ustx", ustx_output="output.ustx",
                track_number=1, expressions=[],
                **_TS,
            )


# ---------------------------------------------------------------------------
# main() — argument parsing and dispatch (lines 154-210)
# ---------------------------------------------------------------------------

# Shared helpers for building a mock argument namespace
def _make_general_arg(type_=str, default=None, help_=""):
    a = Mock()
    a.type    = type_
    a.default = default
    a.help    = help_
    return a


def _make_loader_args_mock():
    """Return a mock for getExpressionLoader(None) whose .args namespace covers
    every general argument that main() accesses."""
    loader_mock = Mock()
    loader_mock.args.utau_path    = _make_general_arg()
    loader_mock.args.ref_path     = _make_general_arg()
    loader_mock.args.ustx_path    = _make_general_arg()
    loader_mock.args.track_number = _make_general_arg(type_=int)
    loader_mock.args.utau_start   = _make_general_arg(default=None)
    loader_mock.args.utau_end     = _make_general_arg(default=None)
    loader_mock.args.ref_start    = _make_general_arg(default=None)
    loader_mock.args.ref_end      = _make_general_arg(default=None)
    loader_mock.get_args_dict.return_value = {}
    loader_mock.__name__ = "MockExpressionLoader"
    return loader_mock


class TestMain:
    """Tests for expressive.main().

    argparse is exercised by injecting sys.argv; all heavy collaborators
    (process_expressions, setup_loggers, getExpressionLoader, …) are mocked.
    """

    # Baseline argv that satisfies every required argument.
    _BASE_ARGV = [
        "expressive",
        "-u", "utau.wav",
        "-r", "ref.wav",
        "-i", "input.ustx",
        "-o", "output.ustx",
        "-t", "1",
        "-e", "dyn",
    ]

    def _run_main(self, argv=None, expressions=("dyn",), extra_loader_args=None):
        """Call main() with a fully-mocked environment.

        Returns (mock_process_expressions, mock_logger_app).
        """
        from expressive import main

        loader_mock = _make_loader_args_mock()
        if extra_loader_args is not None:
            loader_mock.get_args_dict.return_value = extra_loader_args

        mock_logger_app = Mock()
        mock_logger_exp = Mock()
        mock_log_path   = Mock()

        with patch("sys.argv", argv or self._BASE_ARGV), \
             patch("expressive.init_gettext"), \
             patch("expressive.get_registered_expressions", return_value=list(expressions)), \
             patch("expressive.getExpressionLoader", return_value=loader_mock), \
             patch("expressive.add_expression_args_group"), \
             patch("expressive.process_expressions") as mock_pe, \
             patch("expressive.setup_loggers") as mock_sl:

            mock_sl.return_value.__enter__ = Mock(
                return_value=(mock_logger_app, mock_logger_exp, mock_log_path)
            )
            mock_sl.return_value.__exit__ = Mock(return_value=False)

            main()

        return mock_pe, mock_logger_app

    # --- parser construction ---

    def test_main_calls_init_gettext(self):
        from expressive import main
        loader_mock = _make_loader_args_mock()
        with patch("sys.argv", self._BASE_ARGV), \
             patch("expressive.init_gettext") as mock_ig, \
             patch("expressive.get_registered_expressions", return_value=["dyn"]), \
             patch("expressive.getExpressionLoader", return_value=loader_mock), \
             patch("expressive.add_expression_args_group"), \
             patch("expressive.process_expressions"), \
             patch("expressive.setup_loggers") as mock_sl:
            mock_sl.return_value.__enter__ = Mock(return_value=(Mock(), Mock(), Mock()))
            mock_sl.return_value.__exit__  = Mock(return_value=False)
            main()
        mock_ig.assert_called_once()

    def test_main_calls_process_expressions(self):
        mock_pe, _ = self._run_main()
        mock_pe.assert_called_once()

    def test_main_passes_utau_wav(self):
        mock_pe, _ = self._run_main()
        _, kwargs = mock_pe.call_args
        assert mock_pe.call_args[0][0] == "utau.wav"

    def test_main_passes_ref_wav(self):
        mock_pe, _ = self._run_main()
        assert mock_pe.call_args[0][1] == "ref.wav"

    def test_main_passes_ustx_input(self):
        mock_pe, _ = self._run_main()
        assert mock_pe.call_args[0][2] == "input.ustx"

    def test_main_passes_ustx_output(self):
        mock_pe, _ = self._run_main()
        assert mock_pe.call_args[0][3] == "output.ustx"

    def test_main_passes_track_number(self):
        mock_pe, _ = self._run_main()
        assert mock_pe.call_args[0][4] == 1

    def test_main_passes_expressions_list(self):
        mock_pe, _ = self._run_main()
        expressions = mock_pe.call_args[0][9]
        assert isinstance(expressions, list)
        assert any(e["expression"] == "dyn" for e in expressions)

    def test_main_default_timestamps_are_none(self):
        mock_pe, _ = self._run_main()
        args = mock_pe.call_args[0]
        # ref_start=5, ref_end=6, utau_start=7, utau_end=8
        assert args[5] is None  # ref_start
        assert args[6] is None  # ref_end
        assert args[7] is None  # utau_start
        assert args[8] is None  # utau_end

    def test_main_passes_explicit_timestamps(self):
        argv = self._BASE_ARGV + [
            "--ref_start",  "0:10",
            "--ref_end",    "1:30",
            "--utau_start", "0:05",
            "--utau_end",   "1:25",
        ]
        mock_pe, _ = self._run_main(argv=argv)
        args = mock_pe.call_args[0]
        assert args[5] == "0:10"
        assert args[6] == "1:30"
        assert args[7] == "0:05"
        assert args[8] == "1:25"

    # --- logging behaviour ---

    def test_main_logs_starting_message(self):
        _, mock_logger_app = self._run_main()
        logged = " ".join(str(c) for c in mock_logger_app.info.call_args_list)
        assert "Starting" in logged or mock_logger_app.info.called

    def test_main_logs_success_on_no_exception(self):
        _, mock_logger_app = self._run_main()
        success_calls = [
            c for c in mock_logger_app.info.call_args_list
            if "completed" in str(c).lower() or "successfully" in str(c).lower()
        ]
        assert len(success_calls) >= 1

    def test_main_logs_exception_on_error(self):
        from expressive import main
        loader_mock = _make_loader_args_mock()
        mock_logger_app = Mock()

        with patch("sys.argv", self._BASE_ARGV), \
             patch("expressive.init_gettext"), \
             patch("expressive.get_registered_expressions", return_value=["dyn"]), \
             patch("expressive.getExpressionLoader", return_value=loader_mock), \
             patch("expressive.add_expression_args_group"), \
             patch("expressive.process_expressions", side_effect=RuntimeError("boom")), \
             patch("expressive.setup_loggers") as mock_sl:
            mock_sl.return_value.__enter__ = Mock(
                return_value=(mock_logger_app, Mock(), Mock())
            )
            mock_sl.return_value.__exit__ = Mock(return_value=False)
            main()

        mock_logger_app.exception.assert_called_once()

    def test_main_does_not_raise_on_process_error(self):
        """main() must not propagate exceptions from process_expressions."""
        from expressive import main
        loader_mock = _make_loader_args_mock()

        with patch("sys.argv", self._BASE_ARGV), \
             patch("expressive.init_gettext"), \
             patch("expressive.get_registered_expressions", return_value=["dyn"]), \
             patch("expressive.getExpressionLoader", return_value=loader_mock), \
             patch("expressive.add_expression_args_group"), \
             patch("expressive.process_expressions", side_effect=RuntimeError("boom")), \
             patch("expressive.setup_loggers") as mock_sl:
            mock_sl.return_value.__enter__ = Mock(
                return_value=(Mock(), Mock(), Mock())
            )
            mock_sl.return_value.__exit__ = Mock(return_value=False)
            main()   # must not raise

    # --- expression filtering ---

    def test_main_only_includes_selected_expressions(self):
        """Expressions not in -e flags must be excluded from the call."""
        argv = [
            "expressive",
            "-u", "utau.wav", "-r", "ref.wav",
            "-i", "input.ustx", "-o", "output.ustx",
            "-t", "1",
            "-e", "dyn",   # only dyn, not pitd
        ]
        mock_pe, _ = self._run_main(argv=argv, expressions=("dyn", "pitd"))
        expressions = mock_pe.call_args[0][9]
        names = [e["expression"] for e in expressions]
        assert "dyn"  in names
        assert "pitd" not in names

    def test_main_includes_multiple_expressions(self):
        argv = [
            "expressive",
            "-u", "utau.wav", "-r", "ref.wav",
            "-i", "input.ustx", "-o", "output.ustx",
            "-t", "1",
            "-e", "dyn", "-e", "pitd",
        ]
        mock_pe, _ = self._run_main(argv=argv, expressions=("dyn", "pitd"))
        expressions = mock_pe.call_args[0][9]
        names = [e["expression"] for e in expressions]
        assert "dyn"  in names
        assert "pitd" in names

    def test_main_expression_dict_contains_expression_key(self):
        mock_pe, _ = self._run_main()
        for exp in mock_pe.call_args[0][9]:
            assert "expression" in exp
