"""Tests for expressive.py — process_expressions and setup_loggers."""

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
        ustx_dict = load_ustx(ustx_output)
        assert "curves" in ustx_dict["voice_parts"][0]
        assert any(c["abbr"] == "dyn" for c in ustx_dict["voice_parts"][0]["curves"])

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
            assert isinstance(log_path, str)
        Path(log_path).unlink()

    def test_setup_loggers_configures_handlers(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert len(logger_app.handlers) == 2
            handler_types = [type(h).__name__ for h in logger_app.handlers]
            assert 'FileHandler'   in handler_types
            assert 'StreamHandler' in handler_types
            assert len(logger_exp.handlers) == 1
            assert isinstance(logger_exp.handlers[0], logging.FileHandler)
        Path(log_path).unlink()

    def test_setup_loggers_sets_debug_level(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert logger_app.level == logging.DEBUG
            assert logger_exp.level  == logging.DEBUG
        Path(log_path).unlink()

    def test_setup_loggers_writes_to_file(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            logger_app.info("Test message from app")
            logger_exp.debug("Test message from exp")

        log_content = Path(log_path).read_text(encoding='utf-8-sig')
        assert "Test message from app" in log_content
        assert "Test message from exp" in log_content
        Path(log_path).unlink()

    def test_setup_loggers_cleanup_on_exit(self):
        with setup_loggers() as (logger_app, logger_exp, log_path):
            assert len(logger_app.handlers) > 0
            assert len(logger_exp.handlers) > 0
        assert len(logger_app.handlers) == 0
        assert len(logger_exp.handlers) == 0
        Path(log_path).unlink()

    def test_setup_loggers_cleanup_on_exception(self):
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
            assert log_filename.startswith("expressive_cli_")
            assert log_filename.endswith(".log")
        Path(log_path).unlink()

    def test_setup_loggers_writes_final_message(self):
        with setup_loggers() as (_, __, log_path):
            pass
        log_content = Path(log_path).read_text(encoding='utf-8-sig')
        assert f"Logs saved to {log_path}" in log_content
        Path(log_path).unlink()

    def test_setup_loggers_file_encoding(self):
        with setup_loggers() as (logger_app, _, log_path):
            logger_app.info("Test with unicode: 你好世界 Привет мир")
        log_content = Path(log_path).read_text(encoding='utf-8-sig')
        assert "你好世界"    in log_content
        assert "Привет мир" in log_content
        Path(log_path).unlink()
