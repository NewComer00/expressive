import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

from expressive import process_expressions


class TestProcessExpressions:
    """Test the main process_expressions function"""

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_basic(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test basic expression processing flow"""
        # Setup: Mock registered expressions
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Setup: Create a mock loader class and instance
        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'smoothness': Mock(default=2),
            'scaler': Mock(default=1.0)
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Execute
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {
                    "expression": "dyn",
                    "smoothness": 3,
                    "scaler": 2.0
                }
            ]
        )

        # Verify: File was copied
        mock_copy.assert_called_once_with("input.ustx", "output.ustx")

        # Verify: Loader was retrieved
        mock_get_loader.assert_called_once_with("dyn")

        # Verify: Loader was instantiated with correct paths
        mock_loader_class.assert_called_once_with(
            "ref.wav",
            "utau.wav",
            "output.ustx"
        )

        # Verify: get_expression was called with correct args
        mock_loader_instance.get_expression.assert_called_once_with(
            smoothness=3,
            scaler=2.0
        )

        # Verify: load_to_ustx was called
        mock_loader_instance.load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_multiple_expressions(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test processing multiple expressions in sequence"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Create separate mock instances for each expression
        mock_dyn_instance = Mock()
        mock_dyn_instance.get_args_dict.return_value = {
            'smoothness': Mock(default=2)
        }

        mock_pitd_instance = Mock()
        mock_pitd_instance.get_args_dict.return_value = {
            'confidence_utau': Mock(default=0.8)
        }

        # Setup loader to return different instances
        mock_loader_classes = {
            'dyn': Mock(return_value=mock_dyn_instance),
            'pitd': Mock(return_value=mock_pitd_instance)
        }
        mock_get_loader.side_effect = lambda exp: mock_loader_classes[exp]

        # Execute with multiple expressions
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {"expression": "dyn", "smoothness": 3},
                {"expression": "pitd", "confidence_utau": 0.9}
            ]
        )

        # Verify: File copied only once
        assert mock_copy.call_count == 1

        # Verify: Both loaders were called
        assert mock_get_loader.call_count == 2
        mock_get_loader.assert_any_call("dyn")
        mock_get_loader.assert_any_call("pitd")

        # Verify: Both expressions were processed
        mock_dyn_instance.get_expression.assert_called_once()
        mock_dyn_instance.load_to_ustx.assert_called_once_with(1)
        mock_pitd_instance.get_expression.assert_called_once()
        mock_pitd_instance.load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_invalid_type(
        self,
        mock_get_registered,
        mock_copy
    ):
        """Test that invalid expression type raises ValueError"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Execute with invalid expression type
        with pytest.raises(ValueError, match="not supported"):
            process_expressions(
                utau_wav="utau.wav",
                ref_wav="ref.wav",
                ustx_input="input.ustx",
                ustx_output="output.ustx",
                track_number=1,
                expressions=[
                    {"expression": "invalid_expr"}
                ]
            )

        # Verify: File was still copied before error
        mock_copy.assert_called_once()

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_with_defaults(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test that default arguments are used when not provided"""
        mock_get_registered.return_value = ['dyn']

        # Setup loader with default args
        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'smoothness': Mock(default=2),
            'scaler': Mock(default=1.0),
            'align_radius': Mock(default=1)
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Execute with only some args provided
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {
                    "expression": "dyn",
                    "smoothness": 5  # Only override smoothness
                }
            ]
        )

        # Verify: get_expression was called with provided + default args
        mock_loader_instance.get_expression.assert_called_once_with(
            smoothness=5,        # Provided
            scaler=1.0,          # Default
            align_radius=1       # Default
        )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_empty_list(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test processing with empty expression list"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Execute with empty expressions
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[]
        )

        # Verify: File was copied
        mock_copy.assert_called_once()

        # Verify: No loaders were called
        mock_get_loader.assert_not_called()

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_loader_exception(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test handling of exceptions from expression loader"""
        mock_get_registered.return_value = ['dyn']

        # Setup loader to raise exception
        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_instance.get_expression.side_effect = RuntimeError("Audio processing failed")
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Execute and expect exception to propagate
        with pytest.raises(RuntimeError, match="Audio processing failed"):
            process_expressions(
                utau_wav="utau.wav",
                ref_wav="ref.wav",
                ustx_input="input.ustx",
                ustx_output="output.ustx",
                track_number=1,
                expressions=[{"expression": "dyn"}]
            )

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_all_three_types(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test processing all three expression types"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Create mock instances for each type
        mock_instances = {}
        for expr_type in ['dyn', 'pitd', 'tenc']:
            mock_instance = Mock()
            mock_instance.get_args_dict.return_value = {}
            mock_instances[expr_type] = mock_instance

        # Setup loader to return appropriate instance
        def get_loader_side_effect(expr_type):
            return Mock(return_value=mock_instances[expr_type])

        mock_get_loader.side_effect = get_loader_side_effect

        # Execute with all three types
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {"expression": "dyn"},
                {"expression": "pitd"},
                {"expression": "tenc"}
            ]
        )

        # Verify all three were processed
        assert mock_get_loader.call_count == 3
        for expr_type in ['dyn', 'pitd', 'tenc']:
            mock_instances[expr_type].get_expression.assert_called_once()
            mock_instances[expr_type].load_to_ustx.assert_called_once_with(1)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_different_track_numbers(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test processing with different track numbers"""
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {}
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Test with track 2
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=2,
            expressions=[{"expression": "dyn"}]
        )

        # Verify load_to_ustx was called with track 2
        mock_loader_instance.load_to_ustx.assert_called_with(2)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_preserves_arg_order(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test that argument order is preserved"""
        mock_get_registered.return_value = ['dyn']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'arg1': Mock(default=1),
            'arg2': Mock(default=2),
            'arg3': Mock(default=3)
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Execute with specific args
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {
                    "expression": "dyn",
                    "arg1": 10,
                    "arg3": 30
                }
            ]
        )

        # Verify get_expression was called with correct kwargs
        mock_loader_instance.get_expression.assert_called_once_with(
            arg1=10,
            arg2=2,  # Default
            arg3=30
        )


class TestProcessExpressionsIntegration:
    """Integration tests using real components where possible"""

    @pytest.mark.integration
    @pytest.mark.requires_audio
    def test_process_with_real_files(self, tmp_path, has_example_files):
        """Test with real audio files from examples (slow test)"""
        if not has_example_files:
            pytest.skip("Example files not available")

        utau_wav = "examples/Прекрасное Далеко/utau.wav"
        ref_wav = "examples/Прекрасное Далеко/reference.wav"
        ustx_input = "examples/Прекрасное Далеко/project.ustx"
        ustx_output = tmp_path / "output.ustx"

        # Execute with minimal expression
        process_expressions(
            utau_wav=utau_wav,
            ref_wav=ref_wav,
            ustx_input=ustx_input,
            ustx_output=str(ustx_output),
            track_number=1,
            expressions=[
                {
                    "expression": "dyn",
                    "align_radius": 1,
                    "smoothness": 2,
                    "scaler": 2.0
                }
            ]
        )

        # Verify: Output file was created
        assert ustx_output.exists()

        # Verify: Output file contains expression data
        from utils.ustx import load_ustx
        ustx_dict = load_ustx(str(ustx_output))
        assert "curves" in ustx_dict["voice_parts"][0]
        curves = ustx_dict["voice_parts"][0]["curves"]
        assert any(c["abbr"] == "dyn" for c in curves)

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_realistic_scenario(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test realistic scenario with typical parameters"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Create realistic mock instances
        mock_dyn = Mock()
        mock_dyn.get_args_dict.return_value = {
            'align_radius': Mock(default=1),
            'smoothness': Mock(default=2),
            'scaler': Mock(default=2.0)
        }

        mock_pitd = Mock()
        mock_pitd.get_args_dict.return_value = {
            'confidence_utau': Mock(default=0.8),
            'confidence_ref': Mock(default=0.6),
            'align_radius': Mock(default=1),
            'semitone_shift': Mock(default=None),
            'smoothness': Mock(default=2),
            'scaler': Mock(default=2.0)
        }

        mock_tenc = Mock()
        mock_tenc.get_args_dict.return_value = {
            'align_radius': Mock(default=1),
            'smoothness': Mock(default=2),
            'scaler': Mock(default=2.0),
            'bias': Mock(default=20)
        }

        mock_loaders = {
            'dyn': Mock(return_value=mock_dyn),
            'pitd': Mock(return_value=mock_pitd),
            'tenc': Mock(return_value=mock_tenc)
        }
        mock_get_loader.side_effect = lambda exp: mock_loaders[exp]

        # Execute with realistic parameters (from expressive.py example)
        process_expressions(
            utau_wav="examples/test/utau.wav",
            ref_wav="examples/test/reference.wav",
            ustx_input="examples/test/project.ustx",
            ustx_output="examples/test/output.ustx",
            track_number=1,
            expressions=[
                {
                    "expression": "dyn",
                    "align_radius": 1,
                    "smoothness": 2,
                    "scaler": 2.0,
                },
                {
                    "expression": "pitd",
                    "confidence_utau": 0.8,
                    "confidence_ref": 0.6,
                    "align_radius": 1,
                    "semitone_shift": None,
                    "smoothness": 2,
                    "scaler": 2.0,
                },
                {
                    "expression": "tenc",
                    "align_radius": 1,
                    "smoothness": 2,
                    "scaler": 2.0,
                    "bias": 20,
                },
            ]
        )

        # Verify all three were processed with correct parameters
        mock_dyn.get_expression.assert_called_once_with(
            align_radius=1, smoothness=2, scaler=2.0
        )
        mock_pitd.get_expression.assert_called_once_with(
            confidence_utau=0.8,
            confidence_ref=0.6,
            align_radius=1,
            semitone_shift=None,
            smoothness=2,
            scaler=2.0
        )
        mock_tenc.get_expression.assert_called_once_with(
            align_radius=1, smoothness=2, scaler=2.0, bias=20
        )


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch('expressive.copy')
    @patch('expressive.getExpressionLoader')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_with_none_values(
        self,
        mock_get_registered,
        mock_get_loader,
        mock_copy
    ):
        """Test processing with None parameter values"""
        mock_get_registered.return_value = ['pitd']

        mock_loader_instance = Mock()
        mock_loader_instance.get_args_dict.return_value = {
            'semitone_shift': Mock(default=None)
        }
        mock_loader_class = Mock(return_value=mock_loader_instance)
        mock_get_loader.return_value = mock_loader_class

        # Execute with None value
        process_expressions(
            utau_wav="utau.wav",
            ref_wav="ref.wav",
            ustx_input="input.ustx",
            ustx_output="output.ustx",
            track_number=1,
            expressions=[
                {
                    "expression": "pitd",
                    "semitone_shift": None
                }
            ]
        )

        # Verify None was passed correctly
        mock_loader_instance.get_expression.assert_called_once_with(
            semitone_shift=None
        )

    @patch('expressive.copy')
    @patch('expressive.get_registered_expressions')
    def test_process_expressions_case_sensitive(
        self,
        mock_get_registered,
        mock_copy
    ):
        """Test that expression names are case sensitive"""
        mock_get_registered.return_value = ['dyn', 'pitd', 'tenc']

        # Try with wrong case
        with pytest.raises(ValueError):
            process_expressions(
                utau_wav="utau.wav",
                ref_wav="ref.wav",
                ustx_input="input.ustx",
                ustx_output="output.ustx",
                track_number=1,
                expressions=[{"expression": "DYN"}]  # Wrong case
            )
