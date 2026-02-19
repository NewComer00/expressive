import pytest
import logging

from expressions.base import (
    Args,
    ExpressionLoader,
    register_expression,
    getExpressionLoader,
    get_registered_expressions,
    EXPRESSION_LOADER_TABLE
)


class TestArgs:
    """Test Args dataclass"""

    def test_args_creation(self):
        """Test creating Args object"""
        arg = Args(
            name="test_arg",
            type=int,
            default=10,
            help="Test argument"
        )
        assert arg.name == "test_arg"
        assert arg.type is int
        assert arg.default == 10
        assert arg.help == "Test argument"

    def test_args_with_none_default(self):
        """Test Args with None as default"""
        arg = Args(
            name="optional_arg",
            type=str,
            default=None,
            help="Optional argument"
        )
        assert arg.default is None

    def test_args_different_types(self):
        """Test Args with different types"""
        arg_int = Args("int_arg", int, 0, "Integer")
        arg_float = Args("float_arg", float, 0.0, "Float")
        arg_str = Args("str_arg", str, "", "String")
        arg_bool = Args("bool_arg", bool, False, "Boolean")

        assert arg_int.type is int
        assert arg_float.type is float
        assert arg_str.type is str
        assert arg_bool.type is bool


class TestExpressionLoader:
    """Test ExpressionLoader base class"""

    def test_loader_initialization(self, temp_ustx_file):
        """Test loader initialization"""
        loader = ExpressionLoader(
            ref_path="ref.wav",
            utau_path="utau.wav",
            ustx_path=str(temp_ustx_file)
        )

        assert loader.ref_path == "ref.wav"
        assert loader.utau_path == "utau.wav"
        assert loader.ustx_path == str(temp_ustx_file)
        assert loader.tempo == 120  # From temp USTX file
        assert loader.id > 0

    def test_loader_id_increment(self, temp_ustx_file):
        """Test that loader IDs increment"""
        loader1 = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))
        loader2 = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))

        assert loader2.id > loader1.id

    def test_loader_has_logger(self, temp_ustx_file):
        """Test that loader has a logger"""
        loader = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))

        assert hasattr(loader, 'logger')
        assert isinstance(loader.logger, logging.LoggerAdapter)

    def test_loader_reads_tempo(self, temp_dir):
        """Test that loader reads tempo from USTX"""
        # Create USTX with different tempo
        ustx_content = """tempos:
  - bpm: 140
    position: 0
voice_parts:
  - name: Track 1
"""
        ustx_path = temp_dir / "tempo_test.ustx"
        ustx_path.write_text(ustx_content, encoding='utf-8-sig')

        loader = ExpressionLoader("ref.wav", "utau.wav", str(ustx_path))
        assert loader.tempo == 140

    def test_get_args_dict(self):
        """Test getting args dictionary"""
        args_dict = ExpressionLoader.get_args_dict()

        assert isinstance(args_dict, dict)
        assert "ref_path" in args_dict
        assert "utau_path" in args_dict
        assert "ustx_path" in args_dict
        assert "track_number" in args_dict

        # Verify all are Args instances
        for _, value in args_dict.items():
            assert isinstance(value, Args)

    def test_get_args_dict_types(self):
        """Test args dictionary has correct types"""
        args_dict = ExpressionLoader.get_args_dict()

        assert args_dict["ref_path"].type is str
        assert args_dict["utau_path"].type is str
        assert args_dict["ustx_path"].type is str
        assert args_dict["track_number"].type is int

    def test_get_expression_default(self, temp_ustx_file):
        """Test default get_expression method"""
        loader = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))

        tick, val = loader.get_expression()

        # Default implementation returns empty lists
        assert len(tick) == 0
        assert len(val) == 0

    def test_expression_tick_val_initialization(self, temp_ustx_file):
        """Test that expression_tick and expression_val are initialized"""
        loader = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))

        assert hasattr(loader, 'expression_tick')
        assert hasattr(loader, 'expression_val')
        assert len(loader.expression_tick) == 0
        assert len(loader.expression_val) == 0


class TestRegistrationMechanism:
    """Test expression registration mechanism"""

    def test_register_expression(self, clean_expression_table):
        """Test registering an expression loader"""
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"
            expression_info = "Test expression"

        assert "test_expr" in EXPRESSION_LOADER_TABLE
        assert EXPRESSION_LOADER_TABLE["test_expr"] == TestLoader

    def test_register_multiple_expressions(self, clean_expression_table):
        """Test registering multiple expressions"""
        @register_expression
        class TestLoader1(ExpressionLoader):
            expression_name = "expr1"

        @register_expression
        class TestLoader2(ExpressionLoader):
            expression_name = "expr2"

        assert "expr1" in EXPRESSION_LOADER_TABLE
        assert "expr2" in EXPRESSION_LOADER_TABLE
        assert len(EXPRESSION_LOADER_TABLE) == 2

    def test_register_overwrites_existing(self, clean_expression_table):
        """Test that registering same name overwrites"""
        @register_expression
        class TestLoader1(ExpressionLoader):
            expression_name = "test_expr"

        @register_expression
        class TestLoader2(ExpressionLoader):
            expression_name = "test_expr"

        # Should have only one entry
        assert len(EXPRESSION_LOADER_TABLE) == 1
        # Should be the second loader
        assert EXPRESSION_LOADER_TABLE["test_expr"] == TestLoader2

    def test_register_expression_decorator_returns_class(self, clean_expression_table):
        """Test that decorator returns the class"""
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        # Should still be able to use the class
        assert TestLoader.expression_name == "test_expr"


class TestGetExpressionLoader:
    """Test getExpressionLoader function"""

    def test_get_expression_loader_registered(self, clean_expression_table):
        """Test getting a registered loader"""
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        loader_class = getExpressionLoader("test_expr")
        assert loader_class == TestLoader

    def test_get_expression_loader_none(self):
        """Test getting loader with None returns base class"""
        loader_class = getExpressionLoader(None)
        assert loader_class == ExpressionLoader

    def test_get_expression_loader_not_found(self):
        """Test getting unregistered loader raises error"""
        with pytest.raises(ValueError, match="not registered or not supported"):
            getExpressionLoader("nonexistent_expr")

    def test_get_expression_loader_case_sensitive(self, clean_expression_table):
        """Test that expression names are case sensitive"""
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        # Should work with exact name
        loader_class = getExpressionLoader("test_expr")
        assert loader_class == TestLoader

        # Should fail with different case
        with pytest.raises(ValueError):
            getExpressionLoader("TEST_EXPR")


class TestGetRegisteredExpressions:
    """Test get_registered_expressions function"""

    def test_get_registered_expressions_empty(self, clean_expression_table):
        """Test with no registered expressions"""
        registered = get_registered_expressions()
        assert registered == []

    def test_get_registered_expressions_single(self, clean_expression_table):
        """Test with single registered expression"""
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test_expr"

        registered = get_registered_expressions()
        assert "test_expr" in registered
        assert len(registered) == 1

    def test_get_registered_expressions_multiple(self, clean_expression_table):
        """Test with multiple registered expressions"""
        @register_expression
        class TestLoader1(ExpressionLoader):
            expression_name = "expr1"

        @register_expression
        class TestLoader2(ExpressionLoader):
            expression_name = "expr2"

        @register_expression
        class TestLoader3(ExpressionLoader):
            expression_name = "expr3"

        registered = get_registered_expressions()
        assert "expr1" in registered
        assert "expr2" in registered
        assert "expr3" in registered
        assert len(registered) == 3

    def test_get_registered_expressions_returns_list(self, clean_expression_table):
        """Test that function returns a list"""
        registered = get_registered_expressions()
        assert isinstance(registered, list)


class TestLoadToUSTX:
    """Test load_to_ustx method"""

    def test_load_to_ustx_with_data(self, clean_expression_table, temp_ustx_file):
        """Test loading expression to USTX with data"""
        import numpy as np

        # Create a custom loader with proper expression_name
        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "dyn"

        loader = TestLoader("ref.wav", "utau.wav", str(temp_ustx_file))
        loader.expression_tick = np.array([0, 480, 960])
        loader.expression_val = np.array([0, 50, 100])

        # Should not raise exception
        loader.load_to_ustx(track_number=1)

        # Verify file was modified
        from utils.ustx import load_ustx
        ustx_dict = load_ustx(str(temp_ustx_file))

        # Verify curve was added
        assert "curves" in ustx_dict["voice_parts"][0]
        curves = ustx_dict["voice_parts"][0]["curves"]
        assert len(curves) == 1
        assert curves[0]["abbr"] == "dyn"

    def test_load_to_ustx_empty_data(self, temp_ustx_file, caplog):
        """Test loading with empty data logs warning"""
        loader = ExpressionLoader("ref.wav", "utau.wav", str(temp_ustx_file))
        loader.expression_tick = []
        loader.expression_val = []

        with caplog.at_level(logging.WARNING):
            loader.load_to_ustx(track_number=1)

        # Should have warning about empty data
        assert any("empty" in record.message.lower() or "空" in record.message
                   for record in caplog.records)

    def test_load_to_ustx_thread_safety(self):
        """Test that ustx_lock exists for thread safety"""

        assert hasattr(ExpressionLoader, 'ustx_lock')
        # Check that it's a lock object by verifying it has lock methods
        assert hasattr(ExpressionLoader.ustx_lock, 'acquire')
        assert hasattr(ExpressionLoader.ustx_lock, 'release')
        assert callable(ExpressionLoader.ustx_lock.acquire)
        assert callable(ExpressionLoader.ustx_lock.release)


class TestCustomLoader:
    """Test creating custom expression loaders"""

    def test_custom_loader_with_args(self, clean_expression_table, temp_ustx_file):
        """Test custom loader with additional args"""
        from types import SimpleNamespace

        @register_expression
        class CustomLoader(ExpressionLoader):
            expression_name = "custom"
            expression_info = "Custom expression"
            args = SimpleNamespace(
                **ExpressionLoader.args.__dict__,
                custom_param=Args("custom_param", float, 1.0, "Custom parameter")
            )

        # Verify registration
        assert "custom" in EXPRESSION_LOADER_TABLE

        # Verify args
        loader_class = getExpressionLoader("custom")
        args_dict = loader_class.get_args_dict()
        assert "custom_param" in args_dict
        assert args_dict["custom_param"].default == 1.0

    def test_custom_loader_override_get_expression(self, clean_expression_table, temp_ustx_file):
        """Test custom loader with overridden get_expression"""
        import numpy as np

        @register_expression
        class CustomLoader(ExpressionLoader):
            expression_name = "custom"

            def get_expression(self, *args, **kwargs):
                # Custom implementation
                self.expression_tick = np.array([0, 480, 960])
                self.expression_val = np.array([10, 20, 30])
                return self.expression_tick, self.expression_val

        loader = CustomLoader("ref.wav", "utau.wav", str(temp_ustx_file))
        tick, val = loader.get_expression()

        assert len(tick) == 3
        assert len(val) == 3
        assert tick[0] == 0
        assert val[0] == 10


class TestExpressionLoaderIntegration:
    """Integration tests for expression loader"""

    def test_full_loader_workflow(self, clean_expression_table, temp_ustx_file):
        """Test complete workflow: register, instantiate, process"""
        import numpy as np

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test"

            def get_expression(self, smoothness=2):
                # Simple test implementation
                self.expression_tick = np.array([0, 480, 960])
                self.expression_val = np.array([0, 50, 100])
                return self.expression_tick, self.expression_val

        # Get loader class
        loader_class = getExpressionLoader("test")

        # Instantiate
        loader = loader_class("ref.wav", "utau.wav", str(temp_ustx_file))

        # Process
        tick, val = loader.get_expression(smoothness=3)

        # Verify
        assert len(tick) == 3
        assert len(val) == 3

    def test_multiple_loaders_independent(self, clean_expression_table, temp_ustx_file):
        """Test that multiple loader instances are independent"""
        import numpy as np

        @register_expression
        class TestLoader(ExpressionLoader):
            expression_name = "test"

        loader1 = TestLoader("ref1.wav", "utau1.wav", str(temp_ustx_file))
        loader2 = TestLoader("ref2.wav", "utau2.wav", str(temp_ustx_file))

        # Should have different IDs
        assert loader1.id != loader2.id

        # Should have different paths
        assert loader1.ref_path != loader2.ref_path
        assert loader1.utau_path != loader2.utau_path

        # Modifying one shouldn't affect the other
        loader1.expression_tick = np.array([0, 480])
        loader2.expression_tick = np.array([0, 960])

        assert len(loader1.expression_tick) == 2
        assert len(loader2.expression_tick) == 2
        assert loader1.expression_tick[1] != loader2.expression_tick[1]
