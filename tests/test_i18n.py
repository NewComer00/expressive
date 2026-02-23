"""Tests for i18n utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest

from utils.i18n import LazyString, LazyStringEncoder, _, _l, init_gettext, json_dumps, patch_nicegui_json


class TestLazyString:
    """Test LazyString class."""

    def test_lazy_string_creation(self):
        """Test creating a LazyString."""
        ls = LazyString("Hello")
        assert ls._msg == "Hello"

    def test_lazy_string_str_conversion(self):
        """Test converting LazyString to string."""
        ls = LazyString("Hello")
        assert str(ls) == "Hello"

    def test_lazy_string_repr(self):
        """Test LazyString repr."""
        ls = LazyString("Hello")
        assert repr(ls) == "LazyString('Hello')"

    def test_lazy_string_equality(self):
        """Test LazyString equality comparison."""
        ls1 = LazyString("Hello")
        ls2 = LazyString("Hello")
        assert ls1 == ls2
        assert ls1 == "Hello"

    def test_lazy_string_hash(self):
        """Test LazyString hashing."""
        ls1 = LazyString("Hello")
        ls2 = LazyString("Hello")
        assert hash(ls1) == hash(ls2)

    def test_lazy_string_len(self):
        """Test LazyString length."""
        ls = LazyString("Hello")
        assert len(ls) == 5

    def test_lazy_string_add(self):
        """Test LazyString concatenation."""
        ls = LazyString("Hello")
        result = ls + " World"
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_lazy_string_radd(self):
        """Test LazyString right concatenation."""
        ls = LazyString("World")
        result = "Hello " + ls
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_lazy_string_mod(self):
        """Test LazyString %-formatting."""
        ls = LazyString("Hello %s")
        result = ls % "World"
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_lazy_string_format(self):
        """Test LazyString .format()."""
        ls = LazyString("Hello {name}")
        result = ls.format(name="World")
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_lazy_string_format_map(self):
        """Test LazyString .format_map()."""
        ls = LazyString("Hello {name}")
        result = ls.format_map({"name": "World"})
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_lazy_string_json_method(self):
        """Test LazyString __json__() method."""
        ls = LazyString("Hello")
        assert ls.__json__() == "Hello"


class TestTranslationFunctions:
    """Test translation functions."""

    def test_underscore_without_init(self):
        """Test _() returns original string when not initialized."""
        # Reset the translation
        import utils.i18n
        utils.i18n._current_gettext = None

        result = _("Test message")
        assert result == "Test message"

    def test_lazy_string_function(self):
        """Test _l() returns LazyString."""
        result = _l("Test message")
        assert isinstance(result, LazyString)
        assert str(result) == "Test message"

    def test_init_gettext(self, tmp_path):
        """Test init_gettext with fallback."""
        # This should not raise even with non-existent locale
        init_gettext("en_US", str(tmp_path), "messages")
        result = _("Test")
        assert result == "Test"  # Fallback to identity


class TestJSONSerialization:
    """Test JSON serialization of LazyString."""

    def test_lazy_string_encoder_default(self):
        """Test LazyStringEncoder.default() with LazyString."""
        ls = LazyString("Hello")
        result = LazyStringEncoder.default(ls)
        assert result == "Hello"

    def test_lazy_string_encoder_default_raises(self):
        """Test LazyStringEncoder.default() raises for non-LazyString."""
        with pytest.raises(TypeError, match="Type is not JSON serializable"):
            LazyStringEncoder.default(object())

    def test_json_dumps_with_lazy_string(self):
        """Test json_dumps() handles LazyString."""
        data = {"message": _l("Hello"), "count": 42}
        result = json_dumps(data)
        parsed = json.loads(result)
        assert parsed == {"message": "Hello", "count": 42}

    def test_json_dumps_with_nested_lazy_string(self):
        """Test json_dumps() handles nested LazyString."""
        data = {
            "outer": {
                "inner": _l("Nested"),
                "list": [_l("Item1"), _l("Item2")]
            }
        }
        result = json_dumps(data)
        parsed = json.loads(result)
        assert parsed == {
            "outer": {
                "inner": "Nested",
                "list": ["Item1", "Item2"]
            }
        }

    def test_standard_json_dumps_fails(self):
        """Test that standard json.dumps fails with LazyString."""
        data = {"message": _l("Hello")}
        with pytest.raises(TypeError):
            json.dumps(data)


class TestNiceGUIPatch:
    """Test NiceGUI JSON patching."""

    def test_patch_nicegui_json_success(self):
        """Test patch_nicegui_json() successfully patches orjson converter."""
        # Mock the nicegui.json.orjson_wrapper module
        mock_orjson_wrapper = MagicMock()
        original_converter = MagicMock()
        mock_orjson_wrapper._orjson_converter = original_converter

        with patch.dict('sys.modules', {'nicegui.json.orjson_wrapper': mock_orjson_wrapper}):
            patch_nicegui_json()

            # Verify the converter was replaced
            assert mock_orjson_wrapper._orjson_converter != original_converter

            # Test the patched converter handles LazyString
            patched_converter = mock_orjson_wrapper._orjson_converter
            ls = LazyString("Test")
            result = patched_converter(ls)
            assert result == "Test"

            # Test the patched converter falls back to original for other types
            test_obj = object()
            patched_converter(test_obj)
            original_converter.assert_called_once_with(test_obj)

    def test_patch_nicegui_json_no_nicegui(self):
        """Test patch_nicegui_json() handles missing NiceGUI gracefully."""
        with patch.dict('sys.modules', {'nicegui.json.orjson_wrapper': None}):
            # Should not raise
            patch_nicegui_json()

    def test_patch_nicegui_json_import_error(self):
        """Test patch_nicegui_json() handles ImportError gracefully."""
        with patch('utils.i18n.patch_nicegui_json') as mock_patch:
            mock_patch.side_effect = ImportError("No module named 'nicegui'")
            # Should not raise
            try:
                mock_patch()
            except ImportError:
                pass  # Expected
