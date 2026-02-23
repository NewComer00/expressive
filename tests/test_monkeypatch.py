"""Tests for monkeypatch utilities."""

import sys
from unittest.mock import Mock, patch
from types import ModuleType

import pytest
import runpy

from utils.monkeypatch import ensure_same_signature, patch_runpy


class TestEnsureSameSignature:
    """Test signature validation function."""

    def test_ensure_same_signature_identical(self):
        """Test that identical signatures pass."""
        def func1(a, b, c=None):
            pass

        def func2(a, b, c=None):
            pass

        # Should not raise
        ensure_same_signature(func1, func2)

    def test_ensure_same_signature_different_params(self):
        """Test that different parameters raise error."""
        def func1(a, b):
            pass

        def func2(a, b, c):
            pass

        with pytest.raises(RuntimeError, match="Signature mismatch"):
            ensure_same_signature(func1, func2)

    def test_ensure_same_signature_different_defaults(self):
        """Test that different defaults raise error."""
        def func1(a, b=1):
            pass

        def func2(a, b=2):
            pass

        with pytest.raises(RuntimeError, match="Signature mismatch"):
            ensure_same_signature(func1, func2)

    def test_ensure_same_signature_different_types(self):
        """Test that different type annotations raise error."""
        def func1(a: int, b: str):
            pass

        def func2(a: str, b: int):
            pass

        with pytest.raises(RuntimeError, match="Signature mismatch"):
            ensure_same_signature(func1, func2)

    def test_ensure_same_signature_with_kwargs(self):
        """Test signatures with **kwargs."""
        def func1(a, **kwargs):
            pass

        def func2(a, **kwargs):
            pass

        # Should not raise
        ensure_same_signature(func1, func2)


class TestPatchRunpyNonFrozen:
    """Test patch_runpy in non-frozen environment."""

    def test_patch_runpy_no_op_when_not_frozen(self):
        """Test that patch_runpy does nothing when not frozen."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", False, create=True):
            with patch_runpy():
                # Should still be the original function
                assert runpy.run_path == original_run_path

        # Should still be the original after context
        assert runpy.run_path == original_run_path

    def test_patch_runpy_decorator_non_frozen(self):
        """Test patch_runpy as decorator in non-frozen mode."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", False, create=True):
            @patch_runpy()
            def test_func():
                return runpy.run_path

            result = test_func()
            assert result == original_run_path


class TestPatchRunpyFrozen:
    """Test patch_runpy in frozen environment."""

    def test_patch_runpy_patches_when_frozen(self):
        """Test that patch_runpy patches runpy.run_path when frozen."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", True, create=True):
            with patch_runpy(strict_signature=False):
                # Should be patched
                assert runpy.run_path != original_run_path
                assert runpy.run_path == patch_runpy._run_path_frozen

        # Should be restored after context
        assert runpy.run_path == original_run_path

    def test_patch_runpy_restores_on_exit(self):
        """Test that original function is restored on exit."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", True, create=True):
            with patch_runpy(strict_signature=False):
                pass

        assert runpy.run_path == original_run_path

    def test_patch_runpy_restores_on_exception(self):
        """Test that original function is restored even on exception."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", True, create=True):
            try:
                with patch_runpy(strict_signature=False):
                    raise ValueError("Test exception")
            except ValueError:
                pass

        assert runpy.run_path == original_run_path

    def test_patch_runpy_strict_signature_check(self):
        """Test that strict signature checking works."""
        # The actual runpy.run_path signature should match _run_path_frozen
        with patch.object(sys, "frozen", True, create=True):
            # Should not raise with strict_signature=True
            with patch_runpy(strict_signature=True):
                pass


class TestRunPathFrozen:
    """Test the _run_path_frozen implementation."""

    def test_run_path_frozen_basic(self):
        """Test basic execution of _run_path_frozen."""
        # Create a mock __main__ module with code
        mock_main = ModuleType("__main__")
        code = compile("result = 42", "<string>", "exec")
        mock_main._pyi_main_co = code

        with patch.dict(sys.modules, {"__main__": mock_main}):
            result = patch_runpy._run_path_frozen("/fake/path")

            assert result["result"] == 42
            assert result["__name__"] == "__main__"
            assert result["__file__"] == "/fake/path"

    def test_run_path_frozen_with_init_globals(self):
        """Test _run_path_frozen with initial globals."""
        mock_main = ModuleType("__main__")
        code = compile("result = initial_value + 10", "<string>", "exec")
        mock_main._pyi_main_co = code

        with patch.dict(sys.modules, {"__main__": mock_main}):
            result = patch_runpy._run_path_frozen(
                "/fake/path",
                init_globals={"initial_value": 5}
            )

            assert result["result"] == 15

    def test_run_path_frozen_with_run_name(self):
        """Test _run_path_frozen with custom run_name."""
        mock_main = ModuleType("__main__")
        code = compile("name = __name__", "<string>", "exec")
        mock_main._pyi_main_co = code

        with patch.dict(sys.modules, {"__main__": mock_main}):
            result = patch_runpy._run_path_frozen(
                "/fake/path",
                run_name="custom_name"
            )

            assert result["name"] == "custom_name"

    def test_run_path_frozen_no_main_module(self):
        """Test _run_path_frozen raises error when __main__ not found."""
        with patch.dict(sys.modules, {}, clear=True):
            with pytest.raises(RuntimeError, match="Cannot locate __main__ module"):
                patch_runpy._run_path_frozen("/fake/path")

    def test_run_path_frozen_no_code_object(self):
        """Test _run_path_frozen raises error when code object not found."""
        mock_main = ModuleType("__main__")
        # No _pyi_main_co and no __spec__

        with patch.dict(sys.modules, {"__main__": mock_main}):
            with pytest.raises(RuntimeError, match="Failed to retrieve code object"):
                patch_runpy._run_path_frozen("/fake/path")

    def test_run_path_frozen_with_spec_loader(self):
        """Test _run_path_frozen using __spec__.loader.get_code."""
        mock_main = ModuleType("__main__")
        code = compile("result = 'from_spec'", "<string>", "exec")

        # Create mock spec with loader
        mock_loader = Mock()
        mock_loader.get_code.return_value = code
        mock_spec = Mock()
        mock_spec.loader = mock_loader
        mock_main.__spec__ = mock_spec

        with patch.dict(sys.modules, {"__main__": mock_main}):
            result = patch_runpy._run_path_frozen("/fake/path")

            assert result["result"] == "from_spec"
            mock_loader.get_code.assert_called_once_with("__main__")

    def test_run_path_frozen_sets_correct_globals(self):
        """Test that _run_path_frozen sets all required globals."""
        mock_main = ModuleType("__main__")
        code = compile("pass", "<string>", "exec")
        mock_main._pyi_main_co = code

        with patch.dict(sys.modules, {"__main__": mock_main}):
            result = patch_runpy._run_path_frozen("/test/path.py")

            assert result["__name__"] == "__main__"
            assert result["__file__"] == "/test/path.py"
            assert result["__package__"] is None
            assert result["__cached__"] is None
            assert result["__doc__"] is None
            assert "__builtins__" in result


class TestPatchRunpyAsDecorator:
    """Test patch_runpy used as a decorator."""

    def test_patch_runpy_decorator_frozen(self):
        """Test patch_runpy as decorator in frozen mode."""
        original_run_path = runpy.run_path

        with patch.object(sys, "frozen", True, create=True):
            @patch_runpy(strict_signature=False)
            def test_func():
                return runpy.run_path

            result = test_func()
            # Inside decorated function, should be patched
            assert result == patch_runpy._run_path_frozen

        # After function returns, should be restored
        assert runpy.run_path == original_run_path

    def test_patch_runpy_decorator_preserves_return_value(self):
        """Test that decorator preserves function return value."""
        with patch.object(sys, "frozen", True, create=True):
            @patch_runpy(strict_signature=False)
            def test_func():
                return "test_value"

            result = test_func()
            assert result == "test_value"
