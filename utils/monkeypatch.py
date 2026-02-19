import sys
import inspect
from types import ModuleType
from typing import Callable, Any
from contextlib import ContextDecorator

import runpy


def ensure_same_signature(
    original: Callable[..., Any],
    replacement: Callable[..., Any],
) -> None:
    """
    Ensure that two callables have identical signatures.

    Raises RuntimeError if they differ.
    """
    original_sig = inspect.signature(original)
    replacement_sig = inspect.signature(replacement)

    if original_sig != replacement_sig:
        raise RuntimeError(
            "Signature mismatch.\n"
            f"Original:    {original_sig}\n"
            f"Replacement: {replacement_sig}"
        )


class patch_runpy(ContextDecorator):
    """
    Context manager / decorator that temporarily patches runpy.run_path
    in frozen environments only.
    """

    def __init__(self, strict_signature: bool = True):
        self.strict_signature = strict_signature

    def __enter__(self):
        # Only patch in frozen mode
        if not getattr(sys, "frozen", False):
            self._patched = False
            return self

        if self.strict_signature:
            ensure_same_signature(
                runpy.run_path,
                self._run_path_frozen,
            )

        self._previous = runpy.run_path
        runpy.run_path = self._run_path_frozen
        self._patched = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_patched", False):
            runpy.run_path = self._previous
        return False

    @staticmethod
    def _run_path_frozen(path_name, init_globals=None, run_name=None):
        """
        Frozen-environment replacement for runpy.run_path.
        """

        main_mod: ModuleType | None = sys.modules.get("__main__")
        if not main_mod:
            raise RuntimeError("Cannot locate __main__ module in frozen app")

        code = None

        # Try PyInstaller's _pyi_main_co
        if hasattr(main_mod, "_pyi_main_co"):
            code = main_mod._pyi_main_co

        # Try standard __spec__.loader.get_code
        elif getattr(main_mod, "__spec__", None):
            loader = main_mod.__spec__.loader
            if loader and hasattr(loader, "get_code"):
                code = loader.get_code("__main__")

        if code is None:
            raise RuntimeError("Failed to retrieve code object for __main__")

        globals_dict: dict[str, Any] = init_globals.copy() if init_globals else {}

        globals_dict.update(
            {
                "__name__": run_name or "__main__",
                "__file__": path_name,
                "__package__": None,
                "__cached__": None,
                "__doc__": None,
                "__builtins__": __builtins__,
            }
        )

        exec(code, globals_dict)
        return globals_dict


if __name__ == "__main__":
    print("Calling run_path with patch in simulated non-frozen mode (context manager):")
    sys.frozen = False
    with patch_runpy():
        print(runpy.run_path.__doc__)

    print("Calling run_path with patch in simulated frozen mode (decorator):")
    sys.frozen = True
    @patch_runpy()
    def test_decorator():
        print(runpy.run_path.__doc__)

    test_decorator()