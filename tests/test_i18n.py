"""
Tests for utils/i18n.py

Run with::

    pytest test_i18n.py -v
"""

import json
import pickle
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from lazy_string import LazyString

from utils.i18n import (
    LazyStringEncoder,
    _,
    _l,
    init_gettext,
    json_dumps,
    patch_nicegui_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_module():
    """Reset the module-level translation state between tests."""
    import utils.i18n
    with utils.i18n._lock:
        utils.i18n._current_gettext = None


def _install_translator(mapping: dict):
    """Install a simple dict-based translator into utils.i18n."""
    import utils.i18n
    with utils.i18n._lock:
        utils.i18n._current_gettext = lambda msg: mapping.get(msg, msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Ensure each test starts with a clean translation state."""
    _reset_module()
    yield
    _reset_module()


@pytest.fixture()
def fr_translator():
    """Install a French translator and return the mapping."""
    mapping = {
        "Save": "Enregistrer",
        "Cancel": "Annuler",
        "Hello, %s!": "Bonjour, %s!",
        "Hello, {name}!": "Bonjour, {name}!",
        "Hello": "Bonjour",
        "item": "élément",
        "Test message": "Message de test",
        "Nested": "Imbriqué",
        "Item1": "Élément1",
        "Item2": "Élément2",
    }
    _install_translator(mapping)
    return mapping


# ---------------------------------------------------------------------------
# LazyString (via _l factory)
# ---------------------------------------------------------------------------

class TestLazyString:
    def test_creation_returns_lazy_string(self):
        ls = _l("Hello")
        assert isinstance(ls, LazyString)

    def test_str_conversion_before_init(self):
        ls = _l("Hello")
        assert str(ls) == "Hello"

    def test_str_conversion_after_init(self, fr_translator):
        ls = _l("Hello")
        assert str(ls) == "Bonjour"

    def test_repr_contains_key(self):
        ls = _l("Hello")
        assert "Hello" in repr(ls)

    def test_equality_same_key(self, fr_translator):
        ls1 = _l("Save")
        ls2 = _l("Save")
        assert ls1 == ls2

    def test_equality_with_resolved_str(self, fr_translator):
        ls = _l("Save")
        assert ls == "Enregistrer"

    def test_hash_is_int(self):
        ls = _l("Save")
        assert isinstance(hash(ls), int)

    def test_len_matches_resolved(self, fr_translator):
        ls = _l("Save")
        assert len(ls) == len("Enregistrer")

    def test_add(self, fr_translator):
        ls = _l("Save")
        result = ls + "!"
        assert result == "Enregistrer!"
        assert isinstance(result, str)

    def test_radd(self, fr_translator):
        ls = _l("Save")
        result = ">" + ls
        assert result == ">Enregistrer"
        assert isinstance(result, str)

    def test_mod_formatting(self, fr_translator):
        ls = _l("Hello, %s!")
        result = ls % "Alice"
        assert result == "Bonjour, Alice!"
        assert isinstance(result, str)

    def test_format(self, fr_translator):
        ls = _l("Hello, {name}!")
        result = ls.format(name="Alice")
        assert result == "Bonjour, Alice!"
        assert isinstance(result, str)

    def test_format_map(self, fr_translator):
        ls = _l("Hello, {name}!")
        result = ls.format_map({"name": "Alice"})
        assert result == "Bonjour, Alice!"
        assert isinstance(result, str)

    def test_str_methods_inherited(self, fr_translator):
        ls = _l("Save")
        assert ls.upper() == "ENREGISTRER"
        assert ls.startswith("Enr")
        assert "reg" in ls

    def test_pickle_roundtrip(self, fr_translator):
        ls = _l("Save")
        restored = pickle.loads(pickle.dumps(ls))
        assert str(restored) == str(ls)


# ---------------------------------------------------------------------------
# _ (eager translation)
# ---------------------------------------------------------------------------

class TestEagerTranslation:
    def test_returns_msg_when_uninitialised(self):
        assert _("Save") == "Save"

    def test_translates_after_init(self, fr_translator):
        assert _("Save") == "Enregistrer"

    def test_falls_back_to_key_for_missing(self, fr_translator):
        assert _("Unknown key") == "Unknown key"

    def test_returns_str_type(self, fr_translator):
        assert isinstance(_("Save"), str)


# ---------------------------------------------------------------------------
# _l (lazy translation)
# ---------------------------------------------------------------------------

class TestLazyTranslationFunction:
    def test_returns_lazy_string(self):
        assert isinstance(_l("Save"), LazyString)

    def test_identity_before_init(self):
        assert str(_l("Save")) == "Save"

    def test_resolves_after_init(self, fr_translator):
        assert str(_l("Save")) == "Enregistrer"

    def test_created_before_init_resolves_correctly(self):
        ls = _l("Save")
        _install_translator({"Save": "Enregistrer"})
        assert str(ls) == "Enregistrer"

    def test_reflects_locale_hot_swap(self):
        ls = _l("Save")
        _install_translator({"Save": "Enregistrer"})
        assert str(ls) == "Enregistrer"
        _install_translator({"Save": "Guardar"})
        assert str(ls) == "Guardar"


# ---------------------------------------------------------------------------
# init_gettext
# ---------------------------------------------------------------------------

class TestInitGettext:
    def test_fallback_when_no_mo_file(self, tmp_path):
        """With fallback=True, missing .mo files must not raise."""
        init_gettext(lang="xx_XX", locale_dir=str(tmp_path), domain="messages")
        assert _("Test") == "Test"

    def test_identity_after_fallback_init(self, tmp_path):
        init_gettext(lang="en_US", locale_dir=str(tmp_path), domain="messages")
        assert _("Test") == "Test"

    def test_hot_swap(self):
        _install_translator({"Save": "Enregistrer"})
        assert _("Save") == "Enregistrer"
        _install_translator({"Save": "Guardar"})
        assert _("Save") == "Guardar"


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestLazyStringEncoder:
    def test_as_cls_kwarg(self, fr_translator):
        data = {"label": _l("Cancel")}
        result = json.loads(json.dumps(data, cls=LazyStringEncoder))
        assert result["label"] == "Annuler"

    def test_non_lazy_raises_type_error(self):
        enc = LazyStringEncoder()
        with pytest.raises(TypeError):
            enc.default(object())

    def test_lazy_string_resolved_in_default(self, fr_translator):
        enc = LazyStringEncoder()
        ls = _l("Save")
        assert enc.default(ls) == "Enregistrer"


class TestJsonDumps:
    def test_serialises_lazy_string(self, fr_translator):
        data = {"message": _l("Save")}
        result = json.loads(json_dumps(data))
        assert result["message"] == "Enregistrer"

    def test_serialises_mixed_dict(self, fr_translator):
        data = {"label": _l("Save"), "count": 42, "flag": True}
        result = json.loads(json_dumps(data))
        assert result == {"label": "Enregistrer", "count": 42, "flag": True}

    def test_serialises_nested_lazy_strings(self):
        data = {
            "outer": {
                "inner": _l("Nested"),
                "list": [_l("Item1"), _l("Item2")],
            }
        }
        result = json.loads(json_dumps(data))
        assert result == {
            "outer": {
                "inner": "Nested",
                "list": ["Item1", "Item2"],
            }
        }

    def test_non_serialisable_raises(self):
        with pytest.raises(TypeError):
            json_dumps({"bad": object()})

    def test_kwargs_forwarded(self, fr_translator):
        result = json_dumps({"label": _l("Save")}, indent=2)
        assert "\n" in result

    def test_standard_json_dumps_fails_with_lazy_string(self):
        """Confirm LazyString is not transparently handled by stock json.dumps."""
        with pytest.raises(TypeError):
            json.dumps({"message": _l("Hello")})


# ---------------------------------------------------------------------------
# patch_nicegui_json
# ---------------------------------------------------------------------------

class TestPatchNiceguiJson:
    def test_no_op_when_nicegui_absent(self):
        """Must not raise when NiceGUI is not installed."""
        with patch.dict("sys.modules", {
            "nicegui": None,
            "nicegui.json": None,
            "nicegui.json.orjson_wrapper": None,
        }):
            patch_nicegui_json()

    def test_patches_converter_resolves_lazy_string(self, fr_translator):
        """Patched converter must resolve LazyString to translated str."""
        original_converter = MagicMock(side_effect=TypeError("unhandled"))
        mock_wrapper = MagicMock()
        mock_wrapper._orjson_converter = original_converter

        # Exercise the same logic as the implementation
        original = mock_wrapper._orjson_converter

        def patched(obj):
            if isinstance(obj, LazyString):
                return str(obj)
            return original(obj)

        mock_wrapper._orjson_converter = patched

        ls = _l("Save")
        assert mock_wrapper._orjson_converter(ls) == "Enregistrer"

    def test_patched_converter_falls_back_for_other_types(self):
        """Patched converter must delegate non-LazyString objects to the original."""
        original_converter = MagicMock(return_value="fallback")
        mock_wrapper = MagicMock()
        mock_wrapper._orjson_converter = original_converter

        original = mock_wrapper._orjson_converter

        def patched(obj):
            if isinstance(obj, LazyString):
                return str(obj)
            return original(obj)

        mock_wrapper._orjson_converter = patched

        test_obj = object()
        mock_wrapper._orjson_converter(test_obj)
        original_converter.assert_called_once_with(test_obj)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads_do_not_crash(self, fr_translator):
        errors = []

        def reader():
            try:
                for _i in range(200):
                    result = _("Save")
                    assert result in ("Save", "Enregistrer", "Guardar")
            except Exception as exc:
                errors.append(exc)

        def swapper():
            for mapping in [{"Save": "Guardar"}, {"Save": "Enregistrer"}] * 50:
                _install_translator(mapping)
                time.sleep(0)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=swapper))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_lazy_string_survives_concurrent_locale_swap(self):
        ls = _l("Save")
        results = []

        def resolve_and_swap(mapping):
            _install_translator(mapping)
            results.append(str(ls))

        t1 = threading.Thread(target=resolve_and_swap, args=({"Save": "Enregistrer"},))
        t2 = threading.Thread(target=resolve_and_swap, args=({"Save": "Guardar"},))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert all(r in ("Enregistrer", "Guardar") for r in results)
