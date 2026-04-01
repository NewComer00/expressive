"""
Tests for utils/i18n.py

Run with::

    pytest test_i18n.py -v
"""

import json
import pickle
import struct
import threading
import time
from types import GeneratorType
from unittest.mock import MagicMock, patch

import pytest
from lazy_string import LazyString

from utils.i18n import (
    LazyStringEncoder,
    _,
    _l,
    _lf,
    available_locales,
    detect_preferred_langs,
    expand_locale,
    init_gettext,
    iter_translated_message,
    json_dumps,
    normalize_locale,
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


def _make_mo(tmp_path, lang: str, domain: str, mapping: dict):
    """Compile a minimal .mo binary for *lang* and return its path.

    Writes ``<tmp_path>/<lang>/LC_MESSAGES/<domain>.mo`` so that
    ``gettext.translation`` can find it via the standard directory layout.

    The empty-string metadata entry (``"" -> "Content-Type: ..."```) is always
    injected so that Python's gettext parser uses UTF-8 instead of defaulting
    to ASCII, which would raise ``UnicodeDecodeError`` on any non-ASCII value.
    """
    lc_dir = tmp_path / lang / "LC_MESSAGES"
    lc_dir.mkdir(parents=True, exist_ok=True)
    mo_path = lc_dir / f"{domain}.mo"

    # Prepend the mandatory charset metadata entry.
    full_mapping = {
        "": "Content-Type: text/plain; charset=UTF-8\n",
        **mapping,
    }
    keys = list(full_mapping.keys())
    vals = [full_mapping[k] for k in keys]

    def encode_block(strings):
        offsets, data = [], b""
        for s in strings:
            enc = s.encode("utf-8")
            offsets.append((len(enc), len(data)))
            data += enc + b"\x00"
        return offsets, data

    key_offsets, key_data = encode_block(keys)
    val_offsets, val_data = encode_block(vals)

    n = len(keys)
    header_size = 28
    key_table_offset = header_size
    val_table_offset = key_table_offset + 8 * n
    key_data_offset = val_table_offset + 8 * n
    val_data_offset = key_data_offset + len(key_data)

    header = struct.pack(
        "<IIIIIII",
        0x950412DE,  # magic
        0,           # revision
        n,
        key_table_offset,
        val_table_offset,
        0, 0,        # no hash table
    )
    key_table = b"".join(
        struct.pack("<II", length, key_data_offset + offset)
        for length, offset in key_offsets
    )
    val_table = b"".join(
        struct.pack("<II", length, val_data_offset + offset)
        for length, offset in val_offsets
    )

    mo_path.write_bytes(header + key_table + val_table + key_data + val_data)
    return mo_path


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


@pytest.fixture()
def locale_dir(tmp_path):
    """A locale directory with real .mo files for en, fr, and zh_CN."""
    _make_mo(tmp_path, "en",    "messages", {"Welcome": "Welcome"})
    _make_mo(tmp_path, "fr",    "messages", {"Welcome": "Bienvenue"})
    _make_mo(tmp_path, "zh_CN", "messages", {"Welcome": "欢迎"})
    return tmp_path


# ---------------------------------------------------------------------------
# LazyString (via _l factory)
# ---------------------------------------------------------------------------

class TestNormalizeLocale:
    def test_bcp47_hyphen(self):
        assert normalize_locale("en-US") == "en_US"

    def test_posix_underscore(self):
        assert normalize_locale("zh_CN") == "zh_CN"

    def test_lowercase_region(self):
        assert normalize_locale("zh-cn") == "zh_CN"

    def test_bare_language(self):
        assert normalize_locale("fr") == "fr"

    def test_uppercase_input(self):
        assert normalize_locale("FR_FR") == "fr_FR"

    def test_three_part_truncates_to_two(self):
        """Script subtags (zh-Hans-CN) are truncated to the first two parts."""
        assert normalize_locale("zh-Hans-CN") == "zh_HANS"

    def test_empty_returns_empty(self):
        assert normalize_locale("") == ""

    def test_underscore_only_returns_empty(self):
        assert normalize_locale("_") == ""

    def test_hyphen_only_returns_empty(self):
        assert normalize_locale("-") == ""

    def test_whitespace_stripped(self):
        assert normalize_locale("  en_US  ") == "en_US"



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
# expand_locale
# ---------------------------------------------------------------------------

class TestExpandLocale:
    def test_region_form_expands_to_two(self):
        assert expand_locale("zh_CN") == ["zh_CN", "zh"]

    def test_bare_language_expands_to_one(self):
        assert expand_locale("en") == ["en"]

    def test_bcp47_input_normalized(self):
        assert expand_locale("en-US") == ["en_US", "en"]

    def test_garbage_returns_empty_list(self):
        assert expand_locale("_") == []

    def test_empty_returns_empty_list(self):
        assert expand_locale("") == []

    def test_numeric_language_returns_empty_list(self):
        assert expand_locale("123") == []


# ---------------------------------------------------------------------------
# detect_system_lang
# ---------------------------------------------------------------------------

class TestDetectPreferredLangs:
    # --- LANGUAGE env var (highest priority, all platforms) ---

    def test_language_env_takes_priority_on_linux(self):
        # LANGUAGE=fr_FR:en with LANG=de_DE — fr_FR must appear before de
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANGUAGE": "fr_FR:en", "LANG": "de_DE.UTF-8"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"
            assert "de_DE" not in result[:2]

    def test_language_env_takes_priority_on_win32(self):
        mock_windll = MagicMock()
        mock_windll.kernel32.GetUserDefaultUILanguage.return_value = 1033  # en_US
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll", mock_windll, create=True), \
             patch.dict("os.environ", {"LANGUAGE": "ja_JP"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "ja_JP"

    def test_language_env_colon_list_uses_first_entry(self):
        # zh_CN must be first, and zh (fallback) must also be present before fr
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANGUAGE": "zh_CN:fr:en"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "zh_CN"
            assert result.index("zh_CN") < result.index("fr")

    def test_language_env_expands_to_fallback_chain(self):
        # fr_FR must expand to ["fr_FR", "fr", ...]
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANGUAGE": "fr_FR"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"
            assert "fr" in result
            assert result.index("fr_FR") < result.index("fr")

    def test_language_env_strips_bcp47_region(self):
        # zh-Hans-CN normalises to zh_HANS; zh (fallback) must also appear
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANGUAGE": "zh-Hans-CN"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "zh_HANS"
            assert "zh" in result

    def test_language_env_returns_lowercase(self):
        # FR_FR normalises to fr_FR
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANGUAGE": "FR_FR"}, clear=False):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"

    def test_language_env_garbage_value_includes_fallback(self):
        # "_" has no valid language code; it must be skipped so the English
        # fallback is still included in the result.
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=(None, None)), \
             patch.dict("os.environ", {"LANGUAGE": "_"}, clear=True):
            result = detect_preferred_langs()
            assert "en_US" in result or "en" in result

    # --- LANG env var (second priority, all platforms) ---

    def test_lang_env_used_when_language_absent(self):
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANG": "ja_JP.UTF-8"}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "ja_JP"

    def test_lang_env_used_on_win32_when_language_absent(self):
        mock_windll = MagicMock()
        mock_windll.kernel32.GetUserDefaultUILanguage.return_value = 1033
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll", mock_windll, create=True), \
             patch.dict("os.environ", {"LANG": "fr_FR.UTF-8"}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"

    def test_lang_env_strips_encoding_suffix(self):
        # de_DE.UTF-8 → de_DE (encoding stripped, region preserved)
        with patch("sys.platform", "linux"), \
             patch.dict("os.environ", {"LANG": "de_DE.UTF-8"}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "de_DE"

    def test_lang_env_garbage_value_includes_fallback(self):
        # "-" normalises to an invalid tag; fallback must be present
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=(None, None)), \
             patch.dict("os.environ", {"LANG": "-"}, clear=True):
            result = detect_preferred_langs()
            assert "en_US" in result or "en" in result

    # --- OS API fallback ---

    def test_win32_os_api_used_when_no_env(self):
        mock_windll = MagicMock()
        mock_windll.kernel32.GetUserDefaultUILanguage.return_value = 1033  # en_US
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll", mock_windll, create=True), \
             patch.dict("locale.windows_locale", {1033: "en_US"}), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert "en_US" in result or "en" in result

    def test_win32_unknown_lang_id_returns_fallback(self):
        mock_windll = MagicMock()
        mock_windll.kernel32.GetUserDefaultUILanguage.return_value = 9999
        with patch("sys.platform", "win32"), \
             patch("ctypes.windll", mock_windll, create=True), \
             patch.dict("locale.windows_locale", {}, clear=True), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert "en_US" in result or "en" in result

    def test_non_win32_uses_locale_getlocale(self):
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=("fr_FR", "UTF-8")), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"
            assert "fr" in result

    def test_non_win32_returns_fallback_when_all_sources_empty(self):
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=(None, None)), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert result == ["en_US", "en"]

    def test_win32_api_skipped_when_env_vars_set(self):
        """Windows API must not be called when LANGUAGE or LANG env vars are present."""
        with patch("sys.platform", "win32"), \
             patch("utils.i18n._get_windows_preferred_langs") as mock_win_api, \
             patch.dict("os.environ", {"LANG": "fr_FR.UTF-8"}, clear=True):
            result = detect_preferred_langs()
        mock_win_api.assert_not_called()
        assert result[0] == "fr_FR"

    def test_get_windows_preferred_langs_returns_empty_on_non_windows(self):
        """_get_windows_preferred_langs must short-circuit on non-Windows platforms."""
        from utils.i18n import _get_windows_preferred_langs
        with patch("sys.platform", "linux"):
            assert _get_windows_preferred_langs() == []

    # --- normalisation ---

    def test_region_form_comes_before_bare_language(self):
        # expand_locale("zh_CN") → ["zh_CN", "zh"]; both must appear in order
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=("zh_CN", "UTF-8")), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "zh_CN"
            assert "zh" in result
            assert result.index("zh_CN") < result.index("zh")

    def test_strips_bcp47_region_from_os_api(self):
        # _get_windows_preferred_langs is only called when env vars are absent
        # and platform is win32; mock the function directly to return "zh-CN"
        with patch("sys.platform", "win32"), \
             patch("utils.i18n._get_windows_preferred_langs", return_value=["zh-CN"]), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "zh_CN"
            assert "zh" in result

    def test_returns_lowercase_from_os_api(self):
        # locale.getlocale may return uppercase region — must be normalised
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=("FR_FR", "UTF-8")), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert result[0] == "fr_FR"

    def test_never_returns_empty_list(self):
        # Even a locale that normalises to an invalid tag must not produce an
        # empty result — the English fallback must always be present.
        with patch("sys.platform", "linux"), \
             patch("locale.getlocale", return_value=("_bad", "UTF-8")), \
             patch.dict("os.environ", {}, clear=True):
            result = detect_preferred_langs()
            assert len(result) > 0
            assert "en_US" in result or "en" in result


# ---------------------------------------------------------------------------
# init_gettext
# ---------------------------------------------------------------------------

class TestInitGettext:
    def test_c_locale_installs_identity(self):
        """lang='C' must install the identity function without touching gettext."""
        init_gettext(lang="C", locale_dir="/nonexistent", domain="messages")
        assert _("Save") == "Save"
        assert _("anything") == "anything"

    def test_posix_locale_installs_identity(self):
        """lang='POSIX' must install the identity function without touching gettext."""
        init_gettext(lang="POSIX", locale_dir="/nonexistent", domain="messages")
        assert _("Save") == "Save"

    def test_c_locale_case_insensitive(self):
        """'c', 'C', and '  C  ' must all be treated as the POSIX sentinel."""
        for variant in ("c", "C", "  C  "):
            _reset_module()
            init_gettext(lang=variant, locale_dir="/nonexistent", domain="messages")
            assert _("Save") == "Save", f"Failed for lang={variant!r}"

    def test_c_locale_emits_no_warning(self):
        """lang='C' must not trigger any logger warnings."""
        with patch("utils.i18n._logger") as mock_logger:
            init_gettext(lang="C", locale_dir="/nonexistent", domain="messages")
        mock_logger.warning.assert_not_called()

    def test_c_locale_not_treated_as_list_entry(self):
        """'C' inside a list should NOT trigger the sentinel — only a bare string does."""
        # When passed as a list, ["C"] goes through expand_locale which returns []
        # (since "c" has no letters in its region part — actually "c" is valid lang).
        # The important thing: no identity short-circuit, normal gettext path taken.
        with patch("utils.i18n._logger"):  # suppress expected warning
            init_gettext(lang=["C"], locale_dir="/nonexistent", domain="messages")
        # Should still return untranslated (no .mo found), but via gettext fallback
        assert _("Save") == "Save"

    def test_warns_when_locale_dir_missing(self, tmp_path):
        """init_gettext must log a warning when locale_dir does not exist.

        Two warnings fire in this case: one for the missing directory, and a
        second because no .mo file could be matched. Assert both are present
        and that the first specifically mentions locale_dir.
        """
        missing = str(tmp_path / "nonexistent")
        with patch("utils.i18n._logger") as mock_logger:
            init_gettext(lang="fr", locale_dir=missing, domain="messages")
        assert mock_logger.warning.call_count == 2
        first_call_msg = mock_logger.warning.call_args_list[0].args[0]
        assert "locale_dir" in first_call_msg

    def test_warns_when_no_mo_file_matches(self, tmp_path):
        """init_gettext must log a warning when no .mo matches the requested locale."""
        # locale_dir exists but has no .mo files
        with patch("utils.i18n._logger") as mock_logger:
            init_gettext(lang="fr", locale_dir=str(tmp_path), domain="messages")
        mock_logger.warning.assert_called_once()
        assert "no .mo file" in mock_logger.warning.call_args.args[0]

    def test_no_warning_when_mo_found(self, locale_dir):
        """init_gettext must not warn when a valid .mo is resolved."""
        with patch("utils.i18n._logger") as mock_logger:
            init_gettext(lang="fr", locale_dir=str(locale_dir), domain="messages")
        mock_logger.warning.assert_not_called()

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

    def test_autodetects_system_lang(self, locale_dir):
        with patch("utils.i18n.detect_preferred_langs", return_value=["fr", "en"]):
            init_gettext(lang=None, locale_dir=str(locale_dir), domain="messages")
        assert _("Welcome") == "Bienvenue"

    def test_falls_back_to_en_when_detect_returns_en(self, locale_dir):
        # detect_preferred_langs() never returns None; it always returns a list
        # (either detected langs or ["en_US", "en"] as fallback)
        with patch("utils.i18n.detect_preferred_langs", return_value=["en"]):
            init_gettext(lang=None, locale_dir=str(locale_dir), domain="messages")
        assert _("Welcome") == "Welcome"

    def test_explicit_lang_bypasses_autodetect(self, locale_dir):
        with patch("utils.i18n.detect_preferred_langs") as mock_detect:
            init_gettext(lang="zh_CN", locale_dir=str(locale_dir), domain="messages")
            mock_detect.assert_not_called()
        assert _("Welcome") == "欢迎"


# ---------------------------------------------------------------------------
# _lf (lazy formatted translation)
# ---------------------------------------------------------------------------

class TestLazyFormattedTranslation:
    def test_returns_lazy_string(self):
        ls = _lf("Hello, {name}!", name=lambda: "World")
        assert isinstance(ls, LazyString)

    def test_str_format_before_init(self):
        ls = _lf("Hello, {name}!", name=lambda: "World")
        assert str(ls) == "Hello, World!"

    def test_str_format_after_init(self, fr_translator):
        ls = _lf("Hello, {name}!", name=lambda: "World")
        assert str(ls) == "Bonjour, World!"

    def test_percent_formatting_before_init(self):
        ls = _lf("Hello, %s!", lambda: "World")
        assert str(ls) == "Hello, World!"

    def test_percent_formatting_after_init(self, fr_translator):
        ls = _lf("Hello, %s!", lambda: "World")
        assert str(ls) == "Bonjour, World!"

    def test_lazy_format_args(self, fr_translator):
        ls = _lf("Hello, {name}!", name=lambda: "Alice")
        assert str(ls) == "Bonjour, Alice!"

    def test_lazy_percent_args(self, fr_translator):
        ls = _lf("Hello, %s!", lambda: "Alice")
        assert str(ls) == "Bonjour, Alice!"

    def test_resolves_after_init(self):
        ls = _lf("Hello, {name}!", name=lambda: "World")
        _install_translator({"Hello, {name}!": "Bonjour, {name}!"})
        assert str(ls) == "Bonjour, World!"

    def test_reflects_locale_hot_swap(self):
        ls = _lf("Hello, {name}!", name=lambda: "World")
        _install_translator({"Hello, {name}!": "Bonjour, {name}!"})
        assert str(ls) == "Bonjour, World!"
        _install_translator({"Hello, {name}!": "Hola, {name}!"})
        assert str(ls) == "Hola, World!"

    def test_multiple_kwargs(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Hello, {name}, you are {age}!": "Bonjour, {name}, vous avez {age} ans!",
                "Hello": "Bonjour",
                "Save": "Enregistrer",
                "Cancel": "Annuler",
                "Hello, %s!": "Bonjour, %s!",
                "Hello, {name}!": "Bonjour, {name}!",
                "item": "élément",
                "Test message": "Message de test",
                "Nested": "Imbriqué",
                "Item1": "Élément1",
                "Item2": "Élément2",
            }.get(msg, msg)
        ls = _lf("Hello, {name}, you are {age}!", name=lambda: "Alice", age=lambda: "25")
        assert str(ls) == "Bonjour, Alice, vous avez 25 ans!"

    def test_non_callable_kwarg_int(self):
        ls = _lf("Error: {error_code}", error_code=404)
        assert str(ls) == "Error: 404"

    def test_non_callable_kwarg_int_translated(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Error: {error_code}": "Erreur: {error_code}",
            }.get(msg, msg)
        ls = _lf("Error: {error_code}", error_code=404)
        assert str(ls) == "Erreur: 404"

    def test_no_args_returns_plain_lazy_translation(self):
        """_lf with no args must behave identically to _l."""
        ls = _lf("Save")
        assert isinstance(ls, LazyString)
        assert str(ls) == "Save"

    def test_no_args_resolves_after_init(self, fr_translator):
        ls = _lf("Save")
        assert str(ls) == "Enregistrer"

    def test_multiple_positional_args(self):
        """_lf must forward all positional args, not just args[0]."""
        ls = _lf("Found %d error(s) in %s", 3, "main.py")
        assert str(ls) == "Found 3 error(s) in main.py"

    def test_multiple_positional_args_translated(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Found %d error(s) in %s": "Trouvé %d erreur(s) dans %s",
            }.get(msg, msg)
        ls = _lf("Found %d error(s) in %s", 3, "main.py")
        assert str(ls) == "Trouvé 3 erreur(s) dans main.py"

    def test_non_callable_percent_arg(self):
        ls = _lf("Warning: %s", "Low battery")
        assert str(ls) == "Warning: Low battery"

    def test_non_callable_percent_arg_translated(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Warning: %s": "Avertissement: %s",
            }.get(msg, msg)
        ls = _lf("Warning: %s", "Low battery")
        assert str(ls) == "Avertissement: Low battery"

    def test_lazy_callable_arg_in_percent(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Warning: %s": "Avertissement: %s",
            }.get(msg, msg)
        ls = _lf("Warning: %s", lambda: "Low battery")
        assert str(ls) == "Avertissement: Low battery"

    def test_mixed_callable_and_non_callable(self, fr_translator):
        import utils.i18n
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "Item {name} costs ${price}": "Article {name} coûte ${price}",
            }.get(msg, msg)
        ls = _lf("Item {name} costs ${price}", name=lambda: "Widget", price=9.99)
        assert str(ls) == "Article Widget coûte $9.99"

    def test_complex_lazy_content(self):
        backend_choices = {
            "swift-f0": "fast, CPU-based (ONNX Runtime)",
            "crepe": "classic but slow, CPU & NVIDIA GPU (TensorFlow)",
        }
        ls = _lf(
            "**F0 detection backend** ...options:\n\n%s\n\n",
            lambda: "\n".join([f"- `{k}`: {v}" for k, v in backend_choices.items()])
        )
        expected_content = "\n".join([
            "- `swift-f0`: fast, CPU-based (ONNX Runtime)",
            "- `crepe`: classic but slow, CPU & NVIDIA GPU (TensorFlow)",
        ])
        assert str(ls) == f"**F0 detection backend** ...options:\n\n{expected_content}\n\n"

    def test_complex_lazy_content_translated(self, fr_translator):
        import utils.i18n
        backend_choices = {
            "swift-f0": "fast, CPU-based (ONNX Runtime)",
            "crepe": "classic but slow, CPU & NVIDIA GPU (TensorFlow)",
        }
        with utils.i18n._lock:
            utils.i18n._current_gettext = lambda msg: {
                "**F0 detection backend** ...options:\n\n%s\n\n": "**Backend F0** ...options:\n\n%s\n\n",
            }.get(msg, msg)
        ls = _lf(
            "**F0 detection backend** ...options:\n\n%s\n\n",
            lambda: "\n".join([f"- `{k}`: {v}" for k, v in backend_choices.items()])
        )
        expected_content = "\n".join([
            "- `swift-f0`: fast, CPU-based (ONNX Runtime)",
            "- `crepe`: classic but slow, CPU & NVIDIA GPU (TensorFlow)",
        ])
        assert str(ls) == f"**Backend F0** ...options:\n\n{expected_content}\n\n"


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
# _lt alias
# ---------------------------------------------------------------------------

class TestLtAlias:
    def test_lt_is_alias_for_iter_translated_message(self):
        from utils.i18n import _lt
        assert _lt is iter_translated_message


# ---------------------------------------------------------------------------
# available_locales
# ---------------------------------------------------------------------------

class TestAvailableLocales:
    def test_empty_when_no_locales(self, tmp_path):
        assert available_locales(str(tmp_path), "messages") == []

    def test_missing_locale_dir_returns_empty(self, tmp_path):
        assert available_locales(str(tmp_path / "nonexistent"), "messages") == []

    def test_discovers_single_locale(self, tmp_path):
        mo = tmp_path / "en" / "LC_MESSAGES" / "messages.mo"
        mo.parent.mkdir(parents=True)
        mo.touch()
        assert available_locales(str(tmp_path), "messages") == ["en"]

    def test_discovers_multiple_locales(self, tmp_path):
        for lang in ("en", "fr", "zh_CN"):
            mo = tmp_path / lang / "LC_MESSAGES" / "messages.mo"
            mo.parent.mkdir(parents=True)
            mo.touch()
        assert available_locales(str(tmp_path), "messages") == ["en", "fr", "zh_CN"]

    def test_result_is_sorted(self, tmp_path):
        for lang in ("zh_CN", "en", "fr"):
            mo = tmp_path / lang / "LC_MESSAGES" / "messages.mo"
            mo.parent.mkdir(parents=True)
            mo.touch()
        assert available_locales(str(tmp_path), "messages") == ["en", "fr", "zh_CN"]

    def test_ignores_dirs_without_mo_file(self, tmp_path):
        lc = tmp_path / "de" / "LC_MESSAGES"
        lc.mkdir(parents=True)
        (lc / "messages.po").touch()  # .po only, no .mo
        assert available_locales(str(tmp_path), "messages") == []

    def test_ignores_wrong_domain(self, tmp_path):
        mo = tmp_path / "en" / "LC_MESSAGES" / "other_domain.mo"
        mo.parent.mkdir(parents=True)
        mo.touch()
        assert available_locales(str(tmp_path), "messages") == []

    def test_permission_error_returns_empty(self, tmp_path):
        """available_locales must return [] when the directory is unreadable."""
        with patch("os.scandir", side_effect=PermissionError("denied")):
            assert available_locales(str(tmp_path), "messages") == []

    def test_ignores_files_at_top_level(self, tmp_path):
        (tmp_path / "not_a_dir").touch()
        assert available_locales(str(tmp_path), "messages") == []


# ---------------------------------------------------------------------------
# iter_translated_message
# ---------------------------------------------------------------------------

class TestIterTranslatedMessage:
    def test_yields_all_discovered_locales(self, locale_dir):
        results = list(iter_translated_message("Welcome", str(locale_dir), "messages"))
        langs = [lang for lang, _ in results]
        assert langs == ["en", "fr", "zh_CN"]

    def test_yields_correct_translations(self, locale_dir):
        results = dict(iter_translated_message("Welcome", str(locale_dir), "messages"))
        assert results["en"]    == "Welcome"
        assert results["fr"]    == "Bienvenue"
        assert results["zh_CN"] == "欢迎"

    def test_explicit_langs_overrides_discovery(self, locale_dir):
        results = list(iter_translated_message(
            "Welcome", str(locale_dir), "messages", langs=["fr", "en"]
        ))
        assert [lang for lang, _ in results] == ["fr", "en"]

    def test_explicit_langs_controls_order(self, locale_dir):
        results = list(iter_translated_message(
            "Welcome", str(locale_dir), "messages", langs=["zh_CN", "fr"]
        ))
        assert results[0] == ("zh_CN", "欢迎")
        assert results[1] == ("fr",    "Bienvenue")

    def test_fallback_for_missing_translation(self, locale_dir):
        # "Goodbye" has no translation → falls back to the key
        results = dict(iter_translated_message("Goodbye", str(locale_dir), "messages"))
        assert all(text == "Goodbye" for text in results.values())

    def test_empty_locale_dir_yields_nothing(self, tmp_path):
        results = list(iter_translated_message("Welcome", str(tmp_path), "messages"))
        assert results == []

    def test_does_not_mutate_active_gettext(self, locale_dir):
        _install_translator({"Welcome": "Howdy"})
        list(iter_translated_message("Welcome", str(locale_dir), "messages"))
        assert _("Welcome") == "Howdy"

    def test_is_a_generator(self, locale_dir):
        result = iter_translated_message("Welcome", str(locale_dir), "messages")
        assert isinstance(result, GeneratorType)

    def test_explicit_single_lang(self, locale_dir):
        results = list(iter_translated_message(
            "Welcome", str(locale_dir), "messages", langs=["fr"]
        ))
        assert results == [("fr", "Bienvenue")]

    def test_empty_langs_list_yields_nothing(self, locale_dir):
        results = list(iter_translated_message(
            "Welcome", str(locale_dir), "messages", langs=[]
        ))
        assert results == []


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

    def test_iter_translated_message_safe_during_locale_swap(self, locale_dir):
        """iter_translated_message must not be disrupted by a concurrent hot-swap."""
        errors = []

        def iterate():
            try:
                for _lang, text in iter_translated_message(
                    "Welcome", str(locale_dir), "messages"
                ):
                    assert isinstance(text, str)
            except Exception as exc:
                errors.append(exc)

        def swapper():
            for mapping in [{"Welcome": "Howdy"}, {}] * 20:
                _install_translator(mapping)
                time.sleep(0)

        threads = [threading.Thread(target=iterate) for _ in range(4)]
        threads.append(threading.Thread(target=swapper))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
