"""
Lightweight i18n utilities for non-framework Python applications.

Usage
-----
Initialize once at startup::

    from utils.i18n import init_gettext, _, _l

    init_gettext(lang="zh_CN", locale_dir="locales", domain="messages")

Use ``_()`` for runtime translation and ``_l()`` for module/class-level
string constants that are defined before ``init_gettext()`` is called::

    # module-level constant — evaluated lazily
    ERROR_LABEL = _l("Error")

    def greet(name: str) -> str:
        # runtime — evaluated immediately
        return _("Hello, {name}!").format(name=name)

Dependencies
------------
Requires the ``lazy-string`` package::

    pip install lazy-string

Thread safety
-------------
All access to the active translation function is protected by a module-level
``threading.Lock``. ``init_gettext()`` may be called from any thread to hot-
swap the active locale; in-flight translations will complete with the old
locale before the switch takes effect.
"""

import logging
import os
import sys
import json
import locale
import gettext
import threading
from typing import Callable, Iterator, Optional

from lazy_string import LazyString


_logger = logging.getLogger(__name__)
_current_gettext: Optional[Callable[[str], str]] = None
_lock = threading.Lock()

# POSIX locales that explicitly mean "no translation / raw bytes".
# Passing any of these to init_gettext installs the identity function directly
# rather than searching for a .mo file that will never exist.
_UNTRANSLATED_LOCALES: frozenset[str] = frozenset({"c", "posix"})


def _get_windows_preferred_langs() -> list[str]:
    if sys.platform != "win32":
        # Safety guard: ctypes.windll is only available on Windows.
        return []

    import ctypes
    from ctypes import wintypes

    MUI_LANGUAGE_NAME = 0x8  # return locale names like "en-US"

    kernel32 = ctypes.windll.kernel32

    GetUserPreferredUILanguages = kernel32.GetUserPreferredUILanguages
    GetUserPreferredUILanguages.argtypes = [
        wintypes.DWORD,                     # dwFlags
        ctypes.POINTER(wintypes.ULONG),     # pulNumLanguages
        wintypes.LPWSTR,                    # pwszLanguagesBuffer
        ctypes.POINTER(wintypes.ULONG),     # pcchLanguagesBuffer
    ]
    GetUserPreferredUILanguages.restype = wintypes.BOOL

    num_langs = wintypes.ULONG()
    buf_size = wintypes.ULONG()

    # First call: get required buffer size
    if not GetUserPreferredUILanguages(
        MUI_LANGUAGE_NAME,
        ctypes.byref(num_langs),
        None,
        ctypes.byref(buf_size),
    ):
        return []

    # Allocate buffer
    buffer = ctypes.create_unicode_buffer(buf_size.value)

    # Second call: get actual data
    if not GetUserPreferredUILanguages(
        MUI_LANGUAGE_NAME,
        ctypes.byref(num_langs),
        buffer,
        ctypes.byref(buf_size),
    ):
        return []

    # Buffer is a double-null-terminated string list
    raw = buffer[:buf_size.value]
    langs = raw.rstrip("\x00").split("\x00")

    return langs


def normalize_locale(tag: str) -> str:
    """Normalize a locale string to gettext-friendly POSIX format.

    Converts BCP-47 style (``en-US``) and loose inputs into POSIX style
    (``en_US``), which is required by ``gettext`` directory layout.

    Rules:
        - ``-`` is converted to ``_``
        - language code is lowercased
        - region code (if present) is uppercased
        - empty input or input with no valid language code returns ``""``
        - only the first two ``_``-separated parts are used; script subtags and
          other BCP-47 extensions are truncated (e.g. ``"zh_Hans_CN"`` →
          ``"zh_HANS"``). This is intentional: POSIX gettext only supports
          ``lang_REGION`` directory names.

    Examples:
        >>> normalize_locale("en-US")
        'en_US'
        >>> normalize_locale("zh-cn")
        'zh_CN'
        >>> normalize_locale("fr")
        'fr'
        >>> normalize_locale("zh-Hans-CN")
        'zh_HANS'
        >>> normalize_locale("_")
        ''
        >>> normalize_locale("")
        ''
    """
    tag = tag.strip().replace("-", "_")

    if not tag:
        return ""

    parts = tag.split("_")
    lang = parts[0].lower()

    # Reject entries whose language part contains no letters (e.g. "_", "-")
    if not lang or not lang.isalpha():
        return ""

    if len(parts) == 1:
        return lang

    region = parts[1].upper()
    # Accept region only if it looks like a valid code (letters/digits only)
    if region and region.isalnum():
        return f"{lang}_{region}"

    return lang


def expand_locale(tag: str) -> list[str]:
    """Expand a locale into a gettext fallback chain.

    Ensures both region-specific and language-only variants are tried.
    Returns an empty list if *tag* normalizes to an invalid locale.

    Examples:
        >>> expand_locale("zh_CN")
        ['zh_CN', 'zh']
        >>> expand_locale("en")
        ['en']
        >>> expand_locale("_")
        []
    """
    tag = normalize_locale(tag)
    if not tag:
        return []

    parts = tag.split("_")

    if len(parts) == 1:
        return [parts[0]]

    lang, region = parts[0], parts[1]
    return [f"{lang}_{region}", lang]


def detect_preferred_langs() -> list[str]:
    """Return a prioritized list of preferred UI languages.

    Sources (in order of priority):

    1. ``LANGUAGE`` environment variable (GNU gettext style, colon-separated)
    2. ``LANG`` environment variable (POSIX locale)
    3. OS API:
        - Windows: ``GetUserPreferredUILanguages``
        - Others: ``locale.getlocale()``

    The result is:
        - normalized to POSIX format (``en_US``)
        - expanded to include language fallbacks (``["en_US", "en"]``)
        - deduplicated while preserving priority order

    Always returns at least ``["en_US", "en"]``.

    Examples:
        >>> detect_preferred_langs()
        ['zh_CN', 'zh', 'en_US', 'en']
    """
    fallback = ["en_US", "en"]

    raw: list[str] = []

    # 1. LANGUAGE (priority list)
    if language_env := os.environ.get("LANGUAGE"):
        raw.extend(x for x in language_env.split(":") if x.strip())

    # 2. LANG (single fallback)
    if lang_env := os.environ.get("LANG"):
        raw.append(lang_env.split(".")[0])

    # 3. Windows API — only consulted when no env vars provided a raw list.
    # This matches GNU gettext priority: LANGUAGE/LANG always win on Windows too.
    if not raw and sys.platform == "win32":
        raw = _get_windows_preferred_langs()

    # 4. POSIX fallback
    if not raw:
        loc = locale.getlocale()[0]
        if loc:
            raw.append(loc)

    # Normalize + expand + deduplicate
    seen = set()
    result: list[str] = []

    for lang in raw:
        for expanded in expand_locale(lang):
            if expanded not in seen:
                seen.add(expanded)
                result.append(expanded)

    # If every raw entry normalized to an empty string (shouldn't happen in
    # practice), result will be empty and we fall back to English silently.
    return result or fallback


def init_gettext(
    lang: str | list[str] | None,
    locale_dir: str,
    domain: str,
) -> None:
    """Initialize or hot-swap the active ``gettext`` translation.

    This function is thread-safe and may be called multiple times to switch
    locales at runtime.

    Language resolution:

        - ``None``:
            Auto-detect via :func:`detect_preferred_langs`

        - ``str``:
            A single locale (e.g. ``"en-US"`` or ``"zh_CN"``),
            normalized and expanded to a fallback chain

        - ``list[str]``:
            A priority list of locales, each normalized and expanded

    All inputs:
        - accept both ``-`` and ``_`` separators
        - are normalized to POSIX format (``en_US``)
        - are expanded into fallback chains (``["en_US", "en"]``)
        - are deduplicated while preserving order

    Args:
        lang: Language preference (string, list, or None).
            The special POSIX values ``"C"`` and ``"POSIX"`` (case-insensitive)
            install the identity function directly — no .mo lookup is performed
            and no warning is emitted.
        locale_dir: Path to ``gettext`` locale directory
        domain: Message catalog domain (e.g. ``"messages"``)

    Example:
        >>> init_gettext("en-US", ...)
        # internally → ["en_US", "en"]

        >>> init_gettext(["zh-cn", "en-us"], ...)
        # internally → ["zh_CN", "zh", "en_US", "en"]

        >>> init_gettext(None, ...)
        # auto-detected system preference list

        >>> init_gettext("C", ...)
        # installs identity function — all strings returned untranslated

    Notes:
        - ``gettext`` handles fallback resolution in order
        - Missing locales are silently ignored (fallback=True)
    """
    global _current_gettext
    # Short-circuit for the POSIX "no translation" sentinels.
    if isinstance(lang, str) and lang.strip().lower() in _UNTRANSLATED_LOCALES:
        with _lock:
            _current_gettext = lambda msg: msg
        return

    # Resolve + normalize + expand
    if lang is None:
        langs = detect_preferred_langs()
    elif isinstance(lang, str):
        langs = expand_locale(lang)
    else:
        langs = []
        for x in lang:
            langs.extend(expand_locale(x))

    # Deduplicate while preserving order
    seen = set()
    langs = [lg for lg in langs if not (lg in seen or seen.add(lg))]

    if not os.path.isdir(locale_dir):
        _logger.warning(
            "init_gettext: locale_dir %r does not exist — all translations will "
            "fall back to the source language.",
            locale_dir,
        )

    translation = gettext.translation(
        domain,
        localedir=locale_dir,
        languages=langs,
        fallback=True,
    )

    # NullTranslations is used when no .mo file matched any requested locale.
    # Warn so misconfigured locale paths are caught early.
    if type(translation) is gettext.NullTranslations:
        _logger.warning(
            "init_gettext: no .mo file found for langs=%r in %r (domain=%r) — "
            "falling back to source language.",
            langs,
            locale_dir,
            domain,
        )

    with _lock:
        _current_gettext = translation.gettext


def _(msg: str) -> str:
    """Translate *msg* immediately using the active locale.

    This is the primary translation function and should be used for the vast
    majority of strings — anywhere the call site is inside a function or method
    that executes *after* :func:`init_gettext` has been called.

    If :func:`init_gettext` has not yet been called, *msg* is returned
    unchanged (identity fallback).

    Args:
        msg: The source-language string (translation key).

    Returns:
        The translated string, or *msg* itself if no translation is available.

    Example::

        print(_("Hello, world!"))
    """
    with _lock:
        return _current_gettext(msg) if _current_gettext else msg


def _l(msg: str) -> LazyString:
    """Return a :class:`~lazy_string.LazyString` that is translated on first use.

    Use this function **only** for strings that are defined at module or class
    level — i.e. before :func:`init_gettext` has been called.  For all other
    strings prefer the eagerly evaluated :func:`_`.

    The returned ``LazyString`` delegates to :func:`_` each time its value is
    needed, so the correct locale is always applied even after a hot-swap via
    :func:`init_gettext`.

    Args:
        msg: The source-language string (translation key).

    Returns:
        A :class:`~lazy_string.LazyString` proxy wrapping *msg*.

    Example::

        # Evaluated at import time — locale not yet set, so use _l
        BUTTON_LABEL = _l("Save")

        class MyModel:
            verbose_name = _l("item")
    """
    return LazyString(_, msg)


def _lf(msg: str, *args, **kwargs) -> LazyString:
    """Like :func:`_l`, but supports formatting arguments.

    Args:
        msg: The source-language string (translation key), with optional format
            placeholders, e.g. ``"Hello, {name}!"``.
        *args: Positional arguments for old-style ``%`` formatting.  If provided,
            the translated string is formatted using ``msg % args`` (supports
            multiple positional values, e.g. ``"%s and %s"``).  Each argument
            may be a zero-argument callable, in which case it is called lazily
            at format time.
        **kwargs: Keyword arguments for new-style ``str.format`` formatting.  If
            provided, the string is formatted using ``msg.format(**kwargs)``.
            If both *args* and *kwargs* are provided, *args* takes precedence.
            Each value may also be a zero-argument callable for lazy evaluation.

    Returns:
        A :class:`~lazy_string.LazyString` proxy that formats the translated string
        on first use.

    Example::
        ERROR_LABEL = _lf("Error: {error_code}", error_code=404)
        WARNING_LABEL = _lf("Warning: %s", "Low battery")
        MULTI_LABEL = _lf("Found %d error(s) in %s", 3, "main.py")
        # Formatter arguments can also be lazily evaluated:
        help = _lf(
            "**F0 detection backend** ...options:\n\n%s\n\n",
            lambda: "\n".join([f"- `{k}`: {v}" for k, v in PitdLoader.backend_choices.items()])
        )
    """
    if args:
        return LazyString(
            lambda: _(msg) % tuple(a() if callable(a) else a for a in args)
        )
    if kwargs:
        return LazyString(
            lambda: _(msg).format(**{k: (v() if callable(v) else v) for k, v in kwargs.items()})
        )
    # No formatting arguments — plain lazy translation
    return LazyString(_, msg)


class LazyStringEncoder(json.JSONEncoder):
    """JSON encoder that transparently handles :class:`~lazy_string.LazyString`.

    Usage::

        import json
        from utils.i18n import LazyStringEncoder, _l

        data = {"label": _l("Save")}
        print(json.dumps(data, cls=LazyStringEncoder))

    Or use the convenience wrapper :func:`json_dumps`.
    """

    def default(self, obj):
        if isinstance(obj, LazyString):
            return str(obj)
        return super().default(obj)


def json_dumps(obj, **kwargs) -> str:
    """Serialize *obj* to JSON, automatically resolving any ``LazyString`` values.

    A convenience wrapper around :func:`json.dumps` that plugs in
    :class:`LazyStringEncoder`.

    Args:
        obj: The object to serialize.
        **kwargs: Additional keyword arguments forwarded to :func:`json.dumps`.

    Returns:
        A JSON-formatted string.

    Example::

        from utils.i18n import json_dumps, _l

        data = {"label": _l("Save"), "count": 42}
        print(json_dumps(data))
    """
    return json.dumps(obj, cls=LazyStringEncoder, **kwargs)


def available_locales(locale_dir: str, domain: str) -> list[str]:
    """Return locale codes that have a compiled .mo file in *locale_dir*.

    Walks ``<locale_dir>/<lang>/LC_MESSAGES/<domain>.mo``.
    """
    langs = []
    try:
        for entry in sorted(os.scandir(locale_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            mo = os.path.join(entry.path, "LC_MESSAGES", f"{domain}.mo")
            if os.path.isfile(mo):
                langs.append(entry.name)
    except OSError:
        # Covers FileNotFoundError, PermissionError, and other OS-level failures.
        pass
    return langs


def iter_translated_message(
    msg: str,
    locale_dir: str,
    domain: str,
    langs: list[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Cycle through all available locales, yielding (lang, translated_msg).

    Args:
        msg:        The source string to translate, e.g. ``"Welcome"``.
        locale_dir: Same ``localedir`` you pass to ``init_gettext``.
        domain:     Same domain you pass to ``init_gettext``.
        langs:      Explicit list of locale codes to iterate.
                    If ``None``, auto-discovered via :func:`available_locales`.

    Yields:
        ``(lang, translated_text)`` pairs, one per available locale.
    """
    if langs is None:
        langs = available_locales(locale_dir, domain)

    for lang in langs:
        translation = gettext.translation(
            domain,
            localedir=locale_dir,
            languages=[lang],
            fallback=True,
        )
        yield lang, translation.gettext(msg)


#: Short alias for :func:`iter_translated_message`, mirroring the ``_`` / ``_l``
#: naming convention for quick interactive or scripting use.
#:
#: Example::
#:
#:     for lang, text in _lt("Save", "locales", "messages"):
#:         print(lang, text)
_lt = iter_translated_message
