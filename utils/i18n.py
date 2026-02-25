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

import gettext
import json
import threading
from typing import Callable, Optional

from lazy_string import LazyString


_current_gettext: Optional[Callable[[str], str]] = None
_lock = threading.Lock()


def init_gettext(lang: str, locale_dir: str, domain: str) -> None:
    """Initialize or hot-swap the active ``gettext`` translation.

    This function is safe to call from any thread and may be called multiple
    times to switch locales at runtime.  The translation object is constructed
    outside the lock (I/O-bound) and only the final assignment is serialized,
    so two concurrent calls may race on which locale "wins" — callers are
    responsible for avoiding that at the application level.

    Args:
        lang: BCP-47 / POSIX locale code, e.g. ``"zh_CN"`` or ``"en_US"``.
        locale_dir: Path to the directory that contains ``<lang>/LC_MESSAGES/``
            sub-directories (the ``localedir`` argument passed to
            :func:`gettext.translation`).
        domain: Message catalog domain, e.g. ``"messages"``.  Must match the
            ``.mo`` file name inside the locale directory.

    Raises:
        FileNotFoundError: If ``fallback=False`` were set and no ``.mo`` file
            is found.  With the current ``fallback=True`` setting this is
            suppressed and the identity function is used instead.
    """
    translation = gettext.translation(
        domain,
        localedir=locale_dir,
        languages=[lang],
        fallback=True,
    )
    with _lock:
        global _current_gettext
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


def patch_nicegui_json() -> None:
    """Patch NiceGUI's orjson converter to resolve ``LazyString`` during serialization.

    Monkey-patches ``_orjson_converter`` in ``nicegui.json.orjson_wrapper`` so
    that ``LazyString`` objects are automatically converted to plain strings
    when NiceGUI serializes UI state.

    Call this function **once** during application initialization, before
    creating any NiceGUI UI elements::

        from utils.i18n import init_gettext, patch_nicegui_json

        init_gettext("en", "locales", "app")
        patch_nicegui_json()

    If NiceGUI is not installed this function is a no-op.
    """
    try:
        from nicegui.json import orjson_wrapper

        original_converter = orjson_wrapper._orjson_converter

        def patched_converter(obj):
            if isinstance(obj, LazyString):
                return str(obj)
            return original_converter(obj)

        orjson_wrapper._orjson_converter = patched_converter

    except ImportError:
        pass
