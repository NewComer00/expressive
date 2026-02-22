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

Thread safety
-------------
All access to the active translation function is protected by a module-level
``threading.Lock``. ``init_gettext()`` may be called from any thread to hot-
swap the active locale; in-flight translations will complete with the old
locale before the switch takes effect.
"""

import gettext
import threading
from typing import Callable, Optional


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


class LazyString:
    """A translation proxy that defers evaluation until the string is used.

    ``LazyString`` stores only the raw message key and resolves it through
    :func:`_` the first time the instance is converted to :class:`str`.  This
    means the correct locale is applied even if the object was created before
    :func:`init_gettext` was called.

    Prefer :func:`_l` over instantiating this class directly.

    Operator support
    ----------------
    The most common :class:`str` operations are forwarded to the resolved
    string so that ``LazyString`` instances can be used as drop-in
    replacements in most contexts:

    * ``str(ls)`` — resolve and return the translated string
    * ``repr(ls)`` — show the raw key, useful for debugging
    * ``ls == other`` — compare resolved strings
    * ``hash(ls)`` — hash of the *key* (stable across locale switches)
    * ``len(ls)`` — length of the resolved string
    * ``ls + other``, ``other + ls`` — string concatenation
    * ``ls % args`` — %-style formatting
    * ``ls.format(...)`` / ``ls.format_map(...)`` — :meth:`str.format` style

    Note:
        Methods such as :meth:`format` return plain :class:`str` objects, not
        ``LazyString`` instances, so laziness is *not* preserved through
        formatting operations.  This is intentional — once arguments have been
        interpolated the result is no longer locale-independent.
    """

    __slots__ = ("_msg",)

    def __init__(self, msg: str) -> None:
        """Store the untranslated message key.

        Args:
            msg: The source-language string (translation key).
        """
        self._msg = msg

    def __str__(self) -> str:
        """Resolve and return the translated string.

        Delegates to :func:`_`, which acquires the module lock and applies the
        currently active translation.
        """
        return _(self._msg)

    def __repr__(self) -> str:
        """Return an unambiguous representation showing the raw key.

        Example::

            >>> repr(_l("Save"))
            "LazyString('Save')"
        """
        return f"LazyString({self._msg!r})"

    def __eq__(self, other: object) -> bool:
        """Compare the resolved string to *other*.

        Args:
            other: Any object.  Compared via ``str(self) == str(other)``.
        """
        return str(self) == str(other)

    def __hash__(self) -> int:
        """Return a hash of the *key*, not the translated value.

        Hashing on the key ensures stability across locale switches and lets
        ``LazyString`` instances be used as dictionary keys or set members
        without surprising behaviour when the locale changes.
        """
        return hash(self._msg)

    def __len__(self) -> int:
        """Return the length of the resolved (translated) string."""
        return len(str(self))

    def __add__(self, other: object) -> str:
        """Concatenate the resolved string with *other* on the right."""
        return str(self) + str(other)

    def __radd__(self, other: object) -> str:
        """Concatenate *other* on the left with the resolved string."""
        return str(other) + str(self)

    def __mod__(self, args: object) -> str:
        """Apply %-style formatting to the resolved string.

        Example::

            _l("Hello, %s!") % name
        """
        return str(self) % args

    def format(self, *args: object, **kwargs: object) -> str:
        """Apply :meth:`str.format` to the resolved string.

        Returns a plain :class:`str`; laziness is not preserved.

        Example::

            _l("Hello, {name}!").format(name="Alice")
        """
        return str(self).format(*args, **kwargs)

    def format_map(self, mapping: object) -> str:
        """Apply :meth:`str.format_map` to the resolved string.

        Returns a plain :class:`str`; laziness is not preserved.

        Example::

            _l("Hello, {name}!").format_map({"name": "Alice"})
        """
        return str(self).format_map(mapping)  # type: ignore[arg-type]

    def __json__(self) -> str:
        """Return the resolved string for JSON serialization.

        This method is recognized by some JSON libraries (like simplejson)
        and custom encoders to automatically serialize LazyString instances.
        """
        return str(self)


def _l(msg: str) -> LazyString:
    """Return a :class:`LazyString` that is translated on first use.

    Use this function **only** for strings that are defined at module or class
    level — i.e. before :func:`init_gettext` has been called.  For all other
    strings prefer the eagerly evaluated :func:`_`.

    Args:
        msg: The source-language string (translation key).

    Returns:
        A :class:`LazyString` proxy wrapping *msg*.

    Example::

        # Evaluated at import time — locale not yet set, so use _l
        BUTTON_LABEL = _l("Save")

        class MyModel:
            verbose_name = _l("item")
    """
    return LazyString(msg)


class LazyStringEncoder:
    """JSON encoder that handles LazyString instances.

    Usage with json.dumps::

        import json
        from utils.i18n import LazyStringEncoder, _l

        data = {"label": _l("Save")}
        json.dumps(data, cls=LazyStringEncoder)

    Or use the helper function::

        json_dumps(data)
    """

    @staticmethod
    def default(obj):
        """Convert LazyString to str for JSON serialization."""
        if isinstance(obj, LazyString):
            return str(obj)
        raise TypeError(f"Type is not JSON serializable: {type(obj).__name__}")


def json_dumps(obj, **kwargs):
    """Serialize obj to JSON, automatically handling LazyString instances.

    This is a convenience wrapper around json.dumps that uses LazyStringEncoder.

    Args:
        obj: The object to serialize.
        **kwargs: Additional arguments passed to json.dumps.

    Returns:
        A JSON-formatted string.

    Example::

        from utils.i18n import json_dumps, _l

        data = {"label": _l("Save"), "count": 42}
        json_str = json_dumps(data)
    """
    import json
    return json.dumps(obj, default=LazyStringEncoder.default, **kwargs)


def patch_nicegui_json():
    """Patch NiceGUI's orjson converter to handle LazyString instances.

    This function monkey-patches the _orjson_converter in nicegui.json.orjson_wrapper
    to automatically convert LazyString objects to strings during JSON serialization.

    Call this function once during application initialization, before creating any
    NiceGUI UI elements.

    Example::

        from utils.i18n import init_gettext, patch_nicegui_json

        init_gettext("en", "locales", "app")
        patch_nicegui_json()  # Patch before creating UI
    """
    try:
        from nicegui.json import orjson_wrapper

        # Store the original converter
        original_converter = orjson_wrapper._orjson_converter

        def patched_converter(obj):
            """Enhanced converter that handles LazyString."""
            if isinstance(obj, LazyString):
                return str(obj)
            # Fall back to the original converter for other types
            return original_converter(obj)

        # Replace the converter
        orjson_wrapper._orjson_converter = patched_converter

    except ImportError:
        # NiceGUI not installed, skip patching
        pass
