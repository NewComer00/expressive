import gettext
import builtins


# If gettext i18n is needed, this function must be placed BEFORE any ExpressionLoader loaded
def init_gettext(lang: str, locale_dir: str, domain: str):
    lang_translations = gettext.translation(domain, localedir=locale_dir, languages=[lang], fallback=False)
    builtins._ = lang_translations.gettext # type: ignore


def _(mymessage: str) -> str:
    translate = getattr(builtins, '_', None)
    if callable(translate):
        return translate(mymessage) # type: ignore
    return mymessage