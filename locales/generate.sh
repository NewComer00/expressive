#!/bin/bash
set -euxo pipefail

ROOT=$(dirname $(dirname "$(realpath "$0")"))
cd "$ROOT"

# Generate .pot file
xgettext --language=Python --keyword=_ --from-code=UTF-8 --output=locales/app.pot \
    $(git ls-files '*.py' 2>/dev/null)

for locale in $(ls locales | grep -E '^[a-z]{2}(_[A-Z]{2})?$'); do
    # Generate .po files for each locale
    mkdir -p "locales/$locale/LC_MESSAGES"
    if [ ! -f "locales/$locale/LC_MESSAGES/app.po" ]; then
        msginit --no-translator --locale="$locale" --input=locales/app.pot \
            --output-file="locales/$locale/LC_MESSAGES/app.po"
    fi
    msgmerge --update "locales/$locale/LC_MESSAGES/app.po" \
        locales/app.pot

    # Compile .po files to .mo files
    msgfmt --output-file="locales/$locale/LC_MESSAGES/app.mo" \
        "locales/$locale/LC_MESSAGES/app.po"
done
