#!/usr/bin/env python3
"""
Equivalent of the bash i18n script, using pybabel (Babel) instead of
xgettext / msginit / msgmerge / msgfmt.

Usage:
    python3 locales/generate.py

Requirements:
    pip install Babel
"""
import argparse
import re
import subprocess
from pathlib import Path


def run(cmd: list, **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate localization files")
    parser.add_argument(
        "--skip-pot",
        action="store_true",
        help="Skip regenerating the .pot template file",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    locales_dir = root / "locales"
    cfg_file = root / "pyproject.toml"
    pot_file = locales_dir / "app.pot"

    locales_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1.  Extract strings → locales/app.pot
    #     Mirrors: xgettext --language=Python --keyword=_ ...
    # ------------------------------------------------------------------ #
    if not args.skip_pot:
        run([
            "pybabel", "extract",
            "--mapping", str(cfg_file),
            "--output", str(pot_file),
            "--strip-comments",
            "--project", "expressive",
            ".",
        ], cwd=root)

    # ------------------------------------------------------------------ #
    # 2.  For each locale: init (first time) or update, then compile
    #     Mirrors: msginit / msgmerge / msgfmt
    # ------------------------------------------------------------------ #
    locale_re = re.compile(r'^[a-z]{2}(_[A-Z]{2})?$')

    for locale_path in sorted(locales_dir.iterdir()):
        if not locale_path.is_dir():
            continue
        locale = locale_path.name
        if not locale_re.match(locale):
            continue

        lc_dir = locale_path / "LC_MESSAGES"
        lc_dir.mkdir(parents=True, exist_ok=True)
        po_file = lc_dir / "app.po"
        mo_file = lc_dir / "app.mo"

        if not po_file.exists():
            # First time: create a new catalogue for this locale
            run([
                "pybabel", "init",
                "--input-file", str(pot_file),
                "--output-dir", str(locales_dir),
                "--locale", locale,
                "--output-file", str(po_file),
            ])
        else:
            # Already exists: merge new/changed strings in-place
            run([
                "pybabel", "update",
                "--input-file", str(pot_file),
                "--output-dir", str(locales_dir),
                "--locale", locale,
                "--output-file", str(po_file),
            ])

        # Compile .po → .mo
        run([
            "pybabel", "compile",
            "--use-fuzzy",
            "--input-file", str(po_file),
            "--output-file", str(mo_file),
        ])


if __name__ == "__main__":
    main()
