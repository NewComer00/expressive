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

LOCALE_RE = re.compile(r'^[a-z]{2}(_[A-Z]{2})?$')
APP_NAME = "app"


def run(cmd: list, **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def extract_pot(root: Path, cfg_file: Path, pot_file: Path) -> None:
    """Extract translatable strings into a .pot template."""
    run([
        "pybabel", "extract",
        "--mapping", cfg_file,
        "--output", pot_file,
        "--strip-comments",
        "--project", "expressive",
        ".",
    ], cwd=root)


def init_po(pot_file: Path, locales_dir: Path, locale: str, po_file: Path) -> None:
    """Create a new .po catalogue for a locale (first time)."""
    run([
        "pybabel", "init",
        "--input-file", pot_file,
        "--output-dir", locales_dir,
        "--locale", locale,
        "--output-file", po_file,
    ])


def update_po(pot_file: Path, locales_dir: Path, locale: str, po_file: Path) -> None:
    """Merge new/changed strings into an existing .po file."""
    run([
        "pybabel", "update",
        "--input-file", pot_file,
        "--output-dir", locales_dir,
        "--locale", locale,
        "--output-file", po_file,
    ])


def compile_po(po_file: Path, mo_file: Path) -> None:
    """Compile .po → .mo."""
    run([
        "pybabel", "compile",
        "--input-file", po_file,
        "--output-file", mo_file,
    ])


def iter_locales(locales_dir: Path):
    for path in sorted(locales_dir.iterdir()):
        if path.is_dir() and LOCALE_RE.match(path.name):
            yield path


def process_locale(
    locale_path: Path,
    locales_dir: Path,
    pot_file: Path,
    mo_only: bool,
) -> None:
    locale = locale_path.name
    lc_dir = locale_path / "LC_MESSAGES"
    lc_dir.mkdir(parents=True, exist_ok=True)
    po_file = lc_dir / f"{APP_NAME}.po"
    mo_file = lc_dir / f"{APP_NAME}.mo"

    if not mo_only:
        if po_file.exists():
            update_po(pot_file, locales_dir, locale, po_file)
        else:
            init_po(pot_file, locales_dir, locale, po_file)

    compile_po(po_file, mo_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate localization files")
    parser.add_argument(
        "--skip-pot",
        action="store_true",
        help="Skip regenerating the .pot template file",
    )
    parser.add_argument(
        "--mo-only",
        action="store_true",
        help="Skip .pot extraction and .po update; only compile .po → .mo",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    locales_dir = root / "locales"
    pot_file = locales_dir / f"{APP_NAME}.pot"

    locales_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pot and not args.mo_only:
        extract_pot(root, root / "pyproject.toml", pot_file)

    for locale_path in iter_locales(locales_dir):
        process_locale(locale_path, locales_dir, pot_file, args.mo_only)


if __name__ == "__main__":
    main()
