"""Hatch build hook: compile gettext .po -> .mo before wheel packaging."""

from __future__ import annotations

import glob
import os

from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        locales_dir = os.path.join(self.root, "locales")
        for po_file in glob.glob(
            os.path.join(locales_dir, "**", "*.po"), recursive=True
        ):
            mo_file = os.path.splitext(po_file)[0] + ".mo"
            with open(po_file, "rb") as f:
                catalog = read_po(f)
            with open(mo_file, "wb") as f:
                write_mo(f, catalog)
            # artifacts bypasses .gitignore so the compiled .mo is included in the wheel
            build_data["artifacts"].append(
                os.path.relpath(mo_file, self.root)
            )
