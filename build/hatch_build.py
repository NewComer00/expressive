"""Hatch build hook: compile gettext .po -> .mo and download vendored static deps."""

from __future__ import annotations

import glob
import os
import urllib.request

from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        self._compile_locales(build_data)
        self._vendor_static(build_data)

    def _compile_locales(self, build_data: dict) -> None:
        locales_dir = os.path.join(self.root, "locales")
        for po_file in glob.glob(
            os.path.join(locales_dir, "**", "*.po"), recursive=True
        ):
            mo_file = os.path.splitext(po_file)[0] + ".mo"
            with open(po_file, "rb") as f:
                catalog = read_po(f)
            with open(mo_file, "wb") as f:
                write_mo(f, catalog)
            build_data["artifacts"].append(
                os.path.relpath(mo_file, self.root)
            )

    def _vendor_static(self, build_data: dict) -> None:
        vendor_dir = os.path.join(self.root, "static", "vendor")
        os.makedirs(vendor_dir, exist_ok=True)

        for name, url in self.config.get("vendor-static-deps", []):
            dest = os.path.join(vendor_dir, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                print(f"Downloading {name} from {url}")
                urllib.request.urlretrieve(url, dest)
            build_data["artifacts"].append(
                os.path.relpath(dest, self.root)
            )
