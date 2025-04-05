
import os, argparse
from utils.i18n import _, init_gettext
parser = argparse.ArgumentParser(description='Choose application language.')
parser.add_argument('--lang', default='en', help='Set language for localization (e.g. zh_CN, en)')
args = parser.parse_args()
init_gettext(args.lang, os.path.join(os.path.dirname(__file__), 'locales')
, "app")


import json
import asyncio
import collections.abc

import webview
from nicegui import ui, app

from expressive import process_expressions
from utils.gpu import add_cuda11_to_path
from expressions.base import getExpressionLoader


def dict_update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def create_gui():
    # Initialize state
    state = {
        "utau_wav"    : "",
        "ref_wav"     : "",
        "ustx_input"  : "",
        "ustx_output" : "",
        "track_number": 1,
        "expressions" : {
            "dyn": {
                "selected"    : False,
                "align_radius": 1,
                "smoothness"  : 2,
                "scaler"      : 2.0,
            },
            "pitd": {
                "selected"       : False,
                "confidence_utau": 0.8,
                "confidence_ref" : 0.6,
                "align_radius"   : 1,
                "semitone_shift" : None,
                "smoothness"     : 2,
                "scaler"         : 2.0,
            },
        },
    }

    async def export_config(state=state):
        file = await app.native.main_window.create_file_dialog(  # type: ignore
            dialog_type=webview.SAVE_DIALOG,
            file_types=("JSON files (*.json)",),
            save_filename="expressive_config",
        )
        if file and len(file) > 0:
            try:
                with open(file, "w+", encoding="utf-8-sig") as f:  # type: ignore
                    json.dump(state, f, indent=4)
                ui.notify(_("Config exported successfully!"), type="positive")
            except Exception as e:
                ui.notify(_("Failed to export config") + f": {str(e)}", type="negative")

    async def import_config(state=state):
        file = await app.native.main_window.create_file_dialog(  # type: ignore
            dialog_type=webview.OPEN_DIALOG,
            file_types=("JSON files (*.json)",),
        )
        if file and len(file) > 0:
            try:
                with open(file[0], "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                    dict_update(state, cfg)

                ui.notify(_("Config imported successfully!"), type="positive")
                ui.update()
            except Exception as e:
                ui.notify(_("Failed to import config") + f": {str(e)}", type="negative")

    async def run_processing():
        # Prepare expressions list
        expressions = []
        if state["expressions"]["dyn"]["selected"]:
            expressions.append(
                {
                    "expression": "dyn",
                    "align_radius": state["expressions"]["dyn"]["align_radius"],
                    "smoothness": state["expressions"]["dyn"]["smoothness"],
                    "scaler": state["expressions"]["dyn"]["scaler"],
                }
            )

        if state["expressions"]["pitd"]["selected"]:
            expressions.append(
                {
                    "expression": "pitd",
                    "confidence_utau": state["expressions"]["pitd"]["confidence_utau"],
                    "confidence_ref": state["expressions"]["pitd"]["confidence_ref"],
                    "align_radius": state["expressions"]["pitd"]["align_radius"],
                    "semitone_shift": (
                        int(state["expressions"]["pitd"]["semitone_shift"])
                        if state["expressions"]["pitd"]["semitone_shift"] is not None
                        else None
                    ),
                    "smoothness": state["expressions"]["pitd"]["smoothness"],
                    "scaler": state["expressions"]["pitd"]["scaler"],
                }
            )

        with status_row:
            process_button.disable()
            spinner_dialog.open()
            try:
                # Run in executor since process_expressions is likely synchronous
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,  # Uses default executor
                    lambda: process_expressions(
                        state["utau_wav"],
                        state["ref_wav"],
                        state["ustx_input"],
                        state["ustx_output"],
                        state["track_number"],
                        expressions,
                    ),
                )
                ui.notify(_("Processing completed successfully!"), type="positive")
            except Exception as e:
                ui.notify(_("Error during processing") + f": {str(e)}", type="negative")
            finally:
                spinner_dialog.close()
                process_button.enable()

    async def choose_file(field, ftypes):
        file = await app.native.main_window.create_file_dialog(  # type: ignore
            dialog_type=webview.OPEN_DIALOG,
            file_types=ftypes,
        )
        if file is not None and len(file) > 0:
            state[field] = file[0]
            file_inputs[field].set_value(state[field])

    async def save_file(field, ftypes, fname):
        file = await app.native.main_window.create_file_dialog(  # type: ignore
            dialog_type=webview.SAVE_DIALOG,
            file_types=ftypes,
            save_filename=fname,
        )
        if file is not None and len(file) > 0:
            state[field] = file
            file_inputs[field].set_value(state[field])

    async def process_files():
        # Validate inputs
        if (
            not state["utau_wav"]
            or not state["ref_wav"]
            or not state["ustx_input"]
            or not state["ustx_output"]
        ):
            ui.notify(_("Please fill all required file paths"), type="negative")
            return

        if not any(
            [
                state["expressions"]["dyn"]["selected"],
                state["expressions"]["pitd"]["selected"],
            ]
        ):
            ui.notify(_("Please select at least one expression to apply"), type="negative")
            return
        asyncio.create_task(run_processing())

    # File inputs
    file_inputs = {}
    general_args = getExpressionLoader(None).args
    with ui.card().classes("w-full"):
        ui.label(_("File Paths")).classes("text-xl font-bold")

        with ui.row().classes("w-full"):
            file_inputs["ref_wav"] = (
                ui.input(
                    label=_("Reference WAV File"),
                    placeholder=general_args.ref_path.help,
                    validation={_("Input required"): lambda v: bool(v)},
                )
                .bind_value(state, "ref_wav")
                .classes("flex-grow")
            )
            ui.button(
                icon="folder",
                on_click=lambda: choose_file("ref_wav", ("WAV files (*.wav)",)),
            ).classes("self-end")

        with ui.row().classes("w-full"):
            file_inputs["utau_wav"] = (
                ui.input(
                    label=_("UTAU WAV File"),
                    placeholder=general_args.utau_path.help,
                    validation={_("Input required"): lambda v: bool(v)},
                )
                .bind_value(state, "utau_wav")
                .classes("flex-grow")
            )
            ui.button(
                icon="folder",
                on_click=lambda: choose_file("utau_wav", ("WAV files (*.wav)",)),
            ).classes("self-end")

        with ui.row().classes("w-full"):
            file_inputs["ustx_input"] = (
                ui.input(
                    label=_("Input USTX File"),
                    placeholder=general_args.ustx_path.help,
                    validation={_("Input required"): lambda v: bool(v)},
                )
                .bind_value(state, "ustx_input")
                .classes("flex-grow")
            )
            ui.button(
                icon="folder",
                on_click=lambda: choose_file("ustx_input", ("USTX files (*.ustx)",)),
            ).classes("self-end")

        with ui.row().classes("w-full"):
            file_inputs["ustx_output"] = (
                ui.input(
                    label=_("Output USTX File"),
                    placeholder=_("Path to save processed USTX file"),
                    validation={_("Input required"): lambda v: bool(v)},
                )
                .bind_value(state, "ustx_output")
                .classes("flex-grow")
            )
            ui.button(
                icon="save",
                on_click=lambda: save_file(
                    "ustx_output", ("USTX files (*.ustx)",), "output"
                ),
            ).classes("self-end")

        ui.number(label=_("Track Number"), min=1, format="%d").bind_value(
            state, "track_number"
        ).classes("w-full").tooltip(general_args.track_number.help)

    # Expression selection
    with ui.card().classes("w-full"):
        ui.label(_("Expression Selection")).classes("text-xl font-bold")

        with ui.row():
            ui.checkbox(_("Dynamics (dyn)")).bind_value(
                state["expressions"]["dyn"], "selected"
            )
            ui.checkbox(_("Pitch Deviation (pitd)")).bind_value(
                state["expressions"]["pitd"], "selected"
            )

    # Dyn parameters
    dyn_args = getExpressionLoader("dyn").args
    with ui.card().classes("w-full").bind_visibility_from(
        state["expressions"]["dyn"], "selected"
    ):
        ui.label(_("Dynamics Expression Parameters")).classes("text-lg font-bold")

        with ui.grid(columns=3).classes("w-full"):
            ui.number(label=_("Align Radius"), min=1, format="%d").bind_value(
                state["expressions"]["dyn"], "align_radius"
            ).tooltip(dyn_args.align_radius.help)

            ui.number(label=_("Smoothness"), min=0, format="%d").bind_value(
                state["expressions"]["dyn"], "smoothness"
            ).tooltip(dyn_args.smoothness.help)

            ui.number(label=_("Scaler"), min=0.0, step=0.1, format="%.1f").bind_value(
                state["expressions"]["dyn"], "scaler"
            ).tooltip(dyn_args.scaler.help)

    # Pitd parameters
    pitd_args = getExpressionLoader("pitd").args
    with ui.card().classes("w-full").bind_visibility_from(
        state["expressions"]["pitd"], "selected"
    ):
        ui.label(_("Pitch Deviation Expression Parameters")).classes("text-lg font-bold")

        with ui.grid(columns=3).classes("w-full"):
            ui.number(
                label=_("UTAU Confidence"), min=0.0, max=1.0, step=0.1, format="%.1f"
            ).bind_value(state["expressions"]["pitd"], "confidence_utau"
            ).tooltip(pitd_args.confidence_utau.help)

            ui.number(
                label=_("Reference Confidence"), min=0.0, max=1.0, step=0.1, format="%.1f"
            ).bind_value(state["expressions"]["pitd"], "confidence_ref"
            ).tooltip(pitd_args.confidence_ref.help)

            ui.number(label=_("Align Radius"), min=1, format="%d").bind_value(
                state["expressions"]["pitd"], "align_radius"
            ).tooltip(pitd_args.align_radius.help)

            ui.number(label=_("Semitone Shift"), step=1, format="%d").bind_value(
                state["expressions"]["pitd"], "semitone_shift"
            ).tooltip(pitd_args.semitone_shift.help)

            ui.number(label=_("Smoothness"), min=0, format="%d").bind_value(
                state["expressions"]["pitd"], "smoothness"
            ).tooltip(pitd_args.smoothness.help)

            ui.number(label=_("Scaler"), min=0.0, step=0.1, format="%.1f").bind_value(
                state["expressions"]["pitd"], "scaler"
            ).tooltip(pitd_args.scaler.help)

    # Add the config buttons above the Process button
    with ui.row().classes("w-full justify-between"):
        ui.button(
            _("Import Config"),
            on_click=import_config,
            color="secondary",
            icon="file_download",
        )
        ui.button(
            _("Export Config"),
            on_click=export_config,
            color="secondary",
            icon="file_upload",
        )

    # Process button and progress indicator
    with ui.row().classes("w-full") as status_row:
        with ui.dialog() as spinner_dialog, ui.card():
            ui.spinner(size="lg")
        process_button = ui.button(
            _("Process"), on_click=process_files, icon="play_arrow"
        ).classes("flex-grow")


# Run the app
if __name__ in {"__main__", "__mp_main__"}:
    add_cuda11_to_path()
    create_gui()
    ui.run(
        title="Expressive GUI",
        dark=None,
        native=True,
        reload=False,
        window_size=(600, 640),
    )
