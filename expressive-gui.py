# For multiprocessing support in PyInstaller on Windows, following code is needed before using the argparse module
# https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
import multiprocessing
multiprocessing.freeze_support()


# For i18n support, argparse is required to set the language of gettext before importing any other modules
import os, argparse
from utils.i18n import _, init_gettext
parser = argparse.ArgumentParser(description='Choose application language.')
parser.add_argument('--lang', default='en', help='Set language for localization (e.g. zh_CN, en)')
args = parser.parse_args()
init_gettext(args.lang, os.path.join(os.path.dirname(__file__), 'locales'), "app")


# Application code starts here
import sys
import json
import logging
import asyncio
from collections.abc import Mapping
from os.path import splitext, basename

import webview
from nicegui import ui, app

from utils.gpu import add_cuda_to_path
from expressive import process_expressions
from utils.ui import blink_taskbar_window, change_window_style
from expressions.base import getExpressionLoader, get_registered_expressions


class LogElementHandler(logging.Handler):
    """A logging handler that emits messages to a log element."""

    def __init__(self, element: ui.log, level: int = logging.NOTSET) -> None:
        self.element = element
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.element.push(msg)
        except Exception:
            self.handleError(record)


def dict_update(d: dict, u: Mapping):
    """Recursively update a dictionary with another dictionary.
    This function updates the dictionary `d` with the values from the dictionary `u`.
    See: https://stackoverflow.com/a/3233356
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


@app.on_connect
def close_splash():
    """Close the splash screen when the app is connected if this script is frozen.
    This is a workaround for PyInstaller, which doesn't support splash screen in the main thread
    
    See: https://github.com/zauberzeug/nicegui/discussions/3536
         https://stackoverflow.com/questions/71057636/how-can-i-solve-no-module-named-pyi-splash-after-using-pyinstaller
    """
    if getattr(sys, 'frozen', False):
        import pyi_splash # type: ignore
        pyi_splash.close()


def create_gui():
    # Initialize state
    state = {
        "utau_wav"    : "",
        "ref_wav"     : "",
        "ustx_input"  : "",
        "ustx_output" : "",
        "track_number": 1,
        "expressions" : {
            exp_name: {
                "selected": False,
                **{
                    arg.name: arg.default
                    for arg in getExpressionLoader(exp_name).get_args_dict().values()
                },
            } for exp_name in get_registered_expressions()
        },
    }

    # state = {
    #     "utau_wav"    : "",
    #     "ref_wav"     : "",
    #     "ustx_input"  : "",
    #     "ustx_output" : "",
    #     "track_number": 1,
    #     "expressions" : {
    #         "dyn": {
    #             "selected"    : False,
    #             "align_radius": 1,
    #             "smoothness"  : 2,
    #             "scaler"      : 2.0,
    #         },
    #         "pitd": {
    #             "selected"       : False,
    #             "confidence_utau": 0.8,
    #             "confidence_ref" : 0.6,
    #             "align_radius"   : 1,
    #             "semitone_shift" : None,
    #             "smoothness"     : 2,
    #             "scaler"         : 2.0,
    #         },
    #     },
    # }

    def on_color_scheme_changed(event):
        """Change the window style based on the color scheme."""
        color_scheme = event.args
        if color_scheme == 'dark':
            change_window_style(app.config.title, 'mica')
        else:
            change_window_style(app.config.title, 'light')

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
        expressions = [
            {
                "expression": exp_name,
                ** {
                    arg.name: state["expressions"][exp_name][arg.name]
                    for arg in getExpressionLoader(exp_name).get_args_dict().values()
                }
            }
            for exp_name in get_registered_expressions()
            if state["expressions"][exp_name]["selected"]
        ]

        with status_row:
            process_button.disable()
            spinner_dialog.open()
            log_element.clear()
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
                blink_taskbar_window(app.config.title)
                logger_app.info(_("Processing completed successfully!"))
                ui.notify(_("Processing completed successfully!"), type="positive")
            except Exception as e:
                logger_app.exception(_("Error during processing") + f": {str(e)}")
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
                state["expressions"][exp_name]["selected"]
                for exp_name in get_registered_expressions()
            ]
        ):
            ui.notify(_("Please select at least one expression to apply"), type="negative")
            return
        asyncio.create_task(run_processing())

    def setup_loggers(log_element: ui.log):
        """Configure application and expression loggers to write to the log_element."""
        def configure_logger(name: str, formatter: logging.Formatter):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            handler = LogElementHandler(log_element)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger, handler

        formatter_app = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S"
        )
        formatter_exp = logging.Formatter(
            "%(asctime)s %(levelname)s [%(expression)s]: %(message)s", datefmt="%H:%M:%S"
        )

        logger_app, handler_app = configure_logger(splitext(basename(__file__))[0], formatter_app)
        logger_exp, handler_exp = configure_logger(getExpressionLoader(None).__name__, formatter_exp)

        # Clean up on disconnect
        ui.context.client.on_disconnect(lambda: (
            logger_app.removeHandler(handler_app),
            logger_exp.removeHandler(handler_exp)
        ))

        return logger_app, logger_exp

    # Set up the UI dark mode
    ui.dark_mode(None)
    ui.add_head_html('''
    <script>
        // Emit an event when the color scheme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            const colorScheme = event.matches ? "dark" : "light"; 
            emitEvent('color-scheme-changed', colorScheme);
        });

        // Initial check
        window.onload = function() {
            const colorScheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? "dark" : "light";
            emitEvent('color-scheme-changed', colorScheme);
        };
    </script>
    ''')
    ui.on('color-scheme-changed', lambda e: on_color_scheme_changed(e))

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
            state, "track_number",
            forward=lambda v: general_args.track_number.type(v) if v is not None else None,
        ).classes("w-full").tooltip(general_args.track_number.help)

    # Expression selection
    with ui.card().classes("w-full"):
        ui.label(_("Expression Selection")).classes("text-xl font-bold")

        with ui.row():
            for exp_name in get_registered_expressions():
                exp_info = getExpressionLoader(exp_name).expression_info
                ui.checkbox(exp_info).bind_value(
                    state["expressions"][exp_name], "selected"
                )

    # Dyn parameters
    dyn_args = getExpressionLoader("dyn").args
    dyn_info = getExpressionLoader("dyn").expression_info
    with ui.card().classes("w-full").bind_visibility_from(
        state["expressions"]["dyn"], "selected"
    ):
        ui.label(dyn_info).classes("text-lg font-bold")

        with ui.grid(columns=3).classes("w-full"):
            ui.number(label=_("Align Radius"), min=1, format="%d").bind_value(
                state["expressions"]["dyn"], "align_radius",
                forward=lambda v: dyn_args.align_radius.type(v) if v is not None else None,
            ).tooltip(dyn_args.align_radius.help)

            ui.number(label=_("Smoothness"), min=0, format="%d").bind_value(
                state["expressions"]["dyn"], "smoothness",
                forward=lambda v: dyn_args.smoothness.type(v) if v is not None else None,
            ).tooltip(dyn_args.smoothness.help)

            ui.number(label=_("Scaler"), min=0.0, step=0.1, format="%.1f").bind_value(
                state["expressions"]["dyn"], "scaler",
                forward=lambda v: dyn_args.scaler.type(v) if v is not None else None,
            ).tooltip(dyn_args.scaler.help)

    # Pitd parameters
    pitd_args = getExpressionLoader("pitd").args
    pitd_info = getExpressionLoader("pitd").expression_info
    with ui.card().classes("w-full").bind_visibility_from(
        state["expressions"]["pitd"], "selected"
    ):
        ui.label(pitd_info).classes("text-lg font-bold")

        with ui.grid(columns=3).classes("w-full"):
            ui.number(
                label=_("UTAU Confidence"), min=0.0, max=1.0, step=0.1, format="%.1f"
            ).bind_value(state["expressions"]["pitd"], "confidence_utau",
                            forward=lambda v: pitd_args.confidence_utau.type(v) if v is not None else None,
            ).tooltip(pitd_args.confidence_utau.help)

            ui.number(
                label=_("Reference Confidence"), min=0.0, max=1.0, step=0.1, format="%.1f"
            ).bind_value(state["expressions"]["pitd"], "confidence_ref",
                            forward=lambda v: pitd_args.confidence_ref.type(v) if v is not None else None,
            ).tooltip(pitd_args.confidence_ref.help)

            ui.number(label=_("Align Radius"), min=1, format="%d").bind_value(
                state["expressions"]["pitd"], "align_radius",
                forward=lambda v: pitd_args.align_radius.type(v) if v is not None else None,
            ).tooltip(pitd_args.align_radius.help)

            ui.number(label=_("Semitone Shift"), step=1, format="%d").bind_value(
                state["expressions"]["pitd"], "semitone_shift",
                forward=lambda v: pitd_args.semitone_shift.type(v) if v is not None else None,
            ).tooltip(pitd_args.semitone_shift.help)

            ui.number(label=_("Smoothness"), min=0, format="%d").bind_value(
                state["expressions"]["pitd"], "smoothness",
                forward=lambda v: pitd_args.smoothness.type(v) if v is not None else None,
            ).tooltip(pitd_args.smoothness.help)

            ui.number(label=_("Scaler"), min=0.0, step=0.1, format="%.1f").bind_value(
                state["expressions"]["pitd"], "scaler",
                forward=lambda v: pitd_args.scaler.type(v) if v is not None else None,
            ).tooltip(pitd_args.scaler.help)

    # Tenc parameters
    tenc_args = getExpressionLoader("tenc").args
    tenc_info = getExpressionLoader("tenc").expression_info
    with ui.card().classes("w-full").bind_visibility_from(
        state["expressions"]["tenc"], "selected"
    ):
        ui.label(tenc_info).classes("text-lg font-bold")

        with ui.grid(columns=3).classes("w-full"):
            ui.number(label=_("Align Radius"), min=1, format="%d").bind_value(
                state["expressions"]["tenc"], "align_radius",
                forward=lambda v: tenc_args.align_radius.type(v) if v is not None else None,
            ).tooltip(tenc_args.align_radius.help)

            ui.number(label=_("Smoothness"), min=0, format="%d").bind_value(
                state["expressions"]["tenc"], "smoothness",
                forward=lambda v: tenc_args.smoothness.type(v) if v is not None else None,
            ).tooltip(tenc_args.smoothness.help)

            ui.number(label=_("Scaler"), min=0.0, step=0.1, format="%.1f").bind_value(
                state["expressions"]["tenc"], "scaler",
                forward=lambda v: tenc_args.scaler.type(v) if v is not None else None,
            ).tooltip(tenc_args.scaler.help)

            ui.number(label=_("Bias"), format="%d").bind_value(
                state["expressions"]["tenc"], "bias",
                forward=lambda v: tenc_args.bias.type(v) if v is not None else None,
            ).tooltip(tenc_args.bias.help)

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

    # Process button, spinner, and log element
    with ui.row().classes("w-full") as status_row:
        with ui.dialog() as spinner_dialog, ui.card():
            ui.spinner(size="lg")
        process_button = ui.button(
            _("Process"), on_click=process_files, icon="play_arrow"
        ).classes("flex-grow")
        with ui.element().classes("relative w-full h-40"):
            log_element = ui.log().classes("w-full h-full select-text cursor-text")

            # Log auto scrolling on update
            # Embed the script in the head of the HTML document
            # since the id of the log element is static during the app's lifetime
            ui.add_head_html(f'''
            <script>
            document.addEventListener('DOMContentLoaded', function () {{
                const logEl = document.getElementById('c{log_element.id}');
                if (logEl) {{
                    const observer = new MutationObserver(() => {{
                        logEl.scrollTop = logEl.scrollHeight;
                    }});
                    observer.observe(logEl, {{ childList: true }});
                }}
            }});
            </script>
            ''')

            # Clipboard button
            ui.button("📋").props("flat dense").classes(
                "absolute top-0 right-5 m-1 text-sm px-1 py-0.5 opacity-70 hover:opacity-100"
            ).on("click", js_handler=f"""
                () => {{
                    const logEl = document.getElementById("c{log_element.id}");
                    if (logEl) {{
                        const text = [...logEl.children].map(el => el.textContent).join("\\n");
                        navigator.clipboard.writeText(text);
                    }}
                }}
            """)

            logger_app, logger_exp = setup_loggers(log_element)


# Run the app
if __name__ in {"__main__", "__mp_main__"}:
    add_cuda_to_path()
    create_gui()
    try:
        ui.run(
            title="Expressive GUI",
            native=True,
            reload=False,
            window_size=(600, 640),
        )
    except KeyboardInterrupt:
        if getattr(sys, 'frozen', False):
            # Suppress KeyboardInterrupt error on exit
            pass
        else:
            raise
