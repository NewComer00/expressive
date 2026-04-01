from __future__ import annotations

import os
import sys
import asyncio
import argparse
from dataclasses import dataclass, field

from nicegui import ui
import plotly.graph_objects as go

from __version__ import VERSION
from utils.ui import ClosableTabs
from utils.fs import APP_RUNTIME_PATH
from utils.monkeypatch import patch_runpy
from utils.i18n import _, detect_preferred_langs, init_gettext
from utils.relay import Deliverer, set_relay_dir, ExpressionLoaderCollectorNaming


set_relay_dir(APP_RUNTIME_PATH / "relay")

LANG          = detect_preferred_langs()
LOCALE_DIR    = os.path.join(os.path.dirname(__file__), "locales")
LOCALE_DOMAIN = "app"


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlotEntry:
    plot: ui.plotly
    fig: go.Figure


@dataclass
class SourceState:
    source: str
    tag_tabs: ui.tabs
    tag_panels: ui.tab_panels
    tags: set[str] = field(default_factory=set)
    plots: list[PlotEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _add_source_tab(source_tabs: ClosableTabs, source_panels: ui.tab_panels, source: str) -> SourceState:
    exp_name, _exp_id, timestamp = ExpressionLoaderCollectorNaming.parse(source)

    with source_panels:
        with ui.tab_panel(source).classes("p-0") as panel:
            tag_tabs   = ClosableTabs().props("dense").classes("max-w-full overflow-hidden")
            tag_panels = ui.tab_panels(tag_tabs).classes("w-full")

    source_tabs.add_closable_tab(
        name=source,
        label=exp_name,
        panel=panel,
        tooltip_text=f'🏷️ {timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
    )

    source_tabs.set_value(source)
    source_panels.set_value(source)

    return SourceState(source=source, tag_tabs=tag_tabs, tag_panels=tag_panels)


def _add_tag_panel(state: SourceState, tag: str, fig: go.Figure, template: str) -> None:
    with state.tag_panels:
        with ui.tab_panel(tag).classes("p-3") as panel:
            state.plots.append(PlotEntry(plot=_render_fig(fig, template), fig=fig))

    state.tag_tabs.add_closable_tab(name=tag, label=tag, panel=panel)

    state.tags.add(tag)
    state.tag_tabs.set_value(tag)
    state.tag_panels.set_value(tag)


def _render_fig(fig: go.Figure, template: str) -> ui.plotly:
    fig.update_layout(
        template=template,
        autosize=True,
    )
    return ui.plotly(fig).classes("w-full h-[calc(100vh-7rem)]")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/")
async def index():
    # State variables
    source_state: dict[str, SourceState] = {}

    async def delivery_loop():
        nonlocal placeholder, source_state, source_tabs, source_panels, plotly_theme
        first = True
        deliverer = Deliverer(skip_existing=True)
        async for source, tag, fig in deliverer.deliver(sources=ExpressionLoaderCollectorNaming.GLOB):
            if first:
                placeholder.delete()
                splitter.visible = True
                first = False
            if source not in source_state:
                source_state[source] = _add_source_tab(source_tabs, source_panels, source)
            if tag not in source_state[source].tags:
                _add_tag_panel(source_state[source], tag, fig, plotly_theme)

    async def on_color_scheme_changed(event):
        nonlocal plotly_theme, source_state
        plotly_theme = "plotly_dark" if event.args == "dark" else "plotly_white"
        for state in source_state.values():
            for entry in state.plots:
                entry.fig.update_layout(template=plotly_theme)
                entry.plot.update()

    # Set up color scheme listener and initial theme
    ui.add_head_html('''
    <script>
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            const colorScheme = event.matches ? "dark" : "light";
            emitEvent('color-scheme-changed', colorScheme);
        });
    </script>
    ''')
    ui.on('color-scheme-changed', on_color_scheme_changed)
    color_scheme = await ui.run_javascript(
        "return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'"
    )
    plotly_theme = "plotly_dark" if color_scheme == "dark" else "plotly_white"

    # Initial UI state with placeholder
    ui.context.client.content.classes('pt-4 pb-0 h-[calc(100vh-1rem)]')
    placeholder = ui.column().classes("w-full items-center justify-center gap-4 py-16")
    with placeholder:
        ui.icon("sensors", size="6rem").classes("p-4 m-4 text-gray-400 animate-ping")
        ui.label(_("Waiting for data to be displayed...")) \
            .classes("text-gray-400 text-xl")
        ui.label(_("Launch Expressive CLI/GUI and start processing to get your plots")) \
            .classes("text-gray-400 text-xl font-bold")

    # Set up the main UI layout with a splitter
    with ui.splitter(value=None).classes("w-full h-full flex-grow") as splitter:
        with splitter.before:
            source_tabs = ClosableTabs(close_button_position="left").props("vertical").classes("w-full h-full")
        with splitter.after:
            source_panels = ui.tab_panels(source_tabs).props("vertical").classes("w-full h-full")
    splitter.visible = False

    # Start the delivery loop
    asyncio.get_event_loop().create_task(delivery_loop())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_gettext(lang=LANG, locale_dir=LOCALE_DIR, domain=LOCALE_DOMAIN)

    parser = argparse.ArgumentParser(
        description=_("Real-time expression curve viewer for Expressive CLI/GUI"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s v{VERSION}")
    args, unknown = parser.parse_known_args()

    ui_params = {
        "title": f"Expressive Viewer v{VERSION}",
        "native": True,
        "reload": False,
        "dark": None,
        "window_size": (960, 640),
        "reconnect_timeout": 60,
        "language": LANG[0].replace('_', '-') if LANG else 'en-US',
    }
    try:
        import multiprocessing
        multiprocessing.freeze_support()

        with patch_runpy():
            ui.run(**ui_params)

    except KeyboardInterrupt:
        if getattr(sys, 'frozen', False):
            pass
        else:
            raise


if __name__ in {"__main__", "__mp_main__"}:
    main()
