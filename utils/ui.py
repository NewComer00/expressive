import os
import json
import uuid
import ctypes
from typing import Any, Callable, Optional

import webview
from nicegui import ui, app
from markdown import markdown
from webview.dom import DOMEventHandler

from utils.i18n import _


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

app.add_static_files('/static', os.path.join(os.path.dirname(__file__), '..', 'static'))


# ---------------------------------------------------------------------------
# Windows-only helpers
# ---------------------------------------------------------------------------

if os.name == "nt":
    import win32gui
    import win32con
    import pywinstyles

    def get_hwnd_by_title(title):
        """Return the window handle for a visible window with matching title."""
        def callback(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                if title in win32gui.GetWindowText(hwnd):
                    result.append(hwnd)
        result = []
        win32gui.EnumWindows(callback, result)
        return result[0] if result else None


def change_titlebar_color(window_title: str, color="blue"):
    """Change the title bar color of the specified window."""
    if os.name == "nt":
        hwnd = get_hwnd_by_title(window_title)
        if hwnd:
            pywinstyles.change_header_color(hwnd, color=color)


def change_window_style(window_title: str, style: str):
    """Change the window style of the specified window."""
    if os.name == "nt":
        hwnd = get_hwnd_by_title(window_title)
        if hwnd:
            pywinstyles.apply_style(hwnd, style=style)


def blink_taskbar_window(window_title: str, count=5, timeout=0):
    """Blink the taskbar icon of a window with the specified title."""
    if os.name == "nt":
        def flash_window(hwnd, count=5, timeout=0):
            """Flash the specified window using Windows API."""
            class FLASHWINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', ctypes.c_uint),
                    ('hwnd', ctypes.c_void_p),
                    ('dwFlags', ctypes.c_uint),
                    ('uCount', ctypes.c_uint),
                    ('dwTimeout', ctypes.c_uint),
                ]
            fwi = FLASHWINFO()
            fwi.cbSize = ctypes.sizeof(FLASHWINFO)
            fwi.hwnd = hwnd
            fwi.dwFlags = win32con.FLASHW_ALL
            fwi.uCount = count
            fwi.dwTimeout = timeout
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(fwi))

        hwnd = get_hwnd_by_title(window_title)
        if hwnd:
            flash_window(hwnd, count, timeout)


# ---------------------------------------------------------------------------
# JS API bridge
# ---------------------------------------------------------------------------

class JS_API:
    """Allows JavaScript to call Python-side binding methods."""
    def __init__(self) -> None:
        self._bind_table: dict[str, Callable] = {}

    def register_bind(self, element_id: str, bind_method: Callable) -> None:
        self._bind_table[element_id] = bind_method

    def bind(self, element_id: str) -> None:
        if element_id not in self._bind_table:
            raise KeyError(f"No bind method registered for element_id '{element_id}'")
        self._bind_table[element_id]()


def webview_active_window():
    """Get the currently active pywebview window."""
    window = webview.active_window()
    if window:
        return window
    if webview.windows:
        return webview.windows[0]
    raise RuntimeError('No active window found.')


# ---------------------------------------------------------------------------
# NiceguiNativeDropArea
# ---------------------------------------------------------------------------

class NiceguiNativeDropArea(ui.element):
    """Drop area for native NiceGUI apps supporting full file paths via pywebview.

    NOTE: Do not use NiceGUI APIs in the on_* handlers; use pywebview's APIs instead.
    """

    def __init__(
        self,
        on_dragenter: Optional[Callable] = None,
        on_dragleave: Optional[Callable] = None,
        on_dragover: Optional[Callable] = None,
        on_drop: Optional[Callable] = None,
        *args, **kwargs
    ):
        super().__init__(tag='div', *args, **kwargs)  # noqa: B026

        self.on_dragenter = on_dragenter
        self.on_dragleave = on_dragleave
        self.on_dragover = on_dragover
        self.on_drop = on_drop

        self._html_id = f'c{self.id}'
        self._setup_js_api()
        self._inject_bind_script()

    def _bind(self) -> None:
        """Bind native drag-and-drop events via pywebview DOM API."""
        window = webview_active_window()
        element = window.dom.get_element(f'#{self._html_id}')
        if not element:
            raise RuntimeError(f"Element with ID '{self._html_id}' not found in the DOM.")
        if self.on_dragenter:
            element.events.dragenter += DOMEventHandler(self.on_dragenter, True, True)  # type: ignore
        if self.on_dragleave:
            element.events.dragleave += DOMEventHandler(self.on_dragleave, True, True)  # type: ignore
        if self.on_dragover:
            element.events.dragover += DOMEventHandler(self.on_dragover, True, True, debounce=500)  # type: ignore
        if self.on_drop:
            element.events.drop += DOMEventHandler(self.on_drop, True, True)  # type: ignore

    def _setup_js_api(self) -> None:
        """Ensure JS_API is registered and bind this element."""
        if 'js_api' not in app.native.window_args:
            app.native.window_args['js_api'] = JS_API()

        js_api = app.native.window_args['js_api']
        register = getattr(js_api, 'register_bind', None)
        if callable(register):
            register(self._html_id, self._bind)
        else:
            raise RuntimeError("Conflicting js_api already assigned to app.native.window_args['js_api'].")

    def _inject_bind_script(self) -> None:
        """Inject JavaScript to trigger binding after pywebview is ready."""
        ui.add_head_html(f'''
            <script>
            window.addEventListener('pywebviewready', function() {{
                window.pywebview.api.bind('{self._html_id}');
            }});
            </script>
        ''')


# ---------------------------------------------------------------------------
# WaveSurferElement
# ---------------------------------------------------------------------------

def _random_color_js() -> str:
    """Return a JS expression that produces a random rgba colour string."""
    return (
        "`rgba(${Math.floor(Math.random()*256)},"
        "${Math.floor(Math.random()*256)},"
        "${Math.floor(Math.random()*256)},0.5)`"
    )


class WaveSurferElement(ui.element):
    """NiceGUI element that embeds a WaveSurfer.js waveform with the Regions plugin.

    All WaveSurfer / Regions options are forwarded as plain JSON so the Python
    API stays thin and you can still reach any JS-side feature via
    ``self._js_client.run_javascript()``.

    Usage::

        ws = WaveSurferElement(url='/static/audio/track.wav')
        ws.add_region(start=2, end=8, content='Verse', drag=True, resize=True)
        ws.play()
    """

    _WAVESURFER_JS = "/static/vendor/wavesurfer/wavesurfer.esm.js"
    _REGIONS_JS    = "/static/vendor/wavesurfer/regions.esm.js"

    def __init__(
        self,
        url: str = "",
        *,
        wave_color: str = "rgb(200, 0, 200)",
        progress_color: str = "rgb(100, 0, 100)",
        height: int = 128,
        bar_width: Optional[int] = None,
        bar_gap: Optional[int] = None,
        bar_radius: Optional[int] = None,
        normalize: bool = True,
        interact: bool = True,
        loop_regions: bool = True,
        enable_drag_selection: bool = True,
        drag_selection_color: str = "rgba(255,0,0,0.15)",
        show_controls: bool = True,
        on_ready: Optional[Callable[[dict], Any]] = None,
        on_region_clicked: Optional[Callable[[dict], Any]] = None,
        on_region_updated: Optional[Callable[[dict], Any]] = None,
        on_region_created: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        super().__init__(tag="div")
        self._js_client = self.client  # capture at construction; valid even in background tasks

        self._iid = f"ws_{uuid.uuid4().hex[:8]}"
        self._url = url
        self._wave_color = wave_color
        self._progress_color = progress_color
        self._height = height
        self._bar_width = bar_width
        self._bar_gap = bar_gap
        self._bar_radius = bar_radius
        self._normalize = normalize
        self._interact = interact
        self._loop_regions = loop_regions
        self._enable_drag_selection = enable_drag_selection
        self._drag_selection_color = drag_selection_color
        self._show_controls = show_controls

        self._on_ready = on_ready
        self._on_region_clicked = on_region_clicked
        self._on_region_updated = on_region_updated
        self._on_region_created = on_region_created

        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, url: str) -> None:
        """Load a new audio file URL."""
        self._url = url
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.load({json.dumps(url)})")

    def empty(self) -> None:
        """Unload the audio file."""
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.empty()")

    def play(self) -> None:
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.play()")

    def pause(self) -> None:
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.pause()")

    def stop(self) -> None:
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.stop()")

    def seek(self, progress: float) -> None:
        """Seek to a position; progress is 0–1."""
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.seekTo({progress})")

    def zoom(self, min_px_per_sec: int) -> None:
        self._js_client.run_javascript(f"window['{self._iid}']?.ws.zoom({min_px_per_sec})")

    def set_loop(self, enabled: bool) -> None:
        self._js_client.run_javascript(f"if(window['{self._iid}']) window['{self._iid}'].loop={json.dumps(enabled)}")

    def add_region(
        self,
        start: float,
        end: Optional[float] = None,
        content: str = "",
        color: Optional[str] = None,
        drag: bool = True,
        resize: bool = True,
        min_length: Optional[float] = None,
        max_length: Optional[float] = None,
    ) -> None:
        """Add a region (or marker if end is None) to the waveform."""
        region: dict[str, Any] = {"start": start, "drag": drag, "resize": resize}
        if end is not None:
            region["end"] = end
        if content:
            region["content"] = content
        if min_length is not None:
            region["minLength"] = min_length
        if max_length is not None:
            region["maxLength"] = max_length

        js_color = json.dumps(color) if color else _random_color_js()
        js_obj = json.dumps(region)[:-1] + f', "color": {js_color}' + "}"
        self._js_client.run_javascript(f"window['{self._iid}']?.regions.addRegion({js_obj})")

    def clear_regions(self) -> None:
        self._js_client.run_javascript(f"window['{self._iid}']?.regions.clearRegions()")

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        iid = self._iid
        self.classes("w-full flex flex-col gap-2")

        with self:
            waveform_div = ui.element("div")
            waveform_div.props(f'id="{iid}_waveform"')
            waveform_div.classes("w-full rounded-lg overflow-hidden")
            waveform_div.style(f"min-height:{self._height}px; background:var(--ws-bg);")

        ws_opts: dict[str, Any] = {
            "container": f"#{iid}_waveform",
            "waveColor": self._wave_color,
            "progressColor": self._progress_color,
            "height": self._height,
            "normalize": self._normalize,
            "interact": self._interact,
        }
        if self._bar_width is not None:
            ws_opts["barWidth"] = self._bar_width
        if self._bar_gap is not None:
            ws_opts["barGap"] = self._bar_gap
        if self._bar_radius is not None:
            ws_opts["barRadius"] = self._bar_radius
        if self._url:
            ws_opts["url"] = self._url

        opts_json        = json.dumps(ws_opts)
        drag_color_json  = json.dumps(self._drag_selection_color)
        drag_enabled     = json.dumps(self._enable_drag_selection)
        loop_init          = json.dumps(self._loop_regions)
        show_controls_json = json.dumps(self._show_controls)

        play_helper = json.dumps(_("Play/Pause"))
        loop_helper = json.dumps(_("Loop region"))
        zoom_helper = json.dumps(_("Zoom"))

        ui.add_body_html(f"""
<script type="module">
import WaveSurfer from '{self._WAVESURFER_JS}';
import RegionsPlugin from '{self._REGIONS_JS}';

(function bootstrap() {{
  const waveformEl = document.getElementById('{iid}_waveform');
  if (!waveformEl) {{ setTimeout(bootstrap, 50); return; }}

  const regions = RegionsPlugin.create();
  const ws = WaveSurfer.create({{
    ...{opts_json},
    plugins: [regions],
  }});

  window['{iid}'] = {{ ws, regions, loop: {loop_init} }};

  if ({show_controls_json}) {{
    // Overlay controls: inject a style block + overlay div into the waveform container
    const styleEl = document.createElement('style');
    styleEl.textContent = `
    :root {{
        --ws-bg: #f5f5f5;
        --ws-overlay-grad: linear-gradient(to bottom, rgba(255,255,255,0.65) 0%, transparent 100%);
        --ws-btn-color: rgba(0,0,0,0.65);
        --ws-btn-hover: rgba(200,0,200,0.25);
        --ws-range-accent: #c800c8;
        --ws-range-opacity: 0.6;
    }}

    body.body--dark {{
        --ws-bg: #0d0d0d;
        --ws-overlay-grad: linear-gradient(to bottom, rgba(0,0,0,0.55) 0%, transparent 100%);
        --ws-btn-color: rgba(255,255,255,0.75);
        --ws-btn-hover: rgba(200,0,200,0.35);
        --ws-range-accent: #c800c8;
        --ws-range-opacity: 0.35;
    }}

    .ws-wrap-{iid} {{ position:relative; }}

    .ws-overlay-{iid} {{
        position:absolute;
        top:0; left:0; right:0;
        z-index:10;
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:3px 6px;
        background:var(--ws-overlay-grad);
        opacity:0;
        transition:opacity 0.15s ease;
        pointer-events:none;
    }}

    .ws-wrap-{iid}:hover .ws-overlay-{iid} {{
        opacity:1;
        pointer-events:auto;
    }}

    .ws-btn-{iid} {{
        background:none;
        border:none;
        color:var(--ws-btn-color);
        cursor:pointer;
        font-size:.85rem;
        line-height:1;
        padding:2px 4px;
        border-radius:4px;
        transition:color .1s, background .1s;
    }}

    .ws-btn-{iid}:hover {{
        background:var(--ws-btn-hover);
        color:#fff;
    }}

    .ws-range-{iid} {{
        accent-color: var(--ws-range-accent);
        width:120px;
        cursor:pointer;
        vertical-align:middle;
        opacity:var(--ws-range-opacity);
    }}

    .ws-range-{iid}:hover {{
        opacity:1;
    }}

    `;
    document.head.appendChild(styleEl);

    // Scrollbar lives inside WaveSurfer's wrapper shadow — inject there
    const scrollStyle = document.createElement('style');
    scrollStyle.textContent = `
      ::-webkit-scrollbar {{ height: 10px; }}
      ::-webkit-scrollbar-track {{ background: transparent; }}
      ::-webkit-scrollbar-thumb {{ background: rgba(150,150,150,0.35); border-radius: 6px; }}
      body.body--dark ::-webkit-scrollbar-thumb {{ background: rgba(200,200,200,0.25); }}
      ::-webkit-scrollbar-thumb:hover {{ background: #c800c8; }}
    `;
    ws.once('decode', () => ws.getWrapper().appendChild(scrollStyle));

    const wrap = waveformEl.parentElement;
    wrap.classList.add('ws-wrap-{iid}');
    wrap.style.position = 'relative';

    const overlay = document.createElement('div');
    overlay.className = 'ws-overlay-{iid}';
    overlay.innerHTML =
      '<div style="display:flex;gap:2px;align-items:center;">' +
        '<button id="{iid}_play" class="ws-btn-{iid}" title={play_helper}>&#9654;</button>' +
        '<button id="{iid}_loop" class="ws-btn-{iid}" title={loop_helper} style="font-size:1.0rem;opacity:0.85;">&#8635;</button>' +
      '</div>' +
      '<div style="display:flex;align-items:center;gap:3px;">' +
        '<span style="font-size:.8rem;color:rgba(255,255,255,0.6);">&#128269;</span>' +
        '<input type="range" id="{iid}_zoom" title={zoom_helper} min="10" max="1000" value="10" class="ws-range-{iid}"/>' +
      '</div>';
    wrap.appendChild(overlay);

    const playBtn = document.getElementById('{iid}_play');
    ws.on('play',  () => {{ playBtn.innerHTML = '&#9208;'; }});
    ws.on('pause', () => {{ playBtn.innerHTML = '&#9654;'; }});
    playBtn.addEventListener('click', (e) => {{ e.stopPropagation(); ws.playPause(); }});

    const loopBtn = document.getElementById('{iid}_loop');
    let loopOn = {loop_init};
    const updateLoop = () => {{
      loopBtn.style.opacity = loopOn ? '1' : '0.35';
      loopBtn.style.color   = loopOn ? '#c800c8' : '';
      window['{iid}'].loop  = loopOn;
    }};
    updateLoop();
    loopBtn.addEventListener('click', (e) => {{ e.stopPropagation(); loopOn = !loopOn; updateLoop(); }});

    const zoomSlider = document.getElementById('{iid}_zoom');
    zoomSlider.addEventListener('input', () => ws.zoom(Number(zoomSlider.value)));
    ws.on('zoom', (minPxPerSec) => {{ zoomSlider.value = minPxPerSec; }});
  }}

  if ({drag_enabled}) {{
    regions.enableDragSelection({{ color: {drag_color_json} }});
  }}

  // Region loop logic
  let activeRegion = null;
  regions.on('region-in',  (r) => {{ activeRegion = r; }});
  regions.on('region-out', (r) => {{
    if (activeRegion === r) {{
      if (window['{iid}'].loop) {{ r.play(); }}
      else {{ activeRegion = null; }}
    }}
  }});
  regions.on('region-clicked', (r, e) => {{
    e.stopPropagation();
    activeRegion = r;
    r.play(true);
  }});
  ws.on('interaction', () => {{ activeRegion = null; }});


}})();
</script>
""")  # noqa: E501


# ---------------------------------------------------------------------------
# WaveSurferRangeSelector
# ---------------------------------------------------------------------------

def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to 'm:ss.ss' format matching ExpressionLoader timestamp style."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


def serve_wav(wav_path: str) -> str:
    """Serve a local WAV file via NiceGUI's static file server and return its URL.

    Each unique directory is registered once under /wav/<hashed_dir>.
    WaveSurfer then streams the file normally — no base64 overhead.
    """
    import hashlib
    directory = os.path.dirname(os.path.abspath(wav_path))
    dir_hash  = hashlib.md5(directory.encode()).hexdigest()[:8]
    mount     = f"/wav/{dir_hash}"
    # Register the directory only once
    if not any(r.path == mount for r in app.routes):
        app.add_static_files(mount, directory)
    filename = os.path.basename(wav_path)
    return f"{mount}/{filename}"


class WaveSurferRangeSelector(ui.element):
    """WaveSurferElement with NiceGUI-style BindableProperty bindings.

    - ``wav_path``: one-way in via ``bind_wav_path_from(obj, key)`` — reloads on change.
    - ``start`` / ``end``: two-way via ``bind_start`` / ``bind_end`` — region drag
      writes back to the bound object in 'm:ss.ss' format.

    Usage::

        ws = WaveSurferRangeSelector()
        ws.bind_wav_path_from(state, 'ref_wav')
        ws.bind_start(state, 'ref_start')
        ws.bind_end(state, 'ref_end')
    """

    from nicegui.binding import BindableProperty

    wav_path = BindableProperty(
        on_change=lambda self, val: self._on_wav_path_change(val))  # type: ignore[misc]
    start    = BindableProperty(
        on_change=lambda self, val: self._on_range_change())  # type: ignore[misc]
    end      = BindableProperty(
        on_change=lambda self, val: self._on_range_change())  # type: ignore[misc]

    def __init__(
        self,
        *,
        wave_color: str = "rgb(200, 0, 200)",
        progress_color: str = "rgb(100, 0, 100)",
        height: int = 80,
        bar_width: int = 2,
        bar_gap: int = 1,
        bar_radius: int = 2,
    ) -> None:
        super().__init__(tag="div")
        self.wav_path: str = ""
        self.start: str    = ""
        self.end: str      = ""
        self._from_js: bool      = False  # guard against JS→Python→JS round-trip

        self.classes("w-full")

        with self:
            self._ws = WaveSurferElement(
                wave_color=wave_color,
                progress_color=progress_color,
                height=height,
                bar_width=bar_width,
                bar_gap=bar_gap,
                bar_radius=bar_radius,
                loop_regions=False,
                enable_drag_selection=True,
                drag_selection_color="rgba(100,180,255,0.25)",
                show_controls=True,
            )

        iid = self._ws._iid
        ui.add_body_html(f"""
<script>
(function waitForWs() {{
  const inst = window['{iid}'];
  if (!inst) {{ setTimeout(waitForWs, 80); return; }}

  function attachRegionListeners() {{
    function fmt(sec) {{
      const m = Math.floor(sec / 60);
      const s = sec - m * 60;
      return m + ':' + s.toFixed(2).padStart(5, '0');
    }}

    function emitRange(r) {{
      if (inst._updatingFromPython) return;
      inst.regions.getRegions().forEach(other => {{
        if (other.id !== r.id) other.remove();
      }});
      emitEvent('wavesurfer-range-{iid}', {{ start: fmt(r.start), end: fmt(r.end) }});
    }}

    inst.regions.on('region-created', emitRange);
    inst.regions.on('region-updated', emitRange);
  }}

  if (inst.ws.getDuration() > 0) {{ attachRegionListeners(); }}
  else {{ inst.ws.once('ready', attachRegionListeners); }}
  
  inst.ws.on('ready', (duration) => {{
    emitEvent('{iid}-ready', {{ duration: duration }});
  }});
}})();
</script>
""")
        ui.on(f'wavesurfer-range-{iid}', self._handle_region_updated)
        ui.on(f"{iid}-ready", self._on_range_change)

    def _handle_region_updated(self, e) -> None:
        """Called by JS on region-updated; writes back through BindableProperty."""
        self._from_js = True
        self.start = e.args['start'] or None
        self.end   = e.args['end']   or None
        self._from_js = False

    @staticmethod
    def _timestamp_to_seconds(ts: str | None) -> float | None:
        """Parse 'm:ss.ss' → seconds, or bare float string → seconds. Returns None if invalid/empty."""
        if not ts:
            return None
        try:
            if ':' in ts:
                m, s = ts.split(':', 1)
                return int(m) * 60 + float(s)
            return float(ts)
        except ValueError:
            return None

    def _on_range_change(self) -> None:
        """Called by BindableProperty when start or end changes from the Python side."""
        if self._from_js:
            return  # already came from JS, don't echo back
        # Guard: binding may fire before NiceGUI's event loop is ready (e.g. at startup)
        from nicegui import core
        if core.loop is None:
            return

        iid = self._ws._iid
        start_s = self._timestamp_to_seconds(self.start or '')
        end_s   = self._timestamp_to_seconds(self.end   or '')

        # Invalid parse (non-empty but unparseable) or start >= end → clear
        start_invalid = bool((self.start or '').strip()) and start_s is None
        end_invalid   = bool((self.end   or '').strip()) and end_s   is None
        if start_invalid or end_invalid or (
            start_s is not None and end_s is not None and start_s >= end_s
        ):
            self._ws._js_client.run_javascript(
                f"window['{self._ws._iid}']?.regions.clearRegions()"
            )
            return

        # Empty start = beginning (0), empty end = track duration
        start_js = start_s if start_s is not None else 0
        end_js   = f'{end_s}' if end_s is not None else 'inst.ws.getDuration()'
        self._ws._js_client.run_javascript(f"""
            (function applyRegion() {{
                const inst = window['{iid}'];
                if (!inst) {{ setTimeout(applyRegion, 80); return; }}
                const apply = () => {{
                    const start = {start_js};
                    const end   = {end_js};
                    if (start >= end) return;
                    inst._updatingFromPython = true;
                    const regions = inst.regions.getRegions();
                    if (regions.length > 0) {{
                        regions[0].setOptions({{ start, end }});
                    }} else {{
                        inst.regions.addRegion({{
                            start, end,
                            color: 'rgba(100,180,255,0.25)',
                            drag: true, resize: true,
                        }});
                    }}
                    inst._updatingFromPython = false;
                }};
                if (inst.ws.getDuration() > 0.001) {{ apply(); }}
            }})();
        """)

    def _on_wav_path_change(self, path: str) -> None:
        """Called automatically by BindableProperty when wav_path changes.
        Clears the waveform, then loads the new file.
        Re-renders the region once the waveform is ready.
        """
        # Guard: binding may fire before NiceGUI's event loop is ready (e.g. at startup)
        from nicegui import core
        if core.loop is None:
            return
        if path:
            self._ws.empty()
            self._ws.clear_regions()
            self._ws.zoom(0)
            self._ws.load(serve_wav(path))

    def bind_wav_path_from(self, target_object: Any, target_name: str = 'wav_path') -> "WaveSurferRangeSelector":
        """One-way bind: target → wav_path."""
        from nicegui.binding import bind_from
        bind_from(self, 'wav_path', target_object, target_name)
        return self

    def bind_start(self, target_object: Any, target_name: str = 'start') -> "WaveSurferRangeSelector":
        """Two-way bind: start ↔ target."""
        from nicegui.binding import bind
        bind(self, 'start', target_object, target_name)
        return self

    def bind_end(self, target_object: Any, target_name: str = 'end') -> "WaveSurferRangeSelector":
        """Two-way bind: end ↔ target."""
        from nicegui.binding import bind
        bind(self, 'end', target_object, target_name)
        return self


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def tooltip_md(element: ui.element, text: str) -> ui.element:
    """Add a markdown-rendered tooltip to a NiceGUI element. Chainable like .tooltip()."""
    with element:
        with ui.tooltip():
            ui.html(markdown(str(text)))
    return element
