import os
import ctypes


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


import webview
from nicegui import ui, app
from typing import Callable, Optional
from webview.dom import DOMEventHandler


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
        super().__init__(tag='div', *args, **kwargs)

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
        if isinstance(js_api, JS_API):
            js_api.register_bind(self._html_id, self._bind)
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
