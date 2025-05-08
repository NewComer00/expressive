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
