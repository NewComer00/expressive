# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

# Resolve project root regardless of where the spec file lives
# spec is at <root>/build/expressive.spec, so root is one level up
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))


def src(*parts):
    """Absolute path to a file/dir in the project root."""
    return os.path.join(ROOT, *parts)


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
_nicegui  = collect_data_files('nicegui')
_crepe    = collect_data_files('crepe')
_swift_f0 = collect_data_files('swift_f0')

# ---------------------------------------------------------------------------
# expressive cli
# ---------------------------------------------------------------------------
a_cli = Analysis(
    [src('expressive.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (src('examples'), 'examples/'),
        (src('assets'),   'assets/'),
        (src('locales'),  'locales/'),
        (src('README.md'), '.'),
        (src('LICENSE'),   '.'),
        *_crepe,
        *_swift_f0,
    ],
    hiddenimports=[],
    hookspath=[src('build/hooks/expressions'), src('build/hooks/gpu'), src('build/hooks/misc')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_cli = PYZ(a_cli.pure)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='expressive',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=src('assets/icons/cli.ico'),
    contents_directory='.',
)

# ---------------------------------------------------------------------------
# expressive-gui
# ---------------------------------------------------------------------------
a_gui = Analysis(
    [src('expressive_gui.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (src('examples'), 'examples/'),
        (src('assets'),   'assets/'),
        (src('locales'),  'locales/'),
        (src('static'),   'static/'),
        (src('README.md'), '.'),
        (src('LICENSE'),   '.'),
        *_crepe,
        *_nicegui,
        *_swift_f0,
    ],
    hiddenimports=[],
    hookspath=[src('build/hooks/expressions'), src('build/hooks/gpu'), src('build/hooks/misc')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_gui = PYZ(a_gui.pure)

splash_gui = Splash(
    src('assets/splash/big.png'),
    binaries=a_gui.binaries,
    datas=a_gui.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    splash_gui,
    [],
    exclude_binaries=True,
    name='expressive-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=src('assets/icons/gui.ico'),
    contents_directory='.',
)

# ---------------------------------------------------------------------------
# expressive-viewer
# ---------------------------------------------------------------------------
a_viewer = Analysis(
    [src('expressive_viewer.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (src('assets'),   'assets/'),
        (src('locales'),  'locales/'),
        (src('static'),   'static/'),
        (src('README.md'), '.'),
        (src('LICENSE'),   '.'),
        *_nicegui,
    ],
    hiddenimports=[],
    hookspath=[src('build/hooks/misc')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['sklearn', 'scikit-learn'],
    noarchive=False,
    optimize=0,
)
pyz_viewer = PYZ(a_viewer.pure)

exe_viewer = EXE(
    pyz_viewer,
    a_viewer.scripts,
    [],
    exclude_binaries=True,
    name='expressive-viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=src('assets/icons/viewer.ico'),
    contents_directory='.',
)

# ---------------------------------------------------------------------------
# Collect all three into one dist directory
# ---------------------------------------------------------------------------
COLLECT(
    exe_cli,    a_cli.binaries,    a_cli.datas,
    exe_gui,    a_gui.binaries,    a_gui.datas,    splash_gui.binaries,
    exe_viewer, a_viewer.binaries, a_viewer.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='expressive',
)
