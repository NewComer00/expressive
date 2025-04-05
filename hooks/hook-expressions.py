from PyInstaller.utils.hooks import collect_submodules

# This grabs everything in the expressions package (except __init__)
hiddenimports = collect_submodules('expressions')

