import sys


# For jaraco.context, we need to include backports.tarfile for Python < 3.12
if sys.version_info < (3, 12):
    hiddenimports = ['backports.tarfile']