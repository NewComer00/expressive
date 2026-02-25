import importlib.util

from utils.gpu import CUDA_PACKAGES


def _is_importable(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except ModuleNotFoundError:
        return False


hiddenimports = [pkg for pkg in CUDA_PACKAGES if _is_importable(pkg)]
