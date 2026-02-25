import importlib.util

from utils.gpu import CUDA_PACKAGES

hiddenimports = [pkg for pkg in CUDA_PACKAGES if importlib.util.find_spec(pkg) is not None]
