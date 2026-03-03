import os
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)

_cuda_added = False

CUDA_PACKAGES = [
    "nvidia.cuda_nvcc", "nvidia.cuda_runtime", "nvidia.cudnn", "nvidia.cublas",
    "nvidia.cusolver", "nvidia.cusparse", "nvidia.cufft", "nvidia.curand",
]


def add_cuda_to_path(skip_missing: bool = False):
    """Add CUDA to library searching path."""
    global _cuda_added
    if _cuda_added:
        return

    packages = CUDA_PACKAGES

    missing = []
    for package_name in packages:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            if skip_missing:
                missing.append(package_name)
                continue
            raise

        lib_path = Path(package.__path__[0]) / "bin"
        if os.name == "nt":
            os.environ["PATH"] = str(lib_path) + ';' + os.environ.get('PATH', '')
        else:
            os.environ["LD_LIBRARY_PATH"] = str(lib_path) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    if missing:
        logger.warning(
            "The following CUDA packages were not found and will be skipped: %s. "
            "If you are running the CPU-only version, you can safely ignore this message.",
            ", ".join(missing),
        )

    _cuda_added = True
