import os
from pathlib import Path

import nvidia.cuda_nvcc
import nvidia.cuda_runtime
import nvidia.cudnn
import nvidia.cublas
import nvidia.cusolver
import nvidia.cusparse
import nvidia.cufft
import nvidia.curand


def add_cuda11_to_path():
    """Add CUDA 11 to library searching path."""
    for package in [nvidia.cuda_nvcc, nvidia.cuda_runtime, nvidia.cudnn, nvidia.cublas,
                    nvidia.cusolver, nvidia.cusparse, nvidia.cufft, nvidia.curand]:
        lib_path = Path(package.__path__[0]) / "bin"
        if os.name == "nt":
            os.environ["PATH"] = \
                str(lib_path) + ';' + os.environ.get('PATH', '')
        else:
            os.environ["LD_LIBRARY_PATH"] = \
                str(lib_path) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
