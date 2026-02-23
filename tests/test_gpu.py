"""Tests for GPU utilities."""

import os
from unittest.mock import MagicMock, patch

import pytest

from utils.gpu import add_cuda_to_path


class TestAddCudaToPath:
    """Test CUDA path addition."""

    @patch("utils.gpu.nvidia")
    def test_add_cuda_to_path_windows(self, mock_nvidia):
        """Test adding CUDA to PATH on Windows."""
        # Mock the nvidia packages
        mock_packages = []
        for pkg_name in ["cuda_nvcc", "cuda_runtime", "cudnn", "cublas",
                         "cusolver", "cusparse", "cufft", "curand"]:
            mock_pkg = MagicMock()
            mock_pkg.__path__ = [f"/path/to/{pkg_name}"]
            mock_packages.append(mock_pkg)

        # Patch os.name to simulate Windows
        with patch("os.name", "nt"):
            with patch.dict(os.environ, {"PATH": "existing_path"}, clear=True):
                # Patch the nvidia package imports
                with patch("utils.gpu.nvidia.cuda_nvcc", mock_packages[0]), \
                     patch("utils.gpu.nvidia.cuda_runtime", mock_packages[1]), \
                     patch("utils.gpu.nvidia.cudnn", mock_packages[2]), \
                     patch("utils.gpu.nvidia.cublas", mock_packages[3]), \
                     patch("utils.gpu.nvidia.cusolver", mock_packages[4]), \
                     patch("utils.gpu.nvidia.cusparse", mock_packages[5]), \
                     patch("utils.gpu.nvidia.cufft", mock_packages[6]), \
                     patch("utils.gpu.nvidia.curand", mock_packages[7]):

                    add_cuda_to_path()

                    # Verify PATH was updated
                    path = os.environ["PATH"]
                    assert "existing_path" in path
                    # Should contain bin directories
                    assert "bin" in path

    @pytest.mark.skipif(os.name == "nt", reason="Cannot test Linux paths on Windows")
    @patch("utils.gpu.nvidia")
    def test_add_cuda_to_path_linux(self, mock_nvidia):
        """Test adding CUDA to LD_LIBRARY_PATH on Linux."""
        # Mock the nvidia packages
        mock_packages = []
        for pkg_name in ["cuda_nvcc", "cuda_runtime", "cudnn", "cublas",
                         "cusolver", "cusparse", "cufft", "curand"]:
            mock_pkg = MagicMock()
            mock_pkg.__path__ = [f"/path/to/{pkg_name}"]
            mock_packages.append(mock_pkg)

        # Patch os.name to simulate Linux
        with patch("os.name", "posix"):
            with patch.dict(os.environ, {"LD_LIBRARY_PATH": "existing_ld_path"}, clear=True):
                # Patch the nvidia package imports
                with patch("utils.gpu.nvidia.cuda_nvcc", mock_packages[0]), \
                     patch("utils.gpu.nvidia.cuda_runtime", mock_packages[1]), \
                     patch("utils.gpu.nvidia.cudnn", mock_packages[2]), \
                     patch("utils.gpu.nvidia.cublas", mock_packages[3]), \
                     patch("utils.gpu.nvidia.cusolver", mock_packages[4]), \
                     patch("utils.gpu.nvidia.cusparse", mock_packages[5]), \
                     patch("utils.gpu.nvidia.cufft", mock_packages[6]), \
                     patch("utils.gpu.nvidia.curand", mock_packages[7]):

                    add_cuda_to_path()

                    # Verify LD_LIBRARY_PATH was updated
                    ld_path = os.environ["LD_LIBRARY_PATH"]
                    assert "existing_ld_path" in ld_path
                    # Should contain bin directories
                    assert "bin" in ld_path

    @patch("utils.gpu.nvidia")
    def test_add_cuda_to_path_empty_env(self, mock_nvidia):
        """Test adding CUDA when PATH/LD_LIBRARY_PATH is empty."""
        # Mock the nvidia packages
        mock_pkg = MagicMock()
        mock_pkg.__path__ = ["/path/to/cuda"]

        # Test on Windows with empty PATH
        with patch("os.name", "nt"):
            with patch.dict(os.environ, {}, clear=True):
                with patch("utils.gpu.nvidia.cuda_nvcc", mock_pkg), \
                     patch("utils.gpu.nvidia.cuda_runtime", mock_pkg), \
                     patch("utils.gpu.nvidia.cudnn", mock_pkg), \
                     patch("utils.gpu.nvidia.cublas", mock_pkg), \
                     patch("utils.gpu.nvidia.cusolver", mock_pkg), \
                     patch("utils.gpu.nvidia.cusparse", mock_pkg), \
                     patch("utils.gpu.nvidia.cufft", mock_pkg), \
                     patch("utils.gpu.nvidia.curand", mock_pkg):

                    add_cuda_to_path()

                    # Should create PATH with CUDA paths
                    assert "PATH" in os.environ
                    assert "bin" in os.environ["PATH"]

    @patch("utils.gpu.nvidia")
    def test_add_cuda_to_path_multiple_packages(self, mock_nvidia):
        """Test that all CUDA packages are added to path."""
        # Create distinct mock packages
        mock_packages = {}
        for pkg_name in ["cuda_nvcc", "cuda_runtime", "cudnn", "cublas",
                         "cusolver", "cusparse", "cufft", "curand"]:
            mock_pkg = MagicMock()
            mock_pkg.__path__ = [f"/unique/path/{pkg_name}"]
            mock_packages[pkg_name] = mock_pkg

        with patch("os.name", "nt"):
            with patch.dict(os.environ, {"PATH": ""}, clear=True):
                with patch("utils.gpu.nvidia.cuda_nvcc", mock_packages["cuda_nvcc"]), \
                     patch("utils.gpu.nvidia.cuda_runtime", mock_packages["cuda_runtime"]), \
                     patch("utils.gpu.nvidia.cudnn", mock_packages["cudnn"]), \
                     patch("utils.gpu.nvidia.cublas", mock_packages["cublas"]), \
                     patch("utils.gpu.nvidia.cusolver", mock_packages["cusolver"]), \
                     patch("utils.gpu.nvidia.cusparse", mock_packages["cusparse"]), \
                     patch("utils.gpu.nvidia.cufft", mock_packages["cufft"]), \
                     patch("utils.gpu.nvidia.curand", mock_packages["curand"]):

                    add_cuda_to_path()

                    path = os.environ["PATH"]
                    # Verify all packages were added
                    for pkg_name in mock_packages:
                        assert pkg_name in path
