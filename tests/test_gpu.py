"""Tests for GPU utilities."""

import os
from unittest.mock import MagicMock, patch

import pytest

from utils.gpu import add_cuda_to_path


CUDA_PKGS = [
    "nvidia.cuda_nvcc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cublas",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.cufft",
    "nvidia.curand",
]


def make_mock_pkg(path_base: str):
    m = MagicMock()
    m.__path__ = [path_base]
    return m


@pytest.fixture(autouse=True)
def reset_cuda_state():
    """Reset the _cuda_added flag before each test."""
    with patch("utils.gpu._cuda_added", False):
        yield


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_add_cuda_to_path_windows():
    """Test adding CUDA to PATH on Windows using importlib import mock."""
    packages = {name: make_mock_pkg(f"C:\\path\\to\\{name}\\") for name in CUDA_PKGS}

    with patch.dict(os.environ, {"PATH": "existing_path"}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=lambda name: packages[name]):
            add_cuda_to_path()

            path = os.environ["PATH"]
            assert "existing_path" in path
            assert "bin" in path


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific test")
def test_add_cuda_to_path_posix():
    """Test adding CUDA to LD_LIBRARY_PATH on POSIX systems."""
    packages = {name: make_mock_pkg(f"/path/to/{name}/") for name in CUDA_PKGS}

    with patch.dict(os.environ, {"LD_LIBRARY_PATH": "existing_ld_path"}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=lambda name: packages[name]):
            add_cuda_to_path()

            ld_path = os.environ["LD_LIBRARY_PATH"]
            assert "existing_ld_path" in ld_path
            assert "bin" in ld_path


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_add_cuda_to_path_empty_env_windows():
    """Test adding CUDA when PATH is empty on Windows."""
    pkg = make_mock_pkg("C:\\path\\to\\cuda\\")

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", return_value=pkg):
            add_cuda_to_path()

            assert "PATH" in os.environ
            assert "bin" in os.environ["PATH"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific test")
def test_add_cuda_to_path_empty_env_posix():
    """Test adding CUDA when LD_LIBRARY_PATH is empty on POSIX."""
    pkg = make_mock_pkg("/path/to/cuda/")

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", return_value=pkg):
            add_cuda_to_path()

            assert "LD_LIBRARY_PATH" in os.environ
            assert "bin" in os.environ["LD_LIBRARY_PATH"]


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_add_cuda_to_path_multiple_packages_windows():
    """Test that all CUDA packages are added to PATH on Windows."""
    packages = {name: make_mock_pkg(f"C:\\unique\\path\\{name}\\") for name in CUDA_PKGS}

    with patch.dict(os.environ, {"PATH": ""}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=lambda name: packages[name]):
            add_cuda_to_path()

            path = os.environ["PATH"]
            for _name, pkg in packages.items():
                assert pkg.__path__[0] in path


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific test")
def test_add_cuda_to_path_multiple_packages_posix():
    """Test that all CUDA packages are added to LD_LIBRARY_PATH on POSIX."""
    packages = {name: make_mock_pkg(f"/unique/path/{name}/") for name in CUDA_PKGS}

    with patch.dict(os.environ, {"LD_LIBRARY_PATH": ""}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=lambda name: packages[name]):
            add_cuda_to_path()

            ld_path = os.environ["LD_LIBRARY_PATH"]
            for _name, pkg in packages.items():
                assert pkg.__path__[0] in ld_path


def test_add_cuda_to_path_skip_missing():
    """Test that a missing package is skipped with a warning when skip_missing=True."""
    missing_pkg = "nvidia.cudnn"

    def fake_import(name):
        if name == missing_pkg:
            raise ImportError(f"No module named '{name}'")
        return make_mock_pkg(f"/path/to/{name}/")

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=fake_import):
            with patch("utils.gpu.logger") as mock_logger:
                add_cuda_to_path(skip_missing=True)

                mock_logger.warning.assert_called_once()
                assert missing_pkg in mock_logger.warning.call_args[0][1]


def test_add_cuda_to_path_skip_missing_multiple():
    """Test that all missing packages are reported in a single warning."""
    missing_pkgs = {"nvidia.cudnn", "nvidia.curand"}

    def fake_import(name):
        if name in missing_pkgs:
            raise ImportError(f"No module named '{name}'")
        return make_mock_pkg(f"/path/to/{name}/")

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=fake_import):
            with patch("utils.gpu.logger") as mock_logger:
                add_cuda_to_path(skip_missing=True)

                mock_logger.warning.assert_called_once()
                warning_joined = mock_logger.warning.call_args[0][1]
                for pkg in missing_pkgs:
                    assert pkg in warning_joined


def test_add_cuda_to_path_raises_on_missing():
    """Test that ImportError is raised when skip_missing=False."""
    def fake_import(name):
        raise ImportError(f"No module named '{name}'")

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=fake_import):
            with pytest.raises(ImportError):
                add_cuda_to_path(skip_missing=False)


def test_add_cuda_to_path_no_warning_when_all_present():
    """Test that no warning is emitted when all packages are found."""
    packages = {name: make_mock_pkg(f"/path/to/{name}/") for name in CUDA_PKGS}

    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.gpu.importlib.import_module", side_effect=lambda name: packages[name]):
            with patch("utils.gpu.logger") as mock_logger:
                add_cuda_to_path(skip_missing=True)

                mock_logger.warning.assert_not_called()
